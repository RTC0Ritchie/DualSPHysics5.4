#include "JSphCpu.h"
#include "JCellSearch_inline.h"
#include "FunSphKernel.h"
#include "FunctionsMath.h"
#include "UserParameters.h"
#include "JSphCpuSingle.h"
#include "JBinaryData.h"
#include "JLog2.h"
#include "Functions.h"
#include "UserFlexBoundaryCorrection.h"
#include "UserFlexInline.h"

#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <limits>
#include <memory>
#include <exception>

namespace {

  inline tfloat3 FlexMV(
      const tmatrix3f& a,
      const tfloat3& b)
  {
    return TFloat3(
        a.a11*b.x+a.a12*b.y+a.a13*b.z,
        a.a21*b.x+a.a22*b.y+a.a23*b.z,
        a.a31*b.x+a.a32*b.y+a.a33*b.z);
  }

  inline tfloat3 FlexScale(
      const tfloat3& a,
      float s)
  {
    return TFloat3(a.x*s,a.y*s,a.z*s);
  }

  // a -= r tensor q
  inline void FlexSubtractOuter(
      tmatrix3f& a,
      const tfloat3& r,
      const tfloat3& q)
  {
    a.a11-=r.x*q.x;
    a.a12-=r.x*q.y;
    a.a13-=r.x*q.z;

    a.a21-=r.y*q.x;
    a.a22-=r.y*q.y;
    a.a23-=r.y*q.z;

    a.a31-=r.z*q.x;
    a.a32-=r.z*q.y;
    a.a33-=r.z*q.z;
  }
}

namespace {
  namespace FlexRestartCorrection {

  inline bool ForceCacheReady(
      const UserParameters::StFlexRestartState& state,
      bool useGradient,
      bool useHourglass);

  inline bool GradientCacheReady(
      const UserParameters::StFlexRestartState& state,
      unsigned np);

  template<bool simulate2d>
  inline tmatrix3f DefGrad(
      const UserParameters::StFlexRestartState& state,
      unsigned pfs1,
      const tdouble3* pos,
      const unsigned* flexstrucridp);

  inline void AddElastic(
      const UserParameters::StFlexRestartState& state,
      size_t k,
      unsigned pfs1,
      unsigned pfs2,
      const tmatrix3f* flexPk1Reference,
      float mass0p1,
      tfloat3& acep1);

  template<TpKernel tker>
  inline void AddHourglass(
      const UserParameters::StFlexRestartState& state,
      size_t k,
      const tmatrix3f& defgradp1,
      const tmatrix3f& defgradp2,
      const tfloat3& r,
      float rr2,
      const StCteSph& CSP,
      float hgfactor,
      float youngmod,
      float mass0p1,
      tfloat3& acep1);

  }
}

using namespace std;

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> void JSphCpu::InteractionForcesFlexStruc(unsigned np,float visco
    ,StDivDataCpu divdata,const unsigned* dcell
    ,const tdouble3* pos,const tfloat4* velrhop,const float* press,const typecode* code
    ,const byte* boundmode,const tfloat3* tangenvel
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const tdouble3* pos0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr,const tmatrix3f* defgrad
    ,float& flexstrucdt,tfloat3* ace)const{
  //-Initialise flexstructh to calculate max flexstrucdt.
  float flexstructh[OMP_MAXTHREADS*OMP_STRIDE];
  for(int th=0;th<OmpThreads;th++)flexstructh[th*OMP_STRIDE]=0;
  //-Starts execution using OpenMP.
  const int n=int(np);

  // --------------------------------------------------------------------------------------
  // Load user-defined parameters
  const UserParameters& User = *this->User;
  const int FlexBoundContactMode=User.FlexBoundContactMode;

  const int SurfaceType = (FlexBoundContactMode == 2) ? 
            User.SurfaceType.value : 0;

  double WallZ_tmp = 0.0;
  if(SurfaceType==1) WallZ_tmp = User.SurfaceType.WallZ;
  const double WallZ = WallZ_tmp;

  double w1x_tmp = 0.0, w1z_tmp = 0.0, w2x_tmp = 1e-12, w2z_tmp = 0.0;
  if(SurfaceType == 2){
    w1x_tmp = User.SurfaceType.point1.x;
    w1z_tmp = User.SurfaceType.point1.z;
    w2x_tmp = User.SurfaceType.point2.x;
    w2z_tmp = User.SurfaceType.point2.z;
  }
  const double w1x = w1x_tmp, w1z = w1z_tmp, w2x = w2x_tmp, w2z = w2z_tmp;
  const double wallSlope = (w2z-w1z)/(w2x-w1x);

  const int SurfaceNormalDirectionMode = 
          (SurfaceType == 2) ? User.SurfaceType.NormalDirectionMode : 0;

  double wallNx_tmp=0.0, wallNz_tmp=1.0;
  if(SurfaceNormalDirectionMode==2){
  const double invLength=1.0/sqrt(1.0+wallSlope*wallSlope);
  wallNx_tmp=-wallSlope*invLength;
  wallNz_tmp=invLength;
  }
  const double wallNx=wallNx_tmp, wallNz=wallNz_tmp;

  const int KnContactMode=User.KnContactMode;
  const double KnContactConst=User.KnContact;
  const int KnContactLinearity=User.KnContactLinearity;

  const int unContactMode=User.unContactMode;
  const double unContactConst=User.unContact;
  // --------------------------------------------------------------------------------------

  // Prepare constitutive quantities once per flexible particle.
  // Includes clamp particles, which can appear as neighbours.
  const bool isRestart=User.IsFlexRestart();
  const auto& state=User.FlexRestartState;

  if(isRestart &&
    (!state.HasRestartStateLayout() || state.Idp.size()!=np))
    Run_Exceptioon("FlexRestart: restart state is unavailable.");

  const bool useBoundary=
      (User.IsFlexPrescribedRestart() ||
       User.IsFlexModelTransferRestart()) &&
      User.FlexRestartBoundaryCorrectionMode==5;
  const bool useGradient=
      isRestart && User.UseFlexRestartGradientCorrection;

  const bool useHourglass=
      isRestart && User.UseFlexRestartHourglassCorrection;

  if(!FlexRestartCorrection::ForceCacheReady(state,useGradient,useHourglass))
    Run_Exceptioon("Flexible correction cache is unavailable.");

  std::vector<tmatrix3f> flexPk1Reference(
      (useGradient || useBoundary) ? np : 0);

  std::vector<tmatrix3f> flexForceMatrix(np);
  std::vector<float> flexDensity(np);

  for(unsigned pfs=0;pfs<np;pfs++){
    const unsigned p=flexstrucridp[pfs];
    const unsigned id=Idp_c->cptr()[p];

    const StFlexStrucData& material=
        flexstrucdata[CODE_GetIbodyFlexStruc(code[p])];

    const tmatrix3f& frelative=defgrad[pfs];
    const float jrelative=
        fmath::Determinant3x3(frelative);

    if(!std::isfinite(jrelative) || jrelative<=0.f)
      Run_Exceptioon("Invalid relative deformation determinant.");

    if(!isRestart){
      // Ordinary simulation: use original material fields directly.
      const tmatrix3f pk1=
          CalcFlexStrucPK1Stress(frelative,material.cmat);

      flexDensity[pfs]=material.rho0/jrelative;

      // Keep the current force assembly convention.
      flexForceMatrix[pfs]=ScaleFlexMatrix(
          fmath::MulMatrix3x3(pk1,kercorr[pfs]),
          material.vol0*material.vol0);
    }
    else{
      // Verify stable particle identity.
      const tmatrix3f& fstar=
          User.GetFlexRestartFstar(pfs,id);

      const float jstar=state.Jstar[pfs];

      // Total deformation relative to Xn.
      const tmatrix3f ftotal=
          fmath::MulMatrix3x3(frelative,fstar);

      const float jtotal=fmath::Determinant3x3(ftotal);
      if(!std::isfinite(jtotal) || jtotal<=0.f)
        Run_Exceptioon("Invalid total deformation determinant.");

      // Constitutive parameters still come from the original material.
      const tmatrix3f pnatural=
          CalcFlexStrucPK1Stress(ftotal,material.cmat);

      // PK1 relative to this run's X0.
      const tmatrix3f preference=ScaleFlexMatrix(
          fmath::MulMatrix3x3(
              pnatural,fmath::TrasMatrix3x3(fstar)),
          1.f/jstar);
      if(useGradient || useBoundary)
        flexPk1Reference[pfs]=preference;

      // Current density from the X0 reference density.
      flexDensity[pfs]=state.ReferenceDensity[pfs]/jrelative;

      /// Restart: store P_t/0 * B_0 without volume weights.
      // Vi*Vj is applied in the particle-pair loop.
      flexForceMatrix[pfs]=
          useBoundary
          ? preference
          : BuildFlexRestartBoundaryForceMatrix(
              pfs,CODE_GetIbodyFlexStruc(code[p]),
              frelative,preference,kercorr[pfs]);
    }
  }
  std::vector<tdouble3> boundaryForce;

  if(useBoundary){
    const auto referenceKernel=[&](double rr)->double {
      return fsph::GetKernel_Wab<tker>(CSP,float(rr));
    };

    FlexBoundaryCorrection::ElasticForceForSolver(
        User,np,pos0,pos,code,flexstrucdata,flexstrucridp,
        defgrad,flexPk1Reference.data(),referenceKernel,boundaryForce);
  }
  // --------------------------------------------------------------------------------------

  //-Starts execution using OpenMP.
  #ifdef OMP_USE
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p=0;p<n;p++){
    const int pfs1=p;      //-Number of particle.
    const unsigned p1=flexstrucridp[pfs1];

    //-Get codep1.
    const typecode codep1=code[p1];
    if(CODE_IsFlexStrucFlex(codep1)){
      tfloat3 acep1=TFloat3(0);

      //-Obtains basic data of particle p1.
      const tdouble3 posp1=pos[p1];
      tfloat4 velrhop1=velrhop[p1];
      const float pressp1=press[p1];
      const tdouble3 pos0p1=pos0[pfs1];
      const tmatrix3f defgradp1=defgrad[pfs1];

      //-Obtains flexible structure data.
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].vol0;
      const float rho0p1=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].rho0;
      const float youngmod=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].youngmod;
      const float poissratio=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].poissratio;
      const float hgfactor=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].hgfactor;
      // Ordinary: includes volume weights.
      // Restart: stores P_t/0 * B_0; pair volumes are applied below.
      const tmatrix3f pk1kercorrp1=flexForceMatrix[pfs1];

      //-Get current mass of flexible structure particle.
      const float mass0p1=
        isRestart
        ? state.ParticleMass[pfs1]
        : vol0p1*rho0p1;
      if(useBoundary){
        const tdouble3& force=boundaryForce[pfs1];

        acep1.x+=float(force.x/mass0p1);
        acep1.y+=float(force.y/mass0p1);
        acep1.z+=float(force.z/mass0p1);
      }
      const float rhop1=flexDensity[pfs1];

      //-Calculate structural speed of sound.
      const float csp1=float( sqrt(youngmod*(1.0-poissratio)/(rhop1*(1.0+poissratio)*(1.0-2.0*poissratio))) );

      //-Modify values if using mDBC.
      float massp2=MassFluid;
      if(mdbc2>=MDBC2_Std){
        if(boundmode[p1]==BMODE_MDBC2OFF)massp2=0;

        //-Use reconstructed velocity only when available.
        //-Otherwise keep the original particle velocity.
        if(boundmode[p1]!=BMODE_DBC){
          const tfloat3 tangentvelp1=tangenvel[p1];
          velrhop1.x=tangentvelp1.x;
          velrhop1.y=tangentvelp1.y;
          velrhop1.z=tangentvelp1.z;
        }
      }

      //-Obtains neighborhood search limits.
      const StNgSearch ngs=nsearch::Init(dcell[p1],false,divdata);

      //-Flexible structure-Fluid interaction.
      for(int z=ngs.zini;z<ngs.zfin;z++){
        for(int y=ngs.yini;y<ngs.yfin;y++){
          const tuint2 pif=nsearch::ParticleRange(y,z,ngs,divdata);
          for(unsigned p2=pif.x;p2<pif.y;p2++){
            if(CODE_IsFluid(code[p2])){
              const tdouble3 posp2=pos[p2];
              const float drx=float(posp1.x-posp2.x);
              const float dry=float(posp1.y-posp2.y);
              const float drz=float(posp1.z-posp2.z);
              const float rr2=drx*drx+dry*dry+drz*drz;
              if(rr2<=KernelSize2&&rr2>=ALMOSTZERO){
                //-Computes kernel.
                const float fac=fsph::GetKernel_Fac<tker>(CSP,rr2);
                const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
                //-Obtains velocity of particle p2 and compute difference.
                const tfloat4 velrhop2=velrhop[p2];
                const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
                //-Pressure derivative (Momentum equation).
                const float pressp2=press[p2];
                const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)+(tker==KERNEL_Cubic?fsph::GetKernelCubic_Tensil(CSP,rr2,velrhop1.w,pressp1,velrhop2.w,pressp2):0);
                const float p_vpm=-prs*massp2*(MassFluid/mass0p1);
                acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;
                //-Artificial viscosity.
                if(tvisco==VISCO_Artificial){
                  const float dot=drx*dvx+dry*dvy+drz*dvz;
                  if(dot<0){
                    const float dot_rr2=dot/(rr2+Eta2);
                    const float amubar=KernelH*dot_rr2;
                    const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                    const float pi_visc=(float)(-visco*Cs0*amubar/robar)*massp2*(MassFluid/mass0p1);
                    acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
                  }
                }
                  //-Laminar viscosity.
                else{
                  const float robar2=(velrhop1.w+velrhop2.w);
                  const float temp=4.f*visco/((rr2+Eta2)*robar2);
                  const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz)*(MassFluid/mass0p1);
                  acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
                }
              }
            }
          }
        }
      }

      // flexure-bound
      const float KnContact = (KnContactMode==1) ? float(KnContactConst) : 
                  2.f*youngmod*sqrt(float(Dp))/(3.f*(1.f-poissratio*poissratio));
      const float unContact = (unContactMode==1) ? float(unContactConst) : 
                  1.4e-5f * sqrt(6*mass0p1*youngmod*sqrt(float(Dp)/2)); 
                  
      if (FlexBoundContactMode==1){ // flexure-bound particle interaction
        const StNgSearch ngsb=nsearch::Init(dcell[p1],true,divdata);
        for(int z=ngsb.zini;z<ngsb.zfin;z++){
          for(int y=ngsb.yini;y<ngsb.yfin;y++){
            const tuint2 pib=nsearch::ParticleRange(y,z,ngsb,divdata);

            for(unsigned p2=pib.x;p2<pib.y;p2++){
              const typecode codep2=code[p2];

              if(CODE_IsFixed(codep2) && !CODE_IsFlexStrucAny(codep2)){

                const tdouble3 posp2=pos[p2];
                const float drx=float(posp1.x-posp2.x);
                const float dry=float(posp1.y-posp2.y);
                const float drz=float(posp1.z-posp2.z);
                const float rr2=drx*drx+dry*dry+drz*drz;
                
                if(rr2<=KernelSize2&&rr2>=ALMOSTZERO){
                  const float rad = sqrt(rr2);
                  const float overlap=float(Dp)-rad;

                  if(overlap>0.f){
                    // interaction force flexure-bound
                    const float erx=drx/rad;
                    const float ery=dry/rad;
                    const float erz=drz/rad;
                    const float vn=  velrhop[p1].x*erx
                                    + velrhop[p1].y*ery
                                    + velrhop[p1].z*erz;

                    const float fnE = (KnContactLinearity==1)? 
                                      KnContact * overlap : // String model
                                      KnContact * overlap*sqrt(overlap); // Full contact model
                    const float fnD = -unContact*sqrt(sqrt(overlap))*vn;
                    const float fn_tmp = fnE + fnD;
                    const float fn = (fn_tmp>0.f) ? fn_tmp : 0.f; // Only repulsive force

                    const float aceContact=fn/mass0p1;

                    acep1.x+=aceContact*erx;
                    acep1.y+=aceContact*ery;
                    acep1.z+=aceContact*erz;
                  }
                }
              }
            }
          }
        }
      }
      else if (FlexBoundContactMode==2){ // flexure-bound wall interaction
        const double particleRadius=0.5*double(Dp);
        const double particleBottomZ=posp1.z-particleRadius;
        
        if((SurfaceType==1) || 
            (SurfaceType==2 && SurfaceNormalDirectionMode==1)){
          const double localZ=(SurfaceType==1) ? WallZ : w1z+wallSlope*(posp1.x-w1x);

          const double overlap=localZ-particleBottomZ;

          if(overlap>0.0){
            const double fnE=(KnContactLinearity==1)
                ? double(KnContact)*overlap
                : double(KnContact)*overlap*sqrt(overlap);

            const double vn=double(velrhop[p1].z);
            const double fnD = -double(unContact)*sqrt(sqrt(overlap))*vn;
            const double fn_tmp = fnE + fnD;
            const double fn = (fn_tmp>0.0) ? fn_tmp : 0.0;

            acep1.z+=float(fn/double(mass0p1));
          }
        }

        else if(SurfaceType==2 && SurfaceNormalDirectionMode==2){
          const double signedDistance=
          (posp1.x-w1x)*wallNx + (posp1.z-w1z)*wallNz;
          const double overlap=particleRadius-signedDistance;

          if(overlap>0.0){
            const double fnE=(KnContactLinearity==1)
                ? double(KnContact)*overlap
                : double(KnContact)*overlap*sqrt(overlap);

            const double vn=double(velrhop[p1].x)*wallNx + double(velrhop[p1].z)*wallNz;
            const double fnD = -double(unContact)*sqrt(sqrt(overlap))*vn;
            const double fn_tmp = fnE + fnD;
            const double fn = (fn_tmp>0.0) ? fn_tmp : 0.0;

            acep1.x+=float(fn*wallNx/double(mass0p1));
            acep1.z+=float(fn*wallNz/double(mass0p1));
          }
        }
      }

      //-Loop through pairs and calculate forces.
      for(unsigned pair=0;pair<numpairs[pfs1];pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        const tdouble3 pos0p2=pos0[pfs2];
        const float drx0=float(pos0p1.x-pos0p2.x);
        const float dry0=float(pos0p1.y-pos0p2.y);
        const float drz0=float(pos0p1.z-pos0p2.z);
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        const float fac0=fsph::GetKernel_Fac<tker>(CSP,rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        //-Acceleration due to structure.
        const tmatrix3f defgradp2=defgrad[pfs2];
        if(useBoundary){
          // Already assembled once through the matched energy adjoint.
        }
        else if(useGradient){
          FlexRestartCorrection::AddElastic(state,state.CorrectionOffset[pfs1]+pair,pfs1,pfs2,flexPk1Reference.data(),mass0p1,acep1);
        }
        else{
          const tmatrix3f pk1kercorrp2=flexForceMatrix[pfs2];
          tmatrix3f pk1kercorrp1p2;
          pk1kercorrp1p2.a11=pk1kercorrp1.a11+pk1kercorrp2.a11; pk1kercorrp1p2.a12=pk1kercorrp1.a12+pk1kercorrp2.a12; pk1kercorrp1p2.a13=pk1kercorrp1.a13+pk1kercorrp2.a13;
          pk1kercorrp1p2.a21=pk1kercorrp1.a21+pk1kercorrp2.a21; pk1kercorrp1p2.a22=pk1kercorrp1.a22+pk1kercorrp2.a22; pk1kercorrp1p2.a23=pk1kercorrp1.a23+pk1kercorrp2.a23;
          pk1kercorrp1p2.a31=pk1kercorrp1.a31+pk1kercorrp2.a31; pk1kercorrp1p2.a32=pk1kercorrp1.a32+pk1kercorrp2.a32; pk1kercorrp1p2.a33=pk1kercorrp1.a33+pk1kercorrp2.a33;
          tfloat3 pk1kercorrdw;
          pk1kercorrdw.x=pk1kercorrp1p2.a11*frx0+pk1kercorrp1p2.a12*fry0+pk1kercorrp1p2.a13*frz0;
          pk1kercorrdw.y=pk1kercorrp1p2.a21*frx0+pk1kercorrp1p2.a22*fry0+pk1kercorrp1p2.a23*frz0;
          pk1kercorrdw.z=pk1kercorrp1p2.a31*frx0+pk1kercorrp1p2.a32*fry0+pk1kercorrp1p2.a33*frz0;
          if(isRestart){
            const float pairVolume=
                state.ReferenceVolume[pfs1]*state.ReferenceVolume[pfs2];

            const float forceToAcceleration=pairVolume/mass0p1;

            acep1.x+=pk1kercorrdw.x*forceToAcceleration;
            acep1.y+=pk1kercorrdw.y*forceToAcceleration;
            acep1.z+=pk1kercorrdw.z*forceToAcceleration;
          }
          else{
            // Ordinary branch already includes the original volume weights.
            acep1.x+=pk1kercorrdw.x/mass0p1;
            acep1.y+=pk1kercorrdw.y/mass0p1;
            acep1.z+=pk1kercorrdw.z/mass0p1;
          }
        }
        
        //-Hourglass correction.
        if(hgfactor && !useBoundary){
          const unsigned p2=flexstrucridp[pfs2];
          const tdouble3 posp2=pos[p2];
          const float drx=float(posp1.x-posp2.x);
          const float dry=float(posp1.y-posp2.y);
          const float drz=float(posp1.z-posp2.z);
          const float rr2=drx*drx+dry*dry+drz*drz;
          if(useHourglass){
            FlexRestartCorrection::AddHourglass<tker>(state,state.CorrectionOffset[pfs1]+pair,defgradp1,defgradp2,
              TFloat3(drx,dry,drz),rr2,CSP,hgfactor,youngmod,mass0p1,acep1);
          }
          else{
            const float wab=fsph::GetKernel_Wab<tker>(CSP,rr2);
            const float xijbarx=defgradp1.a11*drx0+defgradp1.a12*dry0+defgradp1.a13*drz0;
            const float xijbary=defgradp1.a21*drx0+defgradp1.a22*dry0+defgradp1.a23*drz0;
            const float xijbarz=defgradp1.a31*drx0+defgradp1.a32*dry0+defgradp1.a33*drz0;
            const float xjibarx=-(defgradp2.a11*drx0+defgradp2.a12*dry0+defgradp2.a13*drz0);
            const float xjibary=-(defgradp2.a21*drx0+defgradp2.a22*dry0+defgradp2.a23*drz0);
            const float xjibarz=-(defgradp2.a31*drx0+defgradp2.a32*dry0+defgradp2.a33*drz0);
            const tfloat3 epsij=TFloat3(drx-xijbarx,dry-xijbary,drz-xijbarz);
            const tfloat3 epsji=TFloat3(-drx-xjibarx,-dry-xjibary,-drz-xjibarz);
            const float rr=sqrt(rr2);
            const float deltaij=(epsij.x*drx+epsij.y*dry+epsij.z*drz)/rr;
            const float deltaji=-(epsji.x*drx+epsji.y*dry+epsji.z*drz)/rr;
            const float hgPairVolume = isRestart
              ? state.ReferenceVolume[pfs1]*state.ReferenceVolume[pfs2]
              : vol0p1*vol0p1;
            const float mulfac = hgfactor*hgPairVolume*wab*youngmod/(rr20*rr*mass0p1)*0.5f*(deltaij+deltaji);
            acep1.x-=mulfac*drx; acep1.y-=mulfac*dry; acep1.z-=mulfac*drz;
          }
          
        }
      }

      //-Store results.
      if(acep1.x||acep1.y||acep1.z||csp1){
        ace[p1]=ace[p1]+acep1;
        const int th=omp_get_thread_num();
        if(csp1>flexstructh[th*OMP_STRIDE])flexstructh[th*OMP_STRIDE]=csp1;
      }
    }
  }
  //-Keep max value in flexstrucdt.
  for(int th=0;th<OmpThreads;th++)if(flexstrucdt<flexstructh[th*OMP_STRIDE])flexstrucdt=flexstructh[th*OMP_STRIDE];
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> void JSphCpu::Interaction_ForcesFlexStrucT(float& flexstrucdtmax)const{
  if(CaseNflexstruc){
    InteractionForcesFlexStruc<tker,tvisco,mdbc2>
      (CaseNflexstruc,Visco*ViscoBoundFactor,DivData,Dcell_c->cptr(),Pos_c->cptr(),Velrho_c->cptr(),Press_c->cptr(),Code_c->cptr(),AC_CPTR(BoundMode_c),AC_CPTR(TangenVel_c),FlexStrucDatac,FlexStrucRidpc,Pos0c,NumPairsc,PairIdxc,KerCorrc,DefGradc,flexstrucdtmax,Ace_c->ptr());
  }
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco> void JSphCpu::Interaction_ForcesFlexStruc_ct1(float& flexstrucdtmax)const{
       if(TMdbc2==MDBC2_None )Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_None >(flexstrucdtmax);
  else if(TMdbc2==MDBC2_Std  )Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_Std  >(flexstrucdtmax);
  else if(TMdbc2==MDBC2_NoPen)Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_NoPen>(flexstrucdtmax);
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker> void JSphCpu::Interaction_ForcesFlexStruc_ct0(float& flexstrucdtmax)const{
       if(TVisco==VISCO_Artificial)Interaction_ForcesFlexStruc_ct1<tker,VISCO_Artificial>(flexstrucdtmax);
  else if(TVisco==VISCO_Laminar   )Interaction_ForcesFlexStruc_ct1<tker,VISCO_Laminar   >(flexstrucdtmax);
  else if(TVisco==VISCO_LaminarSPS)Interaction_ForcesFlexStruc_ct1<tker,VISCO_LaminarSPS>(flexstrucdtmax);
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
void JSphCpu::Interaction_ForcesFlexStruc(float& flexstrucdtmax)const{
  if(TKernel==KERNEL_Wendland)  Interaction_ForcesFlexStruc_ct0<KERNEL_Wendland>(flexstrucdtmax);
  else if(TKernel==KERNEL_Cubic)Interaction_ForcesFlexStruc_ct0<KERNEL_Cubic>   (flexstrucdtmax);
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> void JSphCpu::CalcFlexStrucKerCorr(unsigned np,const typecode* code,const StFlexStrucData* flexstrucdata
        ,const unsigned* flexstrucridp,const tdouble3* pos0,const unsigned* numpairs,const unsigned* const* pairidx
        ,tmatrix3f* kercorr)const
{ 
  const bool isRestart=User->IsFlexRestart();
  const auto& state=User->FlexRestartState;

  if(isRestart &&
    (!state.HasRestartStateLayout() || state.Idp.size()!=np))
    Run_Exceptioon("FlexRestart: restart state is unavailable.");

  //-Starts execution using OpenMP.
  const int n=int(np);
  #ifdef OMP_USE
    #pragma omp parallel for schedule (static) if(n>OMP_LIMIT_COMPUTESTEP)
  #endif
  for(int p=0;p<n;p++){
    const unsigned pfs1=p;      //-Number of particle.

    //-Get number of pairs for this particle.
    const unsigned numpairsp1=numpairs[pfs1];

    //-If this particle has pairs.
    if(numpairsp1){
      tmatrix3f kercorrp1={0};

      //-Obtains basic data of particle p1.
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(code[flexstrucridp[pfs1]])].vol0;
      const tdouble3 pos0p1=pos0[pfs1];

      //-Calculate kernel correction matrix.
      for(unsigned pair=0;pair<numpairsp1;pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        // Restart: neighbour volume in X0.
        // Ordinary simulation: preserve the original central-volume weighting.
        const float pairWeightVolume=
            isRestart ? state.ReferenceVolume[pfs2] : vol0p1;
        const tdouble3 pos0p2=pos0[pfs2];
        const float drx0=float(pos0p1.x-pos0p2.x);
        const float dry0=float(pos0p1.y-pos0p2.y);
        const float drz0=float(pos0p1.z-pos0p2.z);
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        //-Computes kernel.
        const float fac0=fsph::GetKernel_Fac<tker>(CSP,rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        kercorrp1.a11-=pairWeightVolume*drx0*frx0;
        kercorrp1.a12-=pairWeightVolume*drx0*fry0;
        kercorrp1.a13-=pairWeightVolume*drx0*frz0;

        kercorrp1.a21-=pairWeightVolume*dry0*frx0;
        kercorrp1.a22-=pairWeightVolume*dry0*fry0;
        kercorrp1.a23-=pairWeightVolume*dry0*frz0;

        kercorrp1.a31-=pairWeightVolume*drz0*frx0;
        kercorrp1.a32-=pairWeightVolume*drz0*fry0;
        kercorrp1.a33-=pairWeightVolume*drz0*frz0;
      }
      if(simulate2d){
        kercorrp1.a12=kercorrp1.a21=kercorrp1.a23=kercorrp1.a32=0.0;
        kercorrp1.a22=1.0;
      }
      kercorr[pfs1]=fmath::InverseMatrix3x3(kercorrp1);
    }
  }
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> void JSphCpu::CalcFlexStrucKerCorrT()const{
  if(CaseNflexstruc){
    CalcFlexStrucKerCorr<tker,simulate2d>
      (CaseNflexstruc,Code_c->cptr(),FlexStrucDatac,FlexStrucRidpc,Pos0c,NumPairsc,PairIdxc,KerCorrc);
  }
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker> void JSphCpu::CalcFlexStrucKerCorr_ct0()const{
  if(Simulate2D)CalcFlexStrucKerCorrT<tker,true> ();
  else          CalcFlexStrucKerCorrT<tker,false>();
}

//==============================================================================
/// Calculates the kernel correction matrix for each flexible structure particle.
/// Calcula la matriz de corrección del kernel para cada partícula de estructura flexible.
//==============================================================================
void JSphCpu::CalcFlexStrucKerCorr()const{
  if(TKernel==KERNEL_Wendland)  CalcFlexStrucKerCorr_ct0<KERNEL_Wendland>();
  else if(TKernel==KERNEL_Cubic)CalcFlexStrucKerCorr_ct0<KERNEL_Cubic>   ();
}

//==============================================================================
/// Calculates the deformation gradient matrix for each flexible structure particle.
/// Calcula la matriz de gradiente de deformación para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> void JSphCpu::CalcFlexStrucDefGrad(unsigned np,const tdouble3* pos,const typecode* code
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const tdouble3* pos0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr
    ,tmatrix3f* defgrad)const
{
  const bool isRestart=User->IsFlexRestart();
  const auto& state=User->FlexRestartState;

  if(isRestart &&
    (!state.HasRestartStateLayout() || state.Idp.size()!=np))
    Run_Exceptioon("FlexRestart: restart state is unavailable.");

  const bool useGradient=
    isRestart && User->UseFlexRestartGradientCorrection;

  if(useGradient && !FlexRestartCorrection::GradientCacheReady(state,np))
    Run_Exceptioon("Flexible gradient cache is unavailable.");

  const bool useBoundary=
      (User->IsFlexPrescribedRestart() ||
       User->IsFlexModelTransferRestart()) &&
      User->FlexRestartBoundaryCorrectionMode==5;

  if(useBoundary){
    FlexBoundaryCorrection::DefGradForSolver(
        *User,np,Npb,pos,code,flexstrucridp,defgrad);
    return;
  }
  //-Starts execution using OpenMP.
  const int n=int(np);
  #ifdef OMP_USE
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p=0;p<n;p++){
    const int pfs1=p;      //-Number of particle.

    //-Get number of pairs for this particle.
    const unsigned numpairsp1=numpairs[pfs1];

    if(useGradient){
      defgrad[pfs1]=FlexRestartCorrection::DefGrad<simulate2d>(
          state,pfs1,pos,flexstrucridp);

      continue;
    }

    //-If this particle has pairs.
    if(numpairsp1){
      tmatrix3f defgradp1={0};

      //-Obtains basic data of particle p1.
      const unsigned p1=flexstrucridp[pfs1];
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(code[p1])].vol0;
      const tdouble3 posp1=pos[p1];
      const tdouble3 pos0p1=pos0[pfs1];
      const tmatrix3f kercorrp1=kercorr[pfs1];

      //-Calculate deformation gradient.
      for(unsigned pair=0;pair<numpairsp1;pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        // Restart: neighbour volume in X0.
        // Ordinary simulation: preserve the original central-volume weighting.
        const float pairWeightVolume=
            isRestart ? state.ReferenceVolume[pfs2] : vol0p1;
        const unsigned p2=flexstrucridp[pfs2];
        const tdouble3 posp2=pos[p2];
        const tdouble3 pos0p2=pos0[pfs2];
        const float drx=float(posp1.x-posp2.x);
        const float dry=float(posp1.y-posp2.y);
        const float drz=float(posp1.z-posp2.z);
        const float drx0=float(pos0p1.x-pos0p2.x);
        const float dry0=float(pos0p1.y-pos0p2.y);
        const float drz0=float(pos0p1.z-pos0p2.z);
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        const float fac0=fsph::GetKernel_Fac<tker>(CSP,rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        defgradp1.a11-=pairWeightVolume*drx*frx0;
        defgradp1.a12-=pairWeightVolume*drx*fry0;
        defgradp1.a13-=pairWeightVolume*drx*frz0;

        defgradp1.a21-=pairWeightVolume*dry*frx0;
        defgradp1.a22-=pairWeightVolume*dry*fry0;
        defgradp1.a23-=pairWeightVolume*dry*frz0;

        defgradp1.a31-=pairWeightVolume*drz*frx0;
        defgradp1.a32-=pairWeightVolume*drz*fry0;
        defgradp1.a33-=pairWeightVolume*drz*frz0;
      }
      if(simulate2d){
        defgradp1.a12=defgradp1.a21=defgradp1.a23=defgradp1.a32=0.0;
        defgradp1.a22=1.0;
      }
      defgrad[pfs1]=fmath::MulMatrix3x3(defgradp1,kercorrp1);
    }
  }
}

//==============================================================================
/// Calculates the deformation gradient matrix for each flexible structure particle.
/// Calcula la matriz de gradiente de deformación para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> void JSphCpu::CalcFlexStrucDefGradT()const{
  if(CaseNflexstruc){
    CalcFlexStrucDefGrad<tker,simulate2d>
      (CaseNflexstruc,Pos_c->cptr(),Code_c->cptr(),FlexStrucDatac,FlexStrucRidpc,Pos0c,NumPairsc,PairIdxc,KerCorrc,DefGradc);
  }
}

//==============================================================================
/// Calculates the deformation gradient matrix for each flexible structure particle.
/// Calcula la matriz de gradiente de deformación para cada partícula de estructura flexible.
//==============================================================================
template<TpKernel tker> void JSphCpu::CalcFlexStrucDefGrad_ct0()const{
  if(Simulate2D)CalcFlexStrucDefGradT<tker,true> ();
  else          CalcFlexStrucDefGradT<tker,false>();
}


//==============================================================================
/// Updates the geometric information for each flexible structure particle.
/// Actualiza la información geométrica de cada partícula de estructura flexible.
//==============================================================================
void JSphCpu::UpdateFlexStrucGeometry()const{
  if(TKernel==KERNEL_Wendland)  CalcFlexStrucDefGrad_ct0<KERNEL_Wendland>();
  else if(TKernel==KERNEL_Cubic)CalcFlexStrucDefGrad_ct0<KERNEL_Cubic>   ();
  if(CaseNflexstruc&&TMdbc2>=MDBC2_Std)CalcFlexStrucNormals();
}



namespace {
  namespace FlexRestartCorrection {

  //------------------------------------------------------------------------------
  // Checks the caches required by the optional force corrections.
  //------------------------------------------------------------------------------
  inline bool ForceCacheReady(
      const UserParameters::StFlexRestartState& state,
      bool useGradient,
      bool useHourglass)
  {
    const size_t count=state.CorrectionNeighbour.size();

    if(useGradient){
      if(!state.GradientReady ||
        state.Grad0I.size()!=count ||
        state.Grad0J.size()!=count)
      {
        return false;
      }
    }

    if(useHourglass){
      if(!state.HourglassReady ||
        state.HgWeight.size()!=count ||
        state.HgMappedI.size()!=count ||
        state.HgMappedJ.size()!=count)
      {
        return false;
      }
    }

    return true;
  }

  //------------------------------------------------------------------------------
  // Checks the cache used to compute the relative deformation gradient.
  //------------------------------------------------------------------------------
  inline bool GradientCacheReady(
      const UserParameters::StFlexRestartState& state,
      unsigned np)
  {
    return state.GradientReady &&
        state.CorrectionOffset.size()==size_t(np)+1 &&
        state.Grad0I.size()==state.CorrectionNeighbour.size();
  }

  //------------------------------------------------------------------------------
  // Computes F_t/0 using the transported discrete gradient operator.
  // The cached coefficient already includes the kernel correction.
  //------------------------------------------------------------------------------
  template<bool simulate2d>
  inline tmatrix3f DefGrad(
      const UserParameters::StFlexRestartState& state,
      unsigned pfs1,
      const tdouble3* pos,
      const unsigned* flexstrucridp)
  {
    tmatrix3f f={0};

    const tdouble3 a=pos[flexstrucridp[pfs1]];

    for(size_t k=state.CorrectionOffset[pfs1];
        k<state.CorrectionOffset[pfs1+1];k++)
    {
      const unsigned j=state.CorrectionNeighbour[k];
      const tdouble3 b=pos[flexstrucridp[j]];

      const tfloat3 r=TFloat3(
          float(a.x-b.x),
          float(a.y-b.y),
          float(a.z-b.z));

      FlexSubtractOuter(f,r,state.Grad0I[k]);
    }

    if(simulate2d){
      f.a12=f.a21=f.a23=f.a32=0.f;
      f.a22=1.f;
    }

    return f;
  }

  //------------------------------------------------------------------------------
  // Adds elastic acceleration using the transported gradient coefficients.
  // flexPk1Reference contains P_t/0 without kernel or volume factors.
  //------------------------------------------------------------------------------
  inline void AddElastic(
      const UserParameters::StFlexRestartState& state,
      size_t k,
      unsigned pfs1,
      unsigned pfs2,
      const tmatrix3f* flexPk1Reference,
      float mass0p1,
      tfloat3& acep1)
  {
    const tfloat3 fi=FlexMV(
        flexPk1Reference[pfs1],
        state.Grad0I[k]);

    const tfloat3 fj=FlexMV(
        flexPk1Reference[pfs2],
        state.Grad0J[k]);

    const float vi=state.ReferenceVolume[pfs1];
    const float vj=state.ReferenceVolume[pfs2];

    acep1.x+=(vi*fi.x+vj*fj.x)/mass0p1;
    acep1.y+=(vi*fi.y+vj*fj.y)/mass0p1;
    acep1.z+=(vi*fi.z+vj*fj.z)/mass0p1;
  }

  //------------------------------------------------------------------------------
  // Adds HG acceleration with historical residuals and reference weights.
  // Both cached mapped vectors use the same i-minus-j direction.
  //------------------------------------------------------------------------------
  template<TpKernel tker>
  inline void AddHourglass(
      const UserParameters::StFlexRestartState& state,
      size_t k,
      const tmatrix3f& defgradp1,
      const tmatrix3f& defgradp2,
      const tfloat3& r,
      float rr2,
      const StCteSph& CSP,
      float hgfactor,
      float youngmod,
      float mass0p1,
      tfloat3& acep1)
  {
    const float drx=r.x;
    const float dry=r.y;
    const float drz=r.z;

    const tfloat3 bi=FlexMV(
        defgradp1,
        state.HgMappedI[k]);

    const tfloat3 bj=FlexMV(
        defgradp2,
        state.HgMappedJ[k]);

    if(rr2>0.f){
      const float rr=sqrt(rr2);

      // Average projected residual:
      // r - F_t/0,i F*_i Rn
      // r - F_t/0,j F*_j Rn
      const float delta=
          0.5f*(
            (2.f*drx-bi.x-bj.x)*drx+
            (2.f*dry-bi.y-bj.y)*dry+
            (2.f*drz-bi.z-bj.z)*drz
          )/rr;

      // Keep the existing current-distance kernel evaluation.
      const float wab=
          fsph::GetKernel_Wab<tker>(CSP,rr2);

      const float mulfac=
          hgfactor*youngmod*wab*
          state.HgWeight[k]*delta/
          (rr*mass0p1);

      acep1.x-=mulfac*drx;
      acep1.y-=mulfac*dry;
      acep1.z-=mulfac*drz;
    }
  }

  } // namespace FlexRestartCorrection
} // anonymous namespace