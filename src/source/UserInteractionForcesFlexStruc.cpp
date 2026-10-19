#include "JSphCpu.h"
#include "JCellSearch_inline.h"
#include "FunSphKernel.h"
#include "FunctionsMath.h"
#include "UserParameters.h"
#include "JSphCpuSingle.h"
#include "JBinaryData.h"
#include "JLog2.h"
#include "Functions.h"

#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <limits>
#include <memory>

namespace {
  tmatrix3f ScaleFlexMatrix(tmatrix3f m,float s){
    m.a11*=s; m.a12*=s; m.a13*=s;
    m.a21*=s; m.a22*=s; m.a23*=s;
    m.a31*=s; m.a32*=s; m.a33*=s;
    return m;
  }

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
/// Calculates the PK1 stress matrix for each flexible structure particle.
/// Calcula la matriz de tensión PK1 para cada partícula de estructura flexible.
//==============================================================================
inline tmatrix3f JSphCpu::CalcFlexStrucPK1Stress(const tmatrix3f& defgrad,const tmatrix6f& cmat)const{
  //-Calculate Green-Lagrange strain from deformation gradient.
  tmatrix3f gl=fmath::MulMatrix3x3(fmath::TrasMatrix3x3(defgrad),defgrad);
  gl.a11-=1.0f; gl.a22-=1.0f; gl.a33-=1.0f;
  gl.a11*=0.5f; gl.a12*=0.5f; gl.a13*=0.5f;
  gl.a21*=0.5f; gl.a22*=0.5f; gl.a23*=0.5f;
  gl.a31*=0.5f; gl.a32*=0.5f; gl.a33*=0.5f;
  //-Convert to Voigt notation.
  const float gl_1=gl.a11;
  const float gl_2=gl.a22;
  const float gl_3=gl.a33;
  const float gl_4=gl.a23+gl.a32;
  const float gl_5=gl.a31+gl.a13;
  const float gl_6=gl.a12+gl.a21;
  //-Multiply with stiffness tensor to get PK2 stress in Voigt form.
  const float pk2_1=cmat.a11*gl_1+cmat.a12*gl_2+cmat.a13*gl_3+cmat.a14*gl_4+cmat.a15*gl_5+cmat.a16*gl_6;
  const float pk2_2=cmat.a21*gl_1+cmat.a22*gl_2+cmat.a23*gl_3+cmat.a24*gl_4+cmat.a25*gl_5+cmat.a26*gl_6;
  const float pk2_3=cmat.a31*gl_1+cmat.a32*gl_2+cmat.a33*gl_3+cmat.a34*gl_4+cmat.a35*gl_5+cmat.a36*gl_6;
  const float pk2_4=cmat.a41*gl_1+cmat.a42*gl_2+cmat.a43*gl_3+cmat.a44*gl_4+cmat.a45*gl_5+cmat.a46*gl_6;
  const float pk2_5=cmat.a51*gl_1+cmat.a52*gl_2+cmat.a53*gl_3+cmat.a54*gl_4+cmat.a55*gl_5+cmat.a56*gl_6;
  const float pk2_6=cmat.a61*gl_1+cmat.a62*gl_2+cmat.a63*gl_3+cmat.a64*gl_4+cmat.a65*gl_5+cmat.a66*gl_6;
  //-Convert PK2 stress back to normal notation.
  const tmatrix3f pk2={pk2_1,pk2_6,pk2_5,pk2_6,pk2_2,pk2_4,pk2_5,pk2_4,pk2_3};
  //-Convert PK2 to PK1 and return.
  return fmath::MulMatrix3x3(defgrad,pk2);
}

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

  const bool useGradient=
      isRestart && User.UseFlexRestartGradientCorrection;

  const bool useHourglass=
      isRestart && User.UseFlexRestartHourglassCorrection;

  if(!FlexRestartCorrection::ForceCacheReady(state,useGradient,useHourglass))
    Run_Exceptioon("Flexible correction cache is unavailable.");

  std::vector<tmatrix3f> flexPk1Reference(
      useGradient ? np : 0);

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
      if(useGradient) flexPk1Reference[pfs]=preference;

      // Current density from the X0 reference density.
      flexDensity[pfs]=state.ReferenceDensity[pfs]/jrelative;

      /// Restart: store P_t/0 * B_0 without volume weights.
      // Vi*Vj is applied in the particle-pair loop.
      flexForceMatrix[pfs]=BuildFlexRestartBoundaryForceMatrix(
          pfs,CODE_GetIbodyFlexStruc(code[p]),
          frelative,preference,kercorr[pfs]);
    }
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
        if(useGradient){
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
        if(hgfactor){
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

//============================================================================
/// Correct the boundary forces for flexible structure particles.
//=============================================================================
void JSphCpu::InitFlexRestartBoundaryCorrection()
{
  auto& data=User->FlexRestartBoundaryData;
  data=UserParameters::StFlexRestartBoundaryData{};

  if(!User->IsFlexPrescribedRestart() &&
     !User->IsFlexModelTransferRestart())return;

  const int mode=User->FlexRestartBoundaryCorrectionMode;
  if(mode==0)return;

  if(mode<1 || mode>4)
    Run_Exceptioon("Invalid restart boundary correction mode.");

  if(User->UseFlexRestartGradientCorrection)
    Run_Exceptioon(
        "Boundary correction cannot use transported direct-continuation gradients.");

  const unsigned np=CaseNflexstruc;
  if(!np || !Pos0c || !KerCorrc || !NumPairsc || !PairIdxc)
    Run_Exceptioon(
        "Boundary correction: reference geometry is unavailable.");

  data.Body.resize(np);
  data.Selected.assign(np,1);

  for(unsigned i=0;i<np;i++){
    data.Body[i]=CODE_GetIbodyFlexStruc(
        Code_c->cptr()[FlexStrucRidpc[i]]);

    if(data.Body[i]>=FlexStrucCount)
      Run_Exceptioon(
          "Boundary correction: invalid elastic body.");
  }

  const int layers=User->FlexRestartBoundaryCorrectionLayers;

  if(layers>0){
    // Peel geometric shells in X0, independently for each elastic body.
    // A missing angular sector identifies an exposed particle.
    const double cutoff2=1.8*1.8*Dp*Dp;
    const double pi=std::acos(-1.0);
    const double angleTolerance=1.e-10;

    std::vector<std::vector<unsigned>> neighbours(np);
    for(unsigned i=0;i<np;i++){
      for(unsigned k=0;k<NumPairsc[i];k++){
        const unsigned j=PairIdxc[i][k];
        if(j>=np || j==i || data.Body[i]!=data.Body[j])continue;

        const tdouble3 r=Pos0c[j]-Pos0c[i];
        const double rr2=r.x*r.x+r.z*r.z+
            (Simulate2D ? 0.0 : r.y*r.y);
        if(rr2>0.0 && rr2<=cutoff2)
          neighbours[i].push_back(j);
      }
    }

    // 98 approximately distributed directions for the 3D empty-cone test.
    std::vector<tdouble3> probeDirections;
    if(!Simulate2D){
      for(int x=-2;x<=2;x++)
        for(int y=-2;y<=2;y++)
          for(int z=-2;z<=2;z++){
            if(std::max(std::abs(x),
                std::max(std::abs(y),std::abs(z)))!=2)continue;
            const double length=std::sqrt(double(x*x+y*y+z*z));
            probeDirections.push_back(
                TDouble3(x/length,y/length,z/length));
          }
    }

    data.Selected.assign(np,0);
    std::vector<byte> queued(np,0);
    std::vector<unsigned> candidates(np),shell,next;
    for(unsigned i=0;i<np;i++)candidates[i]=i;

    std::vector<double> angles;
    std::vector<tdouble3> directions;
    angles.reserve(64);
    directions.reserve(64);

    for(int layer=0;layer<layers && !candidates.empty();layer++){
      shell.clear();

      // Evaluate the complete shell before removing any of its particles.
      for(const unsigned i:candidates){
        angles.clear();
        directions.clear();

        for(const unsigned j:neighbours[i]){
          if(data.Selected[j])continue;
          const tdouble3 r=Pos0c[j]-Pos0c[i];

          if(Simulate2D)angles.push_back(std::atan2(r.z,r.x));
          else{
            const double length=std::sqrt(
                r.x*r.x+r.y*r.y+r.z*r.z);
            directions.push_back(
                TDouble3(r.x/length,r.y/length,r.z/length));
          }
        }

        bool exposed=false;
        if(Simulate2D){
          if(angles.size()<2)exposed=true;
          else{
            std::sort(angles.begin(),angles.end());
            double gap=angles.front()+2.0*pi-angles.back();
            for(size_t k=1;k<angles.size();k++)
              gap=std::max(gap,angles[k]-angles[k-1]);

            // A >=120 degree empty sector tolerates sloped/staircase edges.
            exposed=(gap>=2.0*pi/3.0-angleTolerance);
          }
        }
        else{
          // An empty cone with a 60 degree half-angle marks a 3D surface.
          for(const tdouble3& probe:probeDirections){
            bool occupied=false;
            for(const tdouble3& direction:directions){
              const double dot=probe.x*direction.x+
                  probe.y*direction.y+probe.z*direction.z;
              if(dot>=0.5-angleTolerance){
                occupied=true;
                break;
              }
            }
            if(!occupied){
              exposed=true;
              break;
            }
          }
        }

        if(exposed)shell.push_back(i);
      }

      if(shell.empty())break;
      for(const unsigned i:shell)data.Selected[i]=1;

      // Only neighbours of the removed shell can become newly exposed.
      next.clear();
      for(const unsigned i:shell){
        for(const unsigned j:neighbours[i]){
          if(!data.Selected[j] && !queued[j]){
            queued[j]=1;
            next.push_back(j);
          }
        }
      }
      for(const unsigned i:next)queued[i]=0;
      candidates.swap(next);
    }
  }

  const auto capKernel=[&](const tmatrix3f& input)->tmatrix3f
  {
    double a[3][3]={
      {input.a11,
       0.5*(double(input.a12)+input.a21),
       0.5*(double(input.a13)+input.a31)},
      {0.5*(double(input.a21)+input.a12),
       input.a22,
       0.5*(double(input.a23)+input.a32)},
      {0.5*(double(input.a31)+input.a13),
       0.5*(double(input.a32)+input.a23),
       input.a33}
    };

    double v[3][3]={{1,0,0},{0,1,0},{0,0,1}};

    for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
        if(!std::isfinite(a[i][j]))
          Run_Exceptioon(
              "Boundary correction: non-finite kernel matrix.");

    // Symmetric Jacobi eigensolver; valid for 2D and 3D.
    for(int iter=0;iter<32;iter++){
      int p=0,q=1;

      if(std::fabs(a[0][2])>std::fabs(a[p][q])){
        p=0;
        q=2;
      }
      if(std::fabs(a[1][2])>std::fabs(a[p][q])){
        p=1;
        q=2;
      }

      const double scale=std::max(1.0,
          std::max(std::fabs(a[0][0]),
          std::max(std::fabs(a[1][1]),std::fabs(a[2][2]))));

      if(std::fabs(a[p][q])<=1.e-12*scale)break;

      const double angle=0.5*std::atan2(
          2.0*a[p][q],a[q][q]-a[p][p]);

      const double c=std::cos(angle),s=std::sin(angle);
      const double app=a[p][p],aqq=a[q][q],apq=a[p][q];

      for(int k=0;k<3;k++)if(k!=p && k!=q){
        const double akp=a[k][p],akq=a[k][q];
        a[k][p]=a[p][k]=c*akp-s*akq;
        a[k][q]=a[q][k]=s*akp+c*akq;
      }

      a[p][p]=c*c*app-2.0*c*s*apq+s*s*aqq;
      a[q][q]=s*s*app+2.0*c*s*apq+c*c*aqq;
      a[p][q]=a[q][p]=0.0;

      for(int k=0;k<3;k++){
        const double vkp=v[k][p],vkq=v[k][q];
        v[k][p]=c*vkp-s*vkq;
        v[k][q]=s*vkp+c*vkq;
      }
    }

    bool changed=false;
    const double limit=User->FlexRestartBoundaryCorrectionLimit;

    // Upper clipping only; smaller eigenvalues are retained.
    for(int k=0;k<3;k++){
      if(!std::isfinite(a[k][k]))
        Run_Exceptioon(
            "Boundary correction: non-finite kernel eigenvalue.");

      if(a[k][k]>limit){
        a[k][k]=limit;
        changed=true;
      }
    }

    if(!changed)return input;

    double b[3][3]={};

    for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)
          b[i][j]+=v[i][k]*a[k][k]*v[j][k];

    return TMatrix3f(
        float(b[0][0]),float(b[0][1]),float(b[0][2]),
        float(b[1][0]),float(b[1][1]),float(b[1][2]),
        float(b[2][0]),float(b[2][1]),float(b[2][2]));
  };

  if(mode==2){
    data.ForceKernel.resize(np);

    for(unsigned i=0;i<np;i++)
      data.ForceKernel[i]=data.Selected[i]
          ? capKernel(KerCorrc[i]) : KerCorrc[i];
  }

  if(mode==3){
    const auto& state=User->FlexRestartState;

    if(!state.HasRestartStateLayout() || state.Idp.size()!=np)
      Run_Exceptioon(
          "Boundary correction: initial elastic state is unavailable.");

    data.Prestress.assign(np,TMatrix3f(0));

    for(unsigned i=0;i<np;i++)if(data.Selected[i]){
      const unsigned id=Idp_c->cptr()[FlexStrucRidpc[i]];
      const tmatrix3f& fs=User->GetFlexRestartFstar(i,id);

      const tmatrix3f pn=CalcFlexStrucPK1Stress(
          fs,FlexStrucDatac[data.Body[i]].cmat);

      data.Prestress[i]=ScaleFlexMatrix(
          fmath::MulMatrix3x3(pn,fmath::TrasMatrix3x3(fs)),
          1.f/state.Jstar[i]);
    }
  }

  if(mode==4){
    // Geometry-only calibration. Material reference volumes are NOT used here.
    // C maps scalar weights to the assembled constant-stress area vectors.
    struct Entry{ unsigned row; tdouble3 c; };
    std::vector<std::vector<Entry>> columns(np);
    std::vector<tdouble3> target(np,TDouble3(0)),residual(np,TDouble3(0));
    std::vector<double> beta(np,1.0),diagonal(np,0.0);
    const double vg=Simulate2D ? Dp*Dp : Dp*Dp*Dp;
    const double scale=vg*Dp; // vg^2 / dp^(dimension-1)
    const double eta2=1.e-6,betaMin=0.05;

    const auto dot=[](const tdouble3& a,const tdouble3& b){
      return a.x*b.x+a.y*b.y+a.z*b.z;
    };

    const auto gradient=[&](unsigned i,unsigned j){
      const tdouble3 r=Pos0c[i]-Pos0c[j];
      const float x=float(r.x),y=Simulate2D ? 0.f : float(r.y),z=float(r.z);
      const float rr=x*x+y*y+z*z;
      const float fac=TKernel==KERNEL_Wendland
          ? fsph::GetKernel_Fac<KERNEL_Wendland>(CSP,rr)
          : fsph::GetKernel_Fac<KERNEL_Cubic>(CSP,rr);
      return TDouble3(double(fac)*x,double(fac)*y,double(fac)*z);
    };

    for(unsigned i=0;i<np;i++){
      double a[3][3]={};

      for(unsigned k=0;k<NumPairsc[i];k++){
        const unsigned j=PairIdxc[i][k];
        if(j>=np || j==i || data.Body[i]!=data.Body[j])continue;

        const tdouble3 r=Pos0c[j]-Pos0c[i],g=gradient(i,j);
        const double rv[3]={r.x,Simulate2D ? 0.0 : r.y,r.z};
        const double gv[3]={g.x,g.y,g.z};

        for(unsigned u=0;u<3;u++)for(unsigned v=0;v<3;v++)
          a[u][v]+=vg*rv[u]*gv[v];
      }

      if(Simulate2D)a[1][1]=1.0;

      const tmatrix3d m=TMatrix3d(a[0][0],a[0][1],a[0][2],
          a[1][0],a[1][1],a[1][2],a[2][0],a[2][1],a[2][2]);

      const double det=fmath::Determinant3x3(m);
      if(!std::isfinite(det) || det<=1.e-12)
        Run_Exceptioon("Mode 4: singular geometric kernel moment.");

      const tmatrix3d b=fmath::InverseMatrix3x3(m,det);
      tdouble3 center=TDouble3(0);

      for(unsigned k=0;k<NumPairsc[i];k++){
        const unsigned j=PairIdxc[i][k];
        if(j>=np || j==i || data.Body[i]!=data.Body[j])continue;

        const tdouble3 g=gradient(i,j);
        const tdouble3 raw=TDouble3(scale*g.x,scale*g.y,scale*g.z);

        const tdouble3 q=TDouble3(
            scale*(b.a11*g.x+b.a12*g.y+b.a13*g.z),
            scale*(b.a21*g.x+b.a22*g.y+b.a23*g.z),
            scale*(b.a31*g.x+b.a32*g.y+b.a33*g.z));

        target[i]=target[i]-raw;
        target[j]=target[j]+raw;
        center=center-q;
        columns[i].push_back(Entry{j,q});
      }

      columns[i].push_back(Entry{i,center});

      for(const Entry& e:columns[i]){
        residual[e.row]=residual[e.row]+e.c;
        diagonal[i]+=dot(e.c,e.c);
      }
    }

    double target2=0.0;
    for(unsigned i=0;i<np;i++){
      residual[i]=residual[i]-target[i];
      target2+=dot(target[i],target[i]);
    }

    // Strictly convex bounded least squares; fixed particles stay at beta=1.
    // Include ALL residual rows, including the unmodified transition region.
    bool converged=false;
    unsigned sweeps=0;
    double projected=0.0;

    for(;sweeps<200000;sweeps++){
      for(unsigned i=0;i<np;i++)if(data.Selected[i]){
        double grad=eta2*(beta[i]-1.0);
        for(const Entry& e:columns[i])
          grad+=dot(e.c,residual[e.row]);

        const double next=std::max(betaMin,std::min(1.0,
            beta[i]-grad/(diagonal[i]+eta2)));
        const double change=next-beta[i];
        beta[i]=next;

        for(const Entry& e:columns[i])
          residual[e.row]=residual[e.row]+TDouble3(
              change*e.c.x,change*e.c.y,change*e.c.z);
      }

      projected=0.0;
      for(unsigned i=0;i<np;i++)if(data.Selected[i]){
        double grad=eta2*(beta[i]-1.0);
        for(const Entry& e:columns[i])
          grad+=dot(e.c,residual[e.row]);

        const double next=std::max(betaMin,std::min(1.0,
            beta[i]-grad/(diagonal[i]+eta2)));
        projected=std::max(projected,std::fabs(next-beta[i]));
      }

      if(projected<1.e-9){
        converged=true;
        break;
      }
    }

    if(!converged)
      Run_Exceptioon(fun::PrintStr(
          "Mode 4: geometric beta optimization did not converge "
          "(projected=%.9g).",projected));

    data.Beta.resize(np);
    double r2=0.0,rmax=0.0,bmin=1.0,bmax=0.0;

    for(unsigned i=0;i<np;i++){
      data.Beta[i]=float(beta[i]);
      const double rr=dot(residual[i],residual[i]);
      r2+=rr;
      rmax=std::max(rmax,std::sqrt(rr));
      bmin=std::min(bmin,beta[i]);
      bmax=std::max(bmax,beta[i]);
    }

    Log->Printf(
        "  Mode 4: geometry-only beta, sweeps=%u, "
        "range=[%.9g,%.9g], projected=%.3g.",
        sweeps+1,bmin,bmax,projected);

    Log->Printf(
        "  Mode 4: area residual L2=%.6g, "
        "max/dp^(d-1)=%.6g; eta=0.001, bounds=[0.05,1].",
        std::sqrt(r2/std::max(target2,1.e-30)),rmax);
  }

  std::vector<unsigned> total(FlexStrucCount,0);
  std::vector<unsigned> selected(FlexStrucCount,0);

  for(unsigned i=0;i<np;i++){
    total[data.Body[i]]++;
    if(data.Selected[i])selected[data.Body[i]]++;
  }

  for(unsigned b=0;b<FlexStrucCount;b++)
    Log->Printf(
        "  Restart boundary correction: mode=%d, body=%u, scope=%u/%u particles.",
        mode,b,selected[b],total[b]);
}

tmatrix3f JSphCpu::BuildFlexRestartBoundaryForceMatrix(
    unsigned pfs,unsigned body,
    const tmatrix3f& frelative,const tmatrix3f& preference,
    const tmatrix3f& kercorr)const
{
  const tmatrix3f original=
      fmath::MulMatrix3x3(preference,kercorr);

  if(!User->IsFlexPrescribedRestart() &&
     !User->IsFlexModelTransferRestart())return original;

  const int mode=User->FlexRestartBoundaryCorrectionMode;
  if(mode==0)return original;

  const auto& data=User->FlexRestartBoundaryData;

  if(pfs>=data.Body.size() || pfs>=data.Selected.size() ||
     data.Body[pfs]!=body)
    Run_Exceptioon(
        "Boundary correction: missing or inconsistent particle cache.");

  switch(mode){
    case 1:
      // Selected particles use identity correction in force assembly.
      return data.Selected[pfs] ? preference : original;

    case 2:
      if(pfs>=data.ForceKernel.size())
        Run_Exceptioon(
            "Boundary correction: force kernel cache is unavailable.");

      return fmath::MulMatrix3x3(
          preference,data.ForceKernel[pfs]);

    case 3:{
      if(!data.Selected[pfs])return original;

      if(pfs>=data.Prestress.size())
        Run_Exceptioon(
            "Boundary correction: prestress cache is unavailable.");

      // Q = P*B + F_relative*P_initial*(I-B).
      const tmatrix3f transported=
          fmath::MulMatrix3x3(frelative,data.Prestress[pfs]);

      const tmatrix3f corrected=
          fmath::MulMatrix3x3(transported,kercorr);

      tmatrix3f result=original;

      result.a11+=transported.a11-corrected.a11;
      result.a12+=transported.a12-corrected.a12;
      result.a13+=transported.a13-corrected.a13;

      result.a21+=transported.a21-corrected.a21;
      result.a22+=transported.a22-corrected.a22;
      result.a23+=transported.a23-corrected.a23;

      result.a31+=transported.a31-corrected.a31;
      result.a32+=transported.a32-corrected.a32;
      result.a33+=transported.a33-corrected.a33;

      return result;
    }

    case 4:
      if(pfs>=data.Beta.size() || !std::isfinite(data.Beta[pfs]) ||
         data.Beta[pfs]<=0.f)
        Run_Exceptioon("Mode 4: invalid geometric energy weight.");

      return ScaleFlexMatrix(original,data.Beta[pfs]);

    default:
      Run_Exceptioon("Invalid restart boundary correction mode.");
  }

  return original;
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