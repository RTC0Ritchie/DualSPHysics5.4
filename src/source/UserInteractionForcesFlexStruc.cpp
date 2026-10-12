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

namespace {
tmatrix3f ScaleFlexMatrix(tmatrix3f m,float s){
  m.a11*=s; m.a12*=s; m.a13*=s;
  m.a21*=s; m.a22*=s; m.a23*=s;
  m.a31*=s; m.a32*=s; m.a33*=s;
  return m;
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
  const bool useHistory=(PartBegin!=0);
  const auto& history=User.FlexRestartHistory;

  if(useHistory &&
    (!history.Available() || history.Idp.size()!=np))
    Run_Exceptioon("Flexible restart state is unavailable.");

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

    if(!useHistory){
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
          User.GetFlexRestartDefGrad(pfs,id);

      const float jstar=history.Jstar[pfs];

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

      // Current density from the X0 reference density.
      flexDensity[pfs]=history.Rho0[pfs]/jrelative;

      /// Restart: store P_t/0 * B_0 without volume weights.
      // Vi*Vj is applied in the particle-pair loop.
      flexForceMatrix[pfs]=
          fmath::MulMatrix3x3(preference,kercorr[pfs]);
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
        useHistory
        ? history.Mass[pfs1]
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
        const tmatrix3f pk1kercorrp2=flexForceMatrix[pfs2];
        tmatrix3f pk1kercorrp1p2;
        pk1kercorrp1p2.a11=pk1kercorrp1.a11+pk1kercorrp2.a11; pk1kercorrp1p2.a12=pk1kercorrp1.a12+pk1kercorrp2.a12; pk1kercorrp1p2.a13=pk1kercorrp1.a13+pk1kercorrp2.a13;
        pk1kercorrp1p2.a21=pk1kercorrp1.a21+pk1kercorrp2.a21; pk1kercorrp1p2.a22=pk1kercorrp1.a22+pk1kercorrp2.a22; pk1kercorrp1p2.a23=pk1kercorrp1.a23+pk1kercorrp2.a23;
        pk1kercorrp1p2.a31=pk1kercorrp1.a31+pk1kercorrp2.a31; pk1kercorrp1p2.a32=pk1kercorrp1.a32+pk1kercorrp2.a32; pk1kercorrp1p2.a33=pk1kercorrp1.a33+pk1kercorrp2.a33;
        tfloat3 pk1kercorrdw;
        pk1kercorrdw.x=pk1kercorrp1p2.a11*frx0+pk1kercorrp1p2.a12*fry0+pk1kercorrp1p2.a13*frz0;
        pk1kercorrdw.y=pk1kercorrp1p2.a21*frx0+pk1kercorrp1p2.a22*fry0+pk1kercorrp1p2.a23*frz0;
        pk1kercorrdw.z=pk1kercorrp1p2.a31*frx0+pk1kercorrp1p2.a32*fry0+pk1kercorrp1p2.a33*frz0;
        if(useHistory){
          const float pairVolume=
              history.Vol0[pfs1]*history.Vol0[pfs2];

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
        //-Hourglass correction.
        if(hgfactor){
          const unsigned p2=flexstrucridp[pfs2];
          const tdouble3 posp2=pos[p2];
          const float drx=float(posp1.x-posp2.x);
          const float dry=float(posp1.y-posp2.y);
          const float drz=float(posp1.z-posp2.z);
          const float rr2=drx*drx+dry*dry+drz*drz;
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
          const float mulfac=(hgfactor*vol0p1*vol0p1*wab*youngmod/(rr20*rr*mass0p1)*0.5f*(deltaij+deltaji));
          acep1.x-=mulfac*drx; acep1.y-=mulfac*dry; acep1.z-=mulfac*drz;
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
  const bool useHistory=(PartBegin!=0);
  const auto& history=User->FlexRestartHistory;

  if(useHistory &&
    (!history.Available() || history.Idp.size()!=np))
    Run_Exceptioon("Flexible restart state is unavailable.");

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
            useHistory ? history.Vol0[pfs2] : vol0p1;
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
    const bool useHistory=(PartBegin!=0);
    const auto& history=User->FlexRestartHistory;

    if(useHistory &&
      (!history.Available() || history.Idp.size()!=np))
      Run_Exceptioon("Flexible restart state is unavailable.");
  //-Starts execution using OpenMP.
  const int n=int(np);
  #ifdef OMP_USE
    #pragma omp parallel for schedule (guided)
  #endif
  for(int p=0;p<n;p++){
    const int pfs1=p;      //-Number of particle.

    //-Get number of pairs for this particle.
    const unsigned numpairsp1=numpairs[pfs1];

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
            useHistory ? history.Vol0[pfs2] : vol0p1;
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





//==============================================================================
/// Saves particle IDs and deformation gradients for flexible particles.
//==============================================================================
void JSphCpuSingle::SaveFlexData(){
  if(!User->SaveFlexData || !FlexStruc)
    return;
  if(!(SvData & SDAT_Binx))
    return;

  const unsigned nf=CaseNflexstruc;
  if(!nf)
    return;

  if(!FlexStrucRidpc || !DefGradc)
    Run_Exceptioon("Flexible state arrays are unavailable.");

  const std::string file=
      fun::GetDirWithSlash(DirDataOut)
      +fun::PrintStr("PartFlex_%04u.bi4",unsigned(Part));

  JBinaryData data("JPartFlexBi4");
  data.SetvUint("FormatVer",1);
  data.SetvUint("Cpart",unsigned(Part));
  data.SetvDouble("TimeStep",TimeStep);
  data.SetvUint("Nflex",nf);

  unsigned* ids=data.CreateArrayUint("Idp",nf,false);
  tfloat3* row1=data.CreateArrayFloat3("Frow1",nf,false);
  tfloat3* row2=data.CreateArrayFloat3("Frow2",nf,false);
  tfloat3* row3=data.CreateArrayFloat3("Frow3",nf,false);

  std::unordered_set<unsigned> seen;
  seen.reserve(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    // Flexible-array index -> main-array index -> stable particle ID.
    const unsigned p=FlexStrucRidpc[pfs];

    if(p>=Np)
      Run_Exceptioon("Invalid flexible particle index.");

    const typecode code=Code_c->cptr()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFlexStrucAny(code))
      Run_Exceptioon("Invalid flexible particle in PartFlex output.");

    const unsigned id=Idp_c->cptr()[p];
    if(!seen.insert(id).second)
      Run_Exceptioon("Duplicated flexible particle ID.");

    ids[pfs]=id;

    tmatrix3f f=DefGradc[pfs];

    if(PartBegin){
      const tmatrix3f& fstar=
          User->GetFlexRestartDefGrad(pfs,id);

      f=fmath::MulMatrix3x3(f,fstar);
    }
    row1[pfs]=TFloat3(f.a11,f.a12,f.a13);
    row2[pfs]=TFloat3(f.a21,f.a22,f.a23);
    row3[pfs]=TFloat3(f.a31,f.a32,f.a33);
  }

  data.SaveFile(file,false,true);
}

//==============================================================================
/// Restores flexible deformation gradients by stable particle ID.
//==============================================================================
void JSphCpuSingle::LoadFlexDataRestart(){
  User->FlexRestartHistory.Clear();

  if(!PartBegin || !FlexStruc)
    return;

  const unsigned nf=CaseNflexstruc;
  if(!nf)
    return;

  // Loading history does not require writing to DefGradc.
  if(!FlexStrucRidpc)
    Run_Exceptioon("Flexible particle indices are unavailable.");

  const std::string file=
      fun::GetDirWithSlash(PartBeginDir)
      +fun::PrintStr("PartFlex_%04u.bi4",unsigned(PartBegin));

  JBinaryData data;
  data.LoadFile(file);

  if(data.GetvUint("FormatVer")!=1)
    Run_ExceptioonFile("Unsupported PartFlex format.",file);

  if(data.GetvUint("Cpart")!=unsigned(PartBegin))
    Run_ExceptioonFile("PartFlex frame number does not match.",file);

  if(data.GetvUint("Nflex")!=nf)
    Run_ExceptioonFile("Flexible particle count does not match.",file);

  const double savedtime=data.GetvDouble("TimeStep");
  const double timetol=1e-9*(1.0+std::fabs(TimeStep));
  if(!std::isfinite(savedtime) ||
     std::fabs(savedtime-TimeStep)>timetol)
    Run_ExceptioonFile("PartFlex time does not match restart time.",file);

  // Check array names, types and lengths before accessing their data.
  const unsigned* ids=static_cast<const unsigned*>(
      data.GetArrayTpSize(
          "Idp",JBinaryDataDef::DatUint,nf,file)->GetDataPointer());

  const tfloat3* row1=static_cast<const tfloat3*>(
      data.GetArrayTpSize(
          "Frow1",JBinaryDataDef::DatFloat3,nf,file)->GetDataPointer());

  const tfloat3* row2=static_cast<const tfloat3*>(
      data.GetArrayTpSize(
          "Frow2",JBinaryDataDef::DatFloat3,nf,file)->GetDataPointer());

  const tfloat3* row3=static_cast<const tfloat3*>(
      data.GetArrayTpSize(
          "Frow3",JBinaryDataDef::DatFloat3,nf,file)->GetDataPointer());

  // Stable particle ID -> row in the saved file.
  std::unordered_map<unsigned,unsigned> savedindex;
  savedindex.reserve(nf);

  for(unsigned i=0;i<nf;i++){
    if(!savedindex.emplace(ids[i],i).second)
      Run_ExceptioonFile("Duplicated particle ID in PartFlex.",file);
  }

  std::vector<unsigned char> used(nf,0);
  std::vector<unsigned> restoredid(nf);
  std::vector<tmatrix3f> restored(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned p=FlexStrucRidpc[pfs];

    if(p>=Np)
      Run_Exceptioon("Invalid flexible particle index.");

    const typecode code=Code_c->cptr()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFlexStrucAny(code))
      Run_Exceptioon("Invalid flexible particle during restart.");

    const unsigned id=Idp_c->cptr()[p];
    const auto it=savedindex.find(id);

    if(it==savedindex.end())
      Run_ExceptioonFile("Flexible particle ID missing from PartFlex.",file);

    const unsigned i=it->second;
    if(used[i])
      Run_ExceptioonFile("Duplicated current flexible particle ID.",file);
    used[i]=1;

    const tmatrix3f f={
      row1[i].x,row1[i].y,row1[i].z,
      row2[i].x,row2[i].y,row2[i].z,
      row3[i].x,row3[i].y,row3[i].z
    };

    restoredid[pfs]=id;
    restored[pfs]=f;
  }

  for(unsigned i=0;i<nf;i++){
    if(!used[i])
      Run_ExceptioonFile("Unmatched particle ID in PartFlex.",file);
  }

  // Commit history only after every ID correspondence check succeeds.
  // Keep DefGradc as calculated from the new reference geometry.
  // Build a complete restart state before committing it.
  UserParameters::StFlexRestartHistory history;

  history.Idp.swap(restoredid);
  history.DefGrad.swap(restored);

  history.Jstar.resize(nf);
  history.VolN.resize(nf);
  history.RhoN.resize(nf);
  history.Vol0.resize(nf);
  history.Rho0.resize(nf);
  history.Mass.resize(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned p=FlexStrucRidpc[pfs];

    const unsigned body=
        CODE_GetIbodyFlexStruc(Code_c->cptr()[p]);

    const StFlexStrucData& material=FlexStrucDatac[body];

    const float jstar=
        fmath::Determinant3x3(history.DefGrad[pfs]);

    // Original material inputs remain the natural properties.
    const float voln=material.vol0;
    const float rhon=material.rho0;

    if(!std::isfinite(jstar) || jstar<=0.f ||
      !std::isfinite(voln) || voln<=0.f ||
      !std::isfinite(rhon) || rhon<=0.f)
      Run_Exceptioon("Invalid flexible natural material state.");

    history.Jstar[pfs]=jstar;

    history.VolN[pfs]=voln;
    history.RhoN[pfs]=rhon;

    history.Vol0[pfs]=voln*jstar;
    history.Rho0[pfs]=rhon/jstar;

    history.Mass[pfs]=voln*rhon;

    if(!std::isfinite(history.Vol0[pfs]) ||
      history.Vol0[pfs]<=0.f ||
      !std::isfinite(history.Rho0[pfs]) ||
      history.Rho0[pfs]<=0.f ||
      !std::isfinite(history.Mass[pfs]) ||
      history.Mass[pfs]<=0.f)
      Run_Exceptioon("Invalid flexible restart reference state.");
  }

  User->FlexRestartHistory=history;

  Log->Printf(
    "Flexible restart history and Xn/X0 material state loaded: "
    "%u particles.",nf);
}