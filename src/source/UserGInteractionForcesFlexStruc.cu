//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2025 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License 
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 
 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details. 

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

#include "UserGInteractionForcesFlexStruc.h"
#include "UserGParameters.h"
#include "UserGFlexCte.h"
#include "JSphGpu_ker.h"
#include "Functions.h"
#include "FunctionsCuda.h"
#include <cfloat>
#include <type_traits>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "UserGFlexPrecision.h"
#include <math_constants.h>

namespace cusph{
namespace usergflex{

__constant__ StCteInteraction CTE;
#define CTE_AVAILABLE

#include "FunctionsBasic_iker.h"
#include "FunctionsMath_iker.h"
#include "FunctionsGeo3d_iker.h"
#include "FunSphKernel_iker.h"
#include "FunSphEos_iker.h"
#include "JCellSearch_iker.h"

//==============================================================================
/// Calculates the PK1 stress matrix for each flexible structure particle.
/// Calcula la matriz de tensión PK1 para cada partícula de estructura flexible.
//==============================================================================
// UserGFlexMath (CPU-corresponding elastic implementation).
template<UserGParameters::TpFlexPrecisionMode mode>
class UserGFlexMath;

template<>
class UserGFlexMath<UserGParameters::TpFlexPrecisionMode::Original>{
public:
  using Real=float;
  static constexpr bool OriginalCoordinates=true;
  // Elastic-only kernel access. Keep the shared fluid kernel header unchanged.
  template<TpKernel tker>
  static __device__ float GetKernel_Fac(float rr2){
    return cufsph::GetKernel_Fac<tker>(rr2);
  }

  template<TpKernel tker>
  static __device__ float GetKernel_Wab(float rr2){
    return cufsph::GetKernel_Wab<tker>(rr2);
  }

static __device__ tmatrix3f CalcPK1Stress(const tmatrix3f& defgrad,const tmatrix6f& cmat)
{
  //-Calculate Green-Lagrange strain from deformation gradient.
  tmatrix3f gl=cumath::MulMatrix3x3(cumath::TrasMatrix3x3(defgrad),defgrad);
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
  return cumath::MulMatrix3x3(defgrad,pk2);
}

static __device__ tmatrix3f ReferenceStress(const tmatrix3f& relative,
    const tmatrix6f& cmat,const UserGFlexRestartView& state,unsigned i){
  if(!state.Fstar)return CalcPK1Stress(relative,cmat);
  const auto star=state.Fstar[i];
  const auto total=cumath::MulMatrix3x3(relative,star);
  auto p=cumath::MulMatrix3x3(CalcPK1Stress(total,cmat),cumath::TrasMatrix3x3(star));
  for(int k=0;k<9;k++)(&p.a11)[k]/=state.Jstar[i];
  return p;
}
};

//==============================================================================
// UserGFlexCore (CPU-corresponding elastic implementation).
template<class Math,bool corrected>
class UserGFlexCore{
public:
  template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2>
  static __device__ __forceinline__ void InteractionForcesFlexStruc(unsigned n,float visco
    ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid,const unsigned* dcell
    ,const float4* poscell,const float4* velrhop,const typecode* code
    ,const byte* boundmode,const float3* tangenvel
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr,const tmatrix3f* defgrad
    ,float* flexstrucdt,float3* ace,UserGFlexRestartView restart)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const unsigned pfs1=p;      //-Number of particle.
    const unsigned p1=flexstrucridp[pfs1];

    //-Get codep1.
    const typecode codep1=code[p1];
    if(CODE_IsFlexStrucFlex(codep1)){
      float3 acep1=make_float3(0,0,0);

      //-Obtains basic data of particle p1.
      const float4 pscellp1=poscell[p1];
      float4 velrhop1=velrhop[p1];
      const float pressp1=cufsph::ComputePressCte(velrhop1.w);
      const float4 pscell0p1=poscell0[pfs1];
      const tmatrix3f kercorrp1=kercorr[pfs1];
      const tmatrix3f defgradp1=defgrad[pfs1];

      //-Obtains flexible structure data.
      const float vol0p1=restart.Volume?restart.Volume[pfs1]:flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].vol0;
      const float rho0p1=restart.Density?restart.Density[pfs1]:flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].rho0;
      const float youngmod=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].youngmod;
      const float poissratio=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].poissratio;
      const float hgfactor=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].hgfactor;
      const tmatrix6f cmat=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].cmat;
      const tmatrix3f pk1p1=Math::ReferenceStress(defgradp1,cmat,restart,pfs1);
      const tmatrix3f pk1kercorrp1=cumath::MulMatrix3x3(pk1p1,kercorrp1);

      //-Get current mass of flexible structure particle.
      const float mass0p1=restart.Mass?restart.Mass[pfs1]:vol0p1*rho0p1;
      const float rhop1=rho0p1/cumath::Determinant3x3(defgradp1);

      //-Calculate structural speed of sound.
      const float csp1=sqrt(youngmod*(1.0-poissratio)/(rhop1*(1.0+poissratio)*(1.0-2.0*poissratio)));

      //-Modify values if using mDBC.
      float massp2=CTE.massf;
      if(mdbc2>=MDBC2_Std){
        if(boundmode[p1]==BMODE_MDBC2OFF)massp2=0;
        const float3 tangentvelp1=tangenvel[p1];
        velrhop1.x=tangentvelp1.x;
        velrhop1.y=tangentvelp1.y;
        velrhop1.z=tangentvelp1.z;
      }

      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Flexible structure-Fluid interaction.
      for(int c3=ini3;c3<fin3;c3+=nc.w){
        for(int c2=ini2;c2<fin2;c2+=nc.x){
          unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
          if(pfin){
            for(int p2=pini;p2<pfin;p2++){
              if(CODE_IsFluid(code[p2])){
                const float4 pscellp2=poscell[p2];
                const float drx=pscellp1.x-pscellp2.x+CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
                const float dry=pscellp1.y-pscellp2.y+CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
                const float drz=pscellp1.z-pscellp2.z+CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
                const float rr2=drx*drx+dry*dry+drz*drz;
                if(rr2<=CTE.kernelsize2&&rr2>=ALMOSTZERO){
                  //-Computes kernel.
                  const float fac=Math::template GetKernel_Fac<tker>(rr2);
                  const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
                  //-Obtains velocity of particle p2 and compute difference.
                  const float4 velrhop2=velrhop[p2];
                  const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
                  //-Pressure derivative (Momentum equation).
                  const float pressp2=cufsph::ComputePressCte(velrhop2.w);
                  const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)+(tker==KERNEL_Cubic?cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2):0);
                  const float p_vpm=-prs*massp2*(CTE.massf/mass0p1);
                  acep1.x+=p_vpm*frx; acep1.y+=p_vpm*fry; acep1.z+=p_vpm*frz;                  
                  //-Artificial viscosity.
                  if(tvisco==VISCO_Artificial){
                    const float dot=drx*dvx+dry*dvy+drz*dvz;
                    if(dot<0){
                      const float dot_rr2=dot/(rr2+CTE.eta2);
                      const float amubar=CTE.kernelh*dot_rr2;
                      const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                      const float pi_visc=(-visco*CTE.cs0*amubar/robar)*massp2*(CTE.massf/mass0p1);
                      acep1.x-=pi_visc*frx; acep1.y-=pi_visc*fry; acep1.z-=pi_visc*frz;
                    }
                  }
                  //-Laminar viscosity.
                  else{
                    const float robar2=(velrhop1.w+velrhop2.w);
                    const float temp=4.f*visco/((rr2+CTE.eta2)*robar2);
                    const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz)*(CTE.massf/mass0p1);
                    acep1.x+=vtemp*dvx; acep1.y+=vtemp*dvy; acep1.z+=vtemp*dvz;
                  }
                }
              }
            }
          }
        }
      }

      //-Loop through pairs and calculate forces.
      for(unsigned pair=0;pair<numpairs[pfs1];pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        if(corrected){
          const size_t q=pairidx[pfs1]+pair-pairidx[0];
          const tmatrix3f pj=Math::ReferenceStress(defgrad[pfs2],flexstrucdata[CODE_GetIbodyFlexStruc(code[flexstrucridp[pfs2]])].cmat,restart,pfs2);
          const tfloat3 gi=restart.GradI[q],gj=restart.GradJ[q];
          const float a[3]={gi.x,gi.y,gi.z},b[3]={gj.x,gj.y,gj.z};
          float force[3]={};
          for(int r=0;r<3;r++){
            float fi=0,fj=0;for(int c=0;c<3;c++){fi+=(&pk1p1.a11)[3*r+c]*a[c];fj+=(&pj.a11)[3*r+c]*b[c];}
            force[r]=(vol0p1*fi+restart.Volume[pfs2]*fj)/mass0p1;
          }
          acep1.x+=force[0];acep1.y+=force[1];acep1.z+=force[2];
          if(hgfactor){
            const float4 v=poscell[flexstrucridp[pfs2]];
            const float d[3]={pscellp1.x-v.x+CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(v.w)),
              pscellp1.y-v.y+CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(v.w)),
              pscellp1.z-v.z+CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(v.w))};
            const float rr2=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
            if(rr2>0){
              const auto mi=restart.MapI[q],mj=restart.MapJ[q];
              const float ai[3]={mi.x,mi.y,mi.z},aj[3]={mj.x,mj.y,mj.z};
              const auto fj=defgrad[pfs2];float delta=0;
              for(int r=0;r<3;r++){
                float bi=0,bj=0;for(int c=0;c<3;c++){bi+=(&defgradp1.a11)[3*r+c]*ai[c];bj+=(&fj.a11)[3*r+c]*aj[c];}
                delta+=(2*d[r]-bi-bj)*d[r];
              }
              const float scale=hgfactor*youngmod*Math::template GetKernel_Wab<tker>(rr2)*restart.HgWeight[q]*.5f*delta/(rr2*mass0p1);
              acep1.x-=scale*d[0];acep1.y-=scale*d[1];acep1.z-=scale*d[2];
            }
          }
          continue;
        }
        const float4 pscell0p2=poscell0[pfs2];
        float drx0=pscell0p1.x-pscell0p2.x+CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        float dry0=pscell0p1.y-pscell0p2.y+CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        float drz0=pscell0p1.z-pscell0p2.z+CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        const float fac0=Math::template GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        //-Acceleration due to structure.
        const tmatrix3f defgradp2=defgrad[pfs2];
        const tmatrix3f pk1p2=Math::ReferenceStress(defgradp2,restart.Fstar?flexstrucdata[CODE_GetIbodyFlexStruc(code[flexstrucridp[pfs2]])].cmat:cmat,restart,pfs2);
        const tmatrix3f kercorrp2=kercorr[pfs2];
        const tmatrix3f pk1kercorrp2=cumath::MulMatrix3x3(pk1p2,kercorrp2);
        tmatrix3f pk1kercorrp1p2;
        pk1kercorrp1p2.a11=pk1kercorrp1.a11+pk1kercorrp2.a11; pk1kercorrp1p2.a12=pk1kercorrp1.a12+pk1kercorrp2.a12; pk1kercorrp1p2.a13=pk1kercorrp1.a13+pk1kercorrp2.a13;
        pk1kercorrp1p2.a21=pk1kercorrp1.a21+pk1kercorrp2.a21; pk1kercorrp1p2.a22=pk1kercorrp1.a22+pk1kercorrp2.a22; pk1kercorrp1p2.a23=pk1kercorrp1.a23+pk1kercorrp2.a23;
        pk1kercorrp1p2.a31=pk1kercorrp1.a31+pk1kercorrp2.a31; pk1kercorrp1p2.a32=pk1kercorrp1.a32+pk1kercorrp2.a32; pk1kercorrp1p2.a33=pk1kercorrp1.a33+pk1kercorrp2.a33;
        float3 pk1kercorrdw;
        pk1kercorrdw.x=pk1kercorrp1p2.a11*frx0+pk1kercorrp1p2.a12*fry0+pk1kercorrp1p2.a13*frz0;
        pk1kercorrdw.y=pk1kercorrp1p2.a21*frx0+pk1kercorrp1p2.a22*fry0+pk1kercorrp1p2.a23*frz0;
        pk1kercorrdw.z=pk1kercorrp1p2.a31*frx0+pk1kercorrp1p2.a32*fry0+pk1kercorrp1p2.a33*frz0;
        if(restart.Fstar){
          const float scale=vol0p1*restart.Volume[pfs2]/mass0p1;
          acep1.x+=pk1kercorrdw.x*scale;acep1.y+=pk1kercorrdw.y*scale;acep1.z+=pk1kercorrdw.z*scale;
        }
        else{acep1.x+=pk1kercorrdw.x*vol0p1/rho0p1; acep1.y+=pk1kercorrdw.y*vol0p1/rho0p1; acep1.z+=pk1kercorrdw.z*vol0p1/rho0p1;}
        //-Hourglass correction.
        if(hgfactor){
          const unsigned p2=flexstrucridp[pfs2];
          const float4 pscellp2=poscell[p2];
          float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
          float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
          float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
          const float rr2=drx*drx+dry*dry+drz*drz;
          const float wab=Math::template GetKernel_Wab<tker>(rr2);
          const float xijbarx=defgradp1.a11*drx0+defgradp1.a12*dry0+defgradp1.a13*drz0;
          const float xijbary=defgradp1.a21*drx0+defgradp1.a22*dry0+defgradp1.a23*drz0;
          const float xijbarz=defgradp1.a31*drx0+defgradp1.a32*dry0+defgradp1.a33*drz0;
          const float xjibarx=-(defgradp2.a11*drx0+defgradp2.a12*dry0+defgradp2.a13*drz0);
          const float xjibary=-(defgradp2.a21*drx0+defgradp2.a22*dry0+defgradp2.a23*drz0);
          const float xjibarz=-(defgradp2.a31*drx0+defgradp2.a32*dry0+defgradp2.a33*drz0);
          const float3 epsij=make_float3(drx-xijbarx,dry-xijbary,drz-xijbarz);
          const float3 epsji=make_float3(-drx-xjibarx,-dry-xjibary,-drz-xjibarz);
          const float rr=sqrt(rr2);
          const float deltaij=(epsij.x*drx+epsij.y*dry+epsij.z*drz)/rr;
          const float deltaji=-(epsji.x*drx+epsji.y*dry+epsji.z*drz)/rr;
          const float mulfac=(hgfactor*vol0p1*(restart.Volume?restart.Volume[pfs2]:vol0p1)*wab*youngmod/(rr20*rr*mass0p1)*0.5f*(deltaij+deltaji));
          acep1.x-=mulfac*drx; acep1.y-=mulfac*dry; acep1.z-=mulfac*drz;
        }
      }

      //-Store results.
      if(acep1.x||acep1.y||acep1.z||csp1){
        float3 r=ace[p1]; r.x+=acep1.x; r.y+=acep1.y; r.z+=acep1.z; ace[p1]=r;
        if(csp1>flexstrucdt[pfs1])flexstrucdt[pfs1]=csp1;
      }
    }
  }
}

  template<TpKernel tker,bool simulate2d>
  static __device__ __forceinline__ void CalcFlexStrucKerCorr(unsigned n,const typecode* code,const StFlexStrucData* flexstrucdata
    ,const unsigned* flexstrucridp,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx
    ,tmatrix3f* kercorr,UserGFlexRestartView restart)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    if(corrected){tmatrix3f b={};b.a11=b.a22=b.a33=1;kercorr[p]=b;return;}
    const unsigned pfs1=p;      //-Number of particle.

    //-Get number of pairs for this particle.
    const unsigned numpairsp1=numpairs[pfs1];

    //-If this particle has pairs.
    if(numpairsp1){
      tmatrix3f kercorrp1={0};

      //-Obtains basic data of particle p1.
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(code[flexstrucridp[pfs1]])].vol0;
      const float4 pscell0p1=poscell0[pfs1];

      //-Calculate kernel correction matrix.
      for(unsigned pair=0;pair<numpairsp1;pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        const float weight=restart.Volume?restart.Volume[pfs2]:vol0p1;
        const float4 pscell0p2=poscell0[pfs2];
        float drx0=pscell0p1.x-pscell0p2.x + CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        float dry0=pscell0p1.y-pscell0p2.y + CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        float drz0=pscell0p1.z-pscell0p2.z + CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        //-Computes kernel.
        const float fac0=Math::template GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        kercorrp1.a11-=weight*drx0*frx0; kercorrp1.a12-=weight*drx0*fry0; kercorrp1.a13-=weight*drx0*frz0;
        kercorrp1.a21-=weight*dry0*frx0; kercorrp1.a22-=weight*dry0*fry0; kercorrp1.a23-=weight*dry0*frz0;
        kercorrp1.a31-=weight*drz0*frx0; kercorrp1.a32-=weight*drz0*fry0; kercorrp1.a33-=weight*drz0*frz0;
      }
      if(simulate2d){
        kercorrp1.a12=kercorrp1.a21=kercorrp1.a23=kercorrp1.a32=0.0;
        kercorrp1.a22=1.0;
      }
      kercorr[pfs1]=cumath::InverseMatrix3x3(kercorrp1);
    }
  }
}

  template<TpKernel tker,bool simulate2d>
  static __device__ __forceinline__ void CalcFlexStrucDefGrad(unsigned n,const float4* poscell,const typecode* code
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr
    ,tmatrix3f* defgrad,UserGFlexRestartView restart)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    if(corrected){
      tmatrix3f f={};const float4 a=poscell[flexstrucridp[p]];
      for(unsigned k=0;k<numpairs[p];k++){
        const float4 b=poscell[flexstrucridp[pairidx[p][k]]];
        const float d[3]={a.x-b.x+CTE.poscellsize*(PSCEL_GetfX(a.w)-PSCEL_GetfX(b.w)),
          a.y-b.y+CTE.poscellsize*(PSCEL_GetfY(a.w)-PSCEL_GetfY(b.w)),
          a.z-b.z+CTE.poscellsize*(PSCEL_GetfZ(a.w)-PSCEL_GetfZ(b.w))};
        const auto v=restart.GradI[pairidx[p]+k-pairidx[0]];const float g[3]={v.x,v.y,v.z};
        for(int r=0;r<3;r++)for(int c=0;c<3;c++)(&f.a11)[3*r+c]-=d[r]*g[c];
      }
      if(simulate2d){f.a12=f.a21=f.a23=f.a32=0;f.a22=1;}
      defgrad[p]=f;return;
    }
    const unsigned pfs1=p;      //-Number of particle.

    //-Get number of pairs for this particle.
    const unsigned numpairsp1=numpairs[pfs1];

    //-If this particle has pairs.
    if(numpairsp1){
      tmatrix3f defgradp1={0};

      //-Obtains basic data of particle p1.
      const unsigned p1=flexstrucridp[pfs1];
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(code[p1])].vol0;
      const float4 pscellp1=poscell[p1];
      const float4 pscell0p1=poscell0[pfs1];
      const tmatrix3f kercorrp1=kercorr[pfs1];

      //-Calculate deformation gradient.
      for(unsigned pair=0;pair<numpairsp1;pair++){
        const unsigned pfs2=pairidx[pfs1][pair];
        const float weight=restart.Volume?restart.Volume[pfs2]:vol0p1;
        const unsigned p2=flexstrucridp[pfs2];
        const float4 pscellp2=poscell[p2];
        const float4 pscell0p2=poscell0[pfs2];
        const float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
        const float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
        const float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
        const float drx0=pscell0p1.x-pscell0p2.x + CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        const float dry0=pscell0p1.y-pscell0p2.y + CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        const float drz0=pscell0p1.z-pscell0p2.z + CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        const float fac0=Math::template GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        defgradp1.a11-=weight*drx*frx0; defgradp1.a12-=weight*drx*fry0; defgradp1.a13-=weight*drx*frz0;
        defgradp1.a21-=weight*dry*frx0; defgradp1.a22-=weight*dry*fry0; defgradp1.a23-=weight*dry*frz0;
        defgradp1.a31-=weight*drz*frx0; defgradp1.a32-=weight*drz*fry0; defgradp1.a33-=weight*drz*frz0;
      }
      if(simulate2d){
        defgradp1.a12=defgradp1.a21=defgradp1.a23=defgradp1.a32=0.0;
        defgradp1.a22=1.0;
      }
      defgrad[pfs1]=cumath::MulMatrix3x3(defgradp1,kercorrp1);
    }
  }
}

  static __device__ __forceinline__ void CalcFlexStrucNormals(unsigned n,const unsigned* flexstrucridp
    ,const tmatrix3f* defgrad,const float3* boundnor0
    ,float3* boundnor)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
    const float3 boundnor0p=boundnor0[p];
    if(boundnor0p.x!=0 || boundnor0p.y!=0 || boundnor0p.z!=0){
      const tmatrix3f rot=cumath::InverseMatrix3x3(cumath::TrasMatrix3x3(defgrad[p]));
      float3 boundnornewp;
      boundnornewp.x=rot.a11*boundnor0p.x+rot.a12*boundnor0p.y+rot.a13*boundnor0p.z;
      boundnornewp.y=rot.a21*boundnor0p.x+rot.a22*boundnor0p.y+rot.a23*boundnor0p.z;
      boundnornewp.z=rot.a31*boundnor0p.x+rot.a32*boundnor0p.y+rot.a33*boundnor0p.z;
      const float mag0=sqrt(boundnor0p.x*boundnor0p.x+boundnor0p.y*boundnor0p.y+boundnor0p.z*boundnor0p.z);
      const float magnew=sqrt(boundnornewp.x*boundnornewp.x+boundnornewp.y*boundnornewp.y+boundnornewp.z*boundnornewp.z);
      boundnornewp.x*=mag0/magnew; boundnornewp.y*=mag0/magnew; boundnornewp.z*=mag0/magnew;
      boundnor[flexstrucridp[p]]=boundnornewp;
    }
  }
}
};

// CUDA entry points only; all calculation belongs to the class above.
template<class Math,TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2,bool corrected>
__global__ void KerInteractionForcesFlexStruc(unsigned n,float visco
    ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid,const unsigned* dcell
    ,const float4* poscell,const float4* velrhop,const typecode* code
    ,const byte* boundmode,const float3* tangenvel
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr,const tmatrix3f* defgrad
    ,float* flexstrucdt,float3* ace,UserGFlexRestartView restart)
{
  UserGFlexCore<Math,corrected>::template InteractionForcesFlexStruc<tker,tvisco,mdbc2>(n,visco,scelldiv,nc,cellzero,beginendcellfluid,dcell,poscell,velrhop,code,boundmode,tangenvel,flexstrucdata,flexstrucridp,poscell0,numpairs,pairidx,kercorr,defgrad,flexstrucdt,ace,restart);
}

template<class Math,TpKernel tker,bool simulate2d,bool corrected>
__global__ void KerCalcFlexStrucKerCorr(unsigned n,const typecode* code,const StFlexStrucData* flexstrucdata
    ,const unsigned* flexstrucridp,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx
    ,tmatrix3f* kercorr,UserGFlexRestartView restart)
{
  UserGFlexCore<Math,corrected>::template CalcFlexStrucKerCorr<tker,simulate2d>(n,code,flexstrucdata,flexstrucridp,poscell0,numpairs,pairidx,kercorr,restart);
}

template<class Math,TpKernel tker,bool simulate2d,bool corrected>
__global__ void KerCalcFlexStrucDefGrad(unsigned n,const float4* poscell,const typecode* code
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr
    ,tmatrix3f* defgrad,UserGFlexRestartView restart)
{
  UserGFlexCore<Math,corrected>::template CalcFlexStrucDefGrad<tker,simulate2d>(n,poscell,code,flexstrucdata,flexstrucridp,poscell0,numpairs,pairidx,kercorr,defgrad,restart);
}

template<class Math,bool corrected>
__global__ void KerCalcFlexStrucNormals(unsigned n,const unsigned* flexstrucridp
    ,const tmatrix3f* defgrad,const float3* boundnor0
    ,float3* boundnor)
{
  UserGFlexCore<Math,corrected>::CalcFlexStrucNormals(n,flexstrucridp,defgrad,boundnor0,boundnor);
}

//==============================================================================
// Precision arithmetic.
// Geometry subtraction is ALWAYS double; the policy only selects where to narrow.
// No run-time precision branch occurs inside particle or neighbour loops.
template<class Scalar> class UserGFlexPrecisionMath{
public:
  using Real=Scalar;
  static constexpr bool OriginalCoordinates=false;
  using Matrix=UserGFlexMatrix<Real>;
  using Vector=UserGFlexVector<Real>;
  using State=UserGFlexState<Real>;

  template<TpKernel kernel> static __device__ double Fac64(double rr2){
    const double rad=sqrt(rr2),q=rad/double(CTE.kernelh);
    if(kernel==KERNEL_Wendland){
      const double a=1.0-0.5*q;
      return double(CTE.bwenh)*a*a*a;
    }
    if(rad>double(CTE.kernelh)){
      const double a=2.0-q;
      return double(CTE.cubic_c2)*a*a/rad;
    }
    // The limiting value at zero avoids 0/0 for a coincident pair.
    return rr2? (double(CTE.cubic_c1)*q+double(CTE.cubic_d1)*q*q)/rad
              : double(CTE.cubic_c1)/double(CTE.kernelh);
  }
  template<TpKernel kernel> static __device__ double Wab64(double rr2){
    const double rad=sqrt(rr2),q=rad/double(CTE.kernelh);
    if(kernel==KERNEL_Wendland){
      const double a=1.0-0.5*q;
      return double(CTE.awen)*(2.0*q+1.0)*a*a*a*a;
    }
    if(rad>double(CTE.kernelh)){
      const double a=2.0-q;return double(CTE.cubic_a24)*a*a*a;
    }
    return double(CTE.cubic_a2)*(1.0+(0.75*q-1.5)*q*q);
  }
  template<TpKernel kernel> static __device__ Vector Gradient(const State& s,size_t pair){
    if(sizeof(Real)==sizeof(double))return s.PairGradient[pair];
    const float fac=cufsph::GetKernel_Fac<kernel>(float(s.PairNorm[pair]));
    const Vector d=s.PairReference[pair];
    return {{Real(fac*float(d.v[0])),Real(fac*float(d.v[1])),Real(fac*float(d.v[2]))}};
  }
  static __device__ Matrix Multiply(const Matrix& a,const Matrix& b){
    Matrix c={};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)
      for(int k=0;k<3;k++)c.v[3*i+j]+=a.v[3*i+k]*b.v[3*k+j];
    return c;
  }
  static __device__ Matrix ReferenceStress(const Matrix& h,const tmatrix6f& material,
      const UserGFlexRestartView& state,unsigned i){
    if(!state.Fstar)return Stress(h,material);
    if(sizeof(Real)==sizeof(double) && state.Hstar64){
      Matrix hs,transpose;
      for(int r=0;r<3;r++)for(int c=0;c<3;c++){
        hs.v[3*r+c]=Real((&state.Hstar64[i].a11)[3*r+c]);
        transpose.v[3*r+c]=Real((&state.Hstar64[i].a11)[3*c+r])+(r==c?Real(1):Real(0));
      }
      Matrix product=Multiply(h,hs),total;
      for(int k=0;k<9;k++)total.v[k]=(hs.v[k]+h.v[k])+product.v[k];
      Matrix p=Multiply(Stress(total,material),transpose);
      for(int k=0;k<9;k++)p.v[k]/=Real(state.Jstar64[i]);
      return p;
    }
    Matrix star,transpose;
    for(int r=0;r<3;r++)for(int c=0;c<3;c++){
      star.v[3*r+c]=Real((&state.Fstar[i].a11)[3*r+c]);
      transpose.v[3*r+c]=Real((&state.Fstar[i].a11)[3*c+r]);
    }
    Matrix total=Multiply(h,star);
    for(int r=0;r<3;r++)for(int c=0;c<3;c++)total.v[3*r+c]+=star.v[3*r+c]-(r==c?Real(1):Real(0));
    Matrix p=Multiply(Stress(total,material),transpose);
    for(int k=0;k<9;k++)p.v[k]/=Real(state.Jstar[i]);
    return p;
  }
  static __device__ Matrix Stress(const Matrix& h,const tmatrix6f& material){
    // E = (H + H^T + H^T H)/2, exactly the full Green-Lagrange law.
    // Do not reconstruct H by subtracting I from the float F mirror.
    Real e[9];
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      Real v=Real(.5)*(h.v[3*i+j]+h.v[3*j+i]);
      for(int k=0;k<3;k++)v+=Real(.5)*h.v[3*k+i]*h.v[3*k+j];
      e[3*i+j]=v;
    }
    const Real ev[6]={e[0],e[4],e[8],2*e[5],2*e[6],2*e[1]};
    Real sv[6]={};const float* c=&material.a11;
    for(int i=0;i<6;i++)for(int j=0;j<6;j++)sv[i]+=Real(c[6*i+j])*ev[j];
    const Matrix stress={{sv[0],sv[5],sv[4],sv[5],sv[1],sv[3],sv[4],sv[3],sv[2]}};
    Matrix p=stress;
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)
      for(int k=0;k<3;k++)p.v[3*i+j]+=h.v[3*i+k]*stress.v[3*k+j];
    return p;
  }
  static __device__ Real DeterminantF(const Matrix& h){
    const Real a=1+h.v[0],b=h.v[1],c=h.v[2],d=h.v[3],e=1+h.v[4];
    const Real f=h.v[5],g=h.v[6],j=h.v[7],k=1+h.v[8];
    return a*(e*k-f*j)-b*(d*k-f*g)+c*(d*j-e*g);
  }
};

//==============================================================================
// Precision state access.
// Shared entry for future CPU-to-GPU features. Mode differences belong here,
// not in constitutive/boundary/dissipation formulas. All branches are compile-time.
template<class Math> class UserGFlexAccess{
public:
  using R=typename Math::Real;
  using Matrix=UserGFlexMatrix<R>;
  using Vector=UserGFlexVector<R>;
  using State=UserGFlexState<R>;
  static __device__ __forceinline__ Matrix H(const StInterParmsFlexStrucg& t,const State& s,unsigned i){
    if(!Math::OriginalCoordinates)return s.H[i];
    Matrix h;const tmatrix3f f=t.defgrad[i];
    for(int k=0;k<9;k++)h.v[k]=(&f.a11)[k]-(k%4==0?1.f:0.f);
    return h;
  }
  static __device__ __forceinline__ Matrix F(const StInterParmsFlexStrucg& t,const State& s,unsigned i){
    if(Math::OriginalCoordinates){
      Matrix f;const tmatrix3f v=t.defgrad[i];for(int k=0;k<9;k++)f.v[k]=(&v.a11)[k];return f;
    }
    Matrix f=s.H[i];f.v[0]+=1;f.v[4]+=1;f.v[8]+=1;return f;
  }
  static __device__ __forceinline__ Matrix Correction(const StInterParmsFlexStrucg& t,const State& s,unsigned i){
    if(!Math::OriginalCoordinates)return s.Correction[i];
    Matrix b;const tmatrix3f v=t.kercorr[i];for(int k=0;k<9;k++)b.v[k]=(&v.a11)[k];return b;
  }
  template<TpKernel kernel>
  static __device__ __forceinline__ Vector Gradient(const StInterParmsFlexStrucg& t,const State& s,unsigned i,unsigned k){
    if(!Math::OriginalCoordinates)
      return UserGFlexPrecisionMath<R>::template Gradient<kernel>(s,t.pairidx[i]+k-t.pairidx[0]);
    const float4 a=t.poscell0[i],b=t.poscell0[t.pairidx[i][k]];
    const float x=a.x-b.x+CTE.poscellsize*(PSCEL_GetfX(a.w)-PSCEL_GetfX(b.w));
    const float y=a.y-b.y+CTE.poscellsize*(PSCEL_GetfY(a.w)-PSCEL_GetfY(b.w));
    const float z=a.z-b.z+CTE.poscellsize*(PSCEL_GetfZ(a.w)-PSCEL_GetfZ(b.w));
    const float fac=cufsph::GetKernel_Fac<kernel>(x*x+y*y+z*z);
    return {{R(fac*x),R(fac*y),R(fac*z)}};
  }
};

//==============================================================================
// Precision device calculations.
// Compile-time scalar selection; no precision-mode branches in the physical law.
template<class R> class UserGFlexOptionalAccess{
public:
  using Vector=UserGFlexVector<R>;
  using Matrix=UserGFlexMatrix<R>;
  static __device__ Vector Read(const tfloat3* f,const tdouble3* d,size_t k){
    if(sizeof(R)==sizeof(double)){const auto a=d[k];return Vector{{R(a.x),R(a.y),R(a.z)}};}
    const auto a=f[k];return Vector{{R(a.x),R(a.y),R(a.z)}};
  }
  static __device__ Matrix Offset(const UserGFlexRestartView& v,unsigned i){
    Matrix a;for(int k=0;k<9;k++)a.v[k]=sizeof(R)==sizeof(double)?R((&v.Hoffset64[i].a11)[k]):R((&v.Hoffset[i].a11)[k]);return a;
  }
};

template<class Math,bool corrected> class UserGFlexPrecisionCore{
public:
  using R=typename Math::Real;
  using Access=UserGFlexAccess<Math>;
  using Matrix=typename Math::Matrix;
  using Vector=typename Math::Vector;
  using State=typename Math::State;

  static __device__ void Reference(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const unsigned p=t.flexstrucridp[i];
    s.Reference[i]=make_double3(t.posxy[p].x,t.posxy[p].y,t.posz[p]);
  }
  static __device__ void Displacement(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const unsigned p=t.flexstrucridp[i];const double3 x=s.Reference[i];
    s.Displacement[i]=make_double3(t.posxy[p].x-x.x,t.posxy[p].y-x.y,t.posz[p]-x.z);
  }
  template<TpKernel kernel> static __device__ void Pairs(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const double3 a=s.Reference[i];
    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const double3 b=s.Reference[j];
      const size_t q=t.pairidx[i]+k-t.pairidx[0];
      const double d[3]={a.x-b.x,a.y-b.y,a.z-b.z};
      const double rr=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
      s.PairReference[q]={{R(d[0]),R(d[1]),R(d[2])}};s.PairNorm[q]=R(rr);
      if(sizeof(R)==sizeof(double)){
        const double fac=Math::template Fac64<kernel>(rr);
        s.PairGradient[q]={{R(fac*d[0]),R(fac*d[1]),R(fac*d[2])}};
      }
    }
  }
  template<TpKernel kernel,bool sim2d>
  static __device__ void Correction(const StInterParmsFlexStrucg& t,State s,unsigned i){
    if(corrected){
      Matrix b={};b.v[0]=b.v[4]=b.v[8]=1;s.Correction[i]=b;
      tmatrix3f f={};f.a11=f.a22=f.a33=1;const_cast<tmatrix3f*>(t.kercorr)[i]=f;return;
    }
    double a[3][6]={};const double3 x=s.Reference[i];
    const double vol=t.flexstrucdata[CODE_GetIbodyFlexStruc(t.code[t.flexstrucridp[i]])].vol0;
    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const double3 y=s.Reference[j];
      const double d[3]={x.x-y.x,x.y-y.y,x.z-y.z};
      const auto g=Access::template Gradient<kernel>(t,s,i,k);
      for(int r=0;r<3;r++)for(int c=0;c<3;c++)a[r][c]-=(sizeof(R)==sizeof(double) && t.restart.Volume64?t.restart.Volume64[j]:(t.restart.Volume?double(t.restart.Volume[j]):vol))*d[r]*double(g.v[c]);
    }
    if(sim2d){for(int k=0;k<3;k++)a[1][k]=a[k][1]=0;a[1][1]=1;}
    for(int r=0;r<3;r++)a[r][r+3]=1;
    // Partial pivoting, without model-dependent tolerance tuning.
    for(int k=0;k<3;k++){
      int pivot=k;for(int r=k+1;r<3;r++)if(fabs(a[r][k])>fabs(a[pivot][k]))pivot=r;
      if(a[pivot][k]==0 || !isfinite(a[pivot][k])){atomicExch(s.Error,1u);return;}
      if(pivot!=k)for(int c=0;c<6;c++){double z=a[k][c];a[k][c]=a[pivot][c];a[pivot][c]=z;}
      const double scale=a[k][k];for(int c=0;c<6;c++)a[k][c]/=scale;
      for(int r=0;r<3;r++)if(r!=k){
        const double scale=a[r][k];for(int c=0;c<6;c++)a[r][c]-=scale*a[k][c];
      }
    }
    tmatrix3f mirror;
    for(int r=0;r<3;r++)for(int c=0;c<3;c++){
      s.Correction[i].v[3*r+c]=R(a[r][c+3]);(&mirror.a11)[3*r+c]=float(a[r][c+3]);
    }
    const_cast<tmatrix3f*>(t.kercorr)[i]=mirror;
  }
  template<TpKernel kernel,bool sim2d>
  static __device__ void Geometry(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const auto mat=t.flexstrucdata[CODE_GetIbodyFlexStruc(t.code[t.flexstrucridp[i]])];
    const double3 ui=s.Displacement[i];Matrix sum={};
    if(corrected)sum=UserGFlexOptionalAccess<R>::Offset(t.restart,i);
    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const double3 uj=s.Displacement[j];
      const R du[3]={R(ui.x-uj.x),R(ui.y-uj.y),R(ui.z-uj.z)};
      if(corrected){
        const auto g=UserGFlexOptionalAccess<R>::Read(t.restart.GradI,t.restart.GradI64,t.pairidx[i]+k-t.pairidx[0]);
        for(int r=0;r<3;r++)for(int c=0;c<3;c++)sum.v[3*r+c]-=du[r]*g.v[c];
        continue;
      }
      const auto g=Access::template Gradient<kernel>(t,s,i,k);
      for(int r=0;r<3;r++)for(int c=0;c<3;c++)
        if(!sim2d || (r!=1 && c!=1))sum.v[3*r+c]-=(sizeof(R)==sizeof(double) && t.restart.Volume64?R(t.restart.Volume64[j]):R(t.restart.Volume?t.restart.Volume[j]:mat.vol0))*du[r]*g.v[c];
    }
    Matrix h=corrected?sum:Math::Multiply(sum,Access::Correction(t,s,i));
    if(sim2d)h.v[1]=h.v[3]=h.v[4]=h.v[5]=h.v[7]=0;
    s.H[i]=h;
    const Matrix pk=Math::ReferenceStress(h,mat.cmat,t.restart,i);
    s.PK[i]=pk;s.PB[i]=Math::Multiply(pk,s.Correction[i]);
    tmatrix3f f;
    for(int k=0;k<9;k++)(&f.a11)[k]=float(h.v[k]);
    f.a11+=1;f.a22+=1;f.a33+=1;t.defgrad[i]=f;
  }
  static __device__ void Normals(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const float3 n=t.boundnor0[i];if(n.x==0 && n.y==0 && n.z==0)return;
    Matrix f=s.H[i];f.v[0]+=1;f.v[4]+=1;f.v[8]+=1;
    const R* a=f.v;
    // cofactor(F) n = det(F) F^-T n. Preserve the sign of det(F).
    R r[3]={
      (a[4]*a[8]-a[5]*a[7])*R(n.x)+(a[5]*a[6]-a[3]*a[8])*R(n.y)+(a[3]*a[7]-a[4]*a[6])*R(n.z),
      (a[2]*a[7]-a[1]*a[8])*R(n.x)+(a[0]*a[8]-a[2]*a[6])*R(n.y)+(a[1]*a[6]-a[0]*a[7])*R(n.z),
      (a[1]*a[5]-a[2]*a[4])*R(n.x)+(a[2]*a[3]-a[0]*a[5])*R(n.y)+(a[0]*a[4]-a[1]*a[3])*R(n.z)};
    const R mag=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    const R scale=sqrt(R(n.x)*n.x+R(n.y)*n.y+R(n.z)*n.z)/mag*(Math::DeterminantF(s.H[i])<0?-1:1);
    t.boundnor[t.flexstrucridp[i]]=make_float3(float(r[0]*scale),float(r[1]*scale),float(r[2]*scale));
  }

  template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2,bool sim2d>
  static __device__ void Forces(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const unsigned p1=t.flexstrucridp[i];const typecode codep1=t.code[p1];
    if(!CODE_IsFlexStrucFlex(codep1))return;
    const auto mat=t.flexstrucdata[CODE_GetIbodyFlexStruc(codep1)];
    const R vol0p1=(sizeof(R)==sizeof(double) && t.restart.Volume64)?R(t.restart.Volume64[i]):R(t.restart.Volume?t.restart.Volume[i]:mat.vol0);
    const R rho0p1=(sizeof(R)==sizeof(double) && t.restart.Density64)?R(t.restart.Density64[i]):R(t.restart.Density?t.restart.Density[i]:mat.rho0);
    const float mass0p1=t.restart.Mass?t.restart.Mass[i]:mat.vol0*mat.rho0;
    float4 velrhop1=t.velrhop[p1];
    const float pressp1=cufsph::ComputePressCte(velrhop1.w);
    const R rhop1=R(rho0p1)/Math::DeterminantF(s.H[i]);
    const float csp1=float(sqrt(R(mat.youngmod)*(R(1)-R(mat.poissratio))
      /(rhop1*(R(1)+R(mat.poissratio))*(R(1)-R(2)*R(mat.poissratio)))));
    R acc[3]={};
    const auto& dvd=t.divdatag;
    const int scelldiv=dvd.scelldiv;const int4 nc=dvd.nc;const int3 cellzero=dvd.cellzero;
    const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
    const auto dcell=t.dcell;const auto code=t.code;const auto velrhop=t.velrhop;
    const auto boundmode=t.boundmode;const auto tangenvel=t.tangenvel;
    const float visco=t.viscob;
      //-Modify values if using mDBC.
      float massp2=CTE.massf;
      if(mdbc2>=MDBC2_Std){
        if(boundmode[p1]==BMODE_MDBC2OFF)massp2=0;
        const float3 tangentvelp1=tangenvel[p1];
        velrhop1.x=tangentvelp1.x;
        velrhop1.y=tangentvelp1.y;
        velrhop1.z=tangentvelp1.z;
      }

      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Flexible structure-Fluid interaction.
      for(int c3=ini3;c3<fin3;c3+=nc.w){
        for(int c2=ini2;c2<fin2;c2+=nc.x){
          unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
          if(pfin){
            for(int p2=pini;p2<pfin;p2++){
              if(CODE_IsFluid(code[p2])){
                const double dx=t.posxy[p1].x-t.posxy[p2].x;
                const double dy=t.posxy[p1].y-t.posxy[p2].y;
                const double dz=t.posz[p1]-t.posz[p2];
                const float drx=float(dx),dry=float(dy),drz=float(dz);
                const float rr2=float(dx*dx+dy*dy+dz*dz);
                if(rr2<=CTE.kernelsize2&&rr2>=ALMOSTZERO){
                  //-Computes kernel.
                  const float fac=cufsph::GetKernel_Fac<tker>(rr2);
                  const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
                  //-Obtains velocity of particle p2 and compute difference.
                  const float4 velrhop2=velrhop[p2];
                  const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
                  //-Pressure derivative (Momentum equation).
                  const float pressp2=cufsph::ComputePressCte(velrhop2.w);
                  const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)+(tker==KERNEL_Cubic?cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2):0);
                  const float p_vpm=-prs*massp2*(CTE.massf/mass0p1);
                  acc[0]+=p_vpm*frx; acc[1]+=p_vpm*fry; acc[2]+=p_vpm*frz;                  
                  //-Artificial viscosity.
                  if(tvisco==VISCO_Artificial){
                    const float dot=drx*dvx+dry*dvy+drz*dvz;
                    if(dot<0){
                      const float dot_rr2=dot/(rr2+CTE.eta2);
                      const float amubar=CTE.kernelh*dot_rr2;
                      const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                      const float pi_visc=(-visco*CTE.cs0*amubar/robar)*massp2*(CTE.massf/mass0p1);
                      acc[0]-=pi_visc*frx; acc[1]-=pi_visc*fry; acc[2]-=pi_visc*frz;
                    }
                  }
                  //-Laminar viscosity.
                  else{
                    const float robar2=(velrhop1.w+velrhop2.w);
                    const float temp=4.f*visco/((rr2+CTE.eta2)*robar2);
                    const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz)*(CTE.massf/mass0p1);
                    acc[0]+=vtemp*dvx; acc[1]+=vtemp*dvy; acc[2]+=vtemp*dvz;
                  }
                }
              }
            }
          }
        }
      }


    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const size_t flat=t.pairidx[i]+k-t.pairidx[0];
      if(corrected){
        using Opt=UserGFlexOptionalAccess<R>;const auto& v=t.restart;
        const auto gi=Opt::Read(v.GradI,v.GradI64,flat),gj=Opt::Read(v.GradJ,v.GradJ64,flat);
        const Matrix pi=s.PK[i],pj=s.PK[j];
        const R vj=sizeof(R)==sizeof(double)?R(v.Volume64[j]):R(v.Volume[j]);
        for(int r=0;r<3;r++){
          R fi=0,fj=0;for(int c=0;c<3;c++){fi+=pi.v[3*r+c]*gi.v[c];fj+=pj.v[3*r+c]*gj.v[c];}
          acc[r]+=(vol0p1*fi+vj*fj)/R(mass0p1);
        }
        if(mat.hgfactor){
          const unsigned p2=t.flexstrucridp[j];
          const double dd[3]={t.posxy[p1].x-t.posxy[p2].x,t.posxy[p1].y-t.posxy[p2].y,t.posz[p1]-t.posz[p2]};
          const double rr64=dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2];
          const R rr2=R(rr64),d[3]={R(dd[0]),R(dd[1]),R(dd[2])};
          if(rr2>0){
            const auto mi=Opt::Read(v.MapI,v.MapI64,flat),mj=Opt::Read(v.MapJ,v.MapJ64,flat);
            const auto hi=s.H[i],hj=s.H[j];R delta=0;
            for(int r=0;r<3;r++){
              R bi=0,bj=0;for(int c=0;c<3;c++){bi+=hi.v[3*r+c]*mi.v[c];bj+=hj.v[3*r+c]*mj.v[c];}
              delta+=((d[r]-mi.v[r]-bi)+(d[r]-mj.v[r]-bj))*d[r];
            }
            const R w=sizeof(R)==sizeof(double)?R(Math::template Wab64<tker>(rr64)):R(cufsph::GetKernel_Wab<tker>(float(rr2)));
            const R weight=sizeof(R)==sizeof(double)?R(v.HgWeight64[flat]):R(v.HgWeight[flat]);
            const R scale=R(mat.hgfactor)*R(mat.youngmod)*w*weight*R(.5)*delta/(rr2*R(mass0p1));
            for(int r=0;r<3;r++)acc[r]-=scale*d[r];
          }
        }
        continue;
      }
      const auto g=Access::template Gradient<tker>(t,s,i,k);
      const Matrix pi=s.PB[i];
      Matrix pj=s.PB[j];
      // Original law evaluates both stresses with particle i's constitutive tensor.
      // Keep that convention for pairs joining different material/body codes.
      if(!t.restart.Fstar && CODE_GetIbodyFlexStruc(t.code[t.flexstrucridp[j]])!=CODE_GetIbodyFlexStruc(codep1))
        pj=Math::Multiply(Math::Stress(s.H[j],mat.cmat),s.Correction[j]);
      for(int r=0;r<3;r++){
        R force=0;for(int c=0;c<3;c++)force+=(pi.v[3*r+c]+pj.v[3*r+c])*g.v[c];
        if(t.restart.Fstar)acc[r]+=force*(R(vol0p1)*(sizeof(R)==sizeof(double) && t.restart.Volume64?R(t.restart.Volume64[j]):R(t.restart.Volume[j]))/R(mass0p1));
        else acc[r]+=force*R(vol0p1)/R(rho0p1);
      }
      if(mat.hgfactor){
        const unsigned p2=t.flexstrucridp[j];
        const double dd[3]={t.posxy[p1].x-t.posxy[p2].x,t.posxy[p1].y-t.posxy[p2].y,t.posz[p1]-t.posz[p2]};
        const double rr64=dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2];
        const R rr2=R(rr64),rr=sqrt(rr2),d[3]={R(dd[0]),R(dd[1]),R(dd[2])};
        const R wab=sizeof(R)==sizeof(double)?R(Math::template Wab64<tker>(rr64))
          :R(cufsph::GetKernel_Wab<tker>(float(rr2)));
        const auto ref=s.PairReference[flat];const auto ui=s.Displacement[i],uj=s.Displacement[j];
        const R du[3]={R(ui.x-uj.x),R(ui.y-uj.y),R(ui.z-uj.z)};
        const Matrix hi=s.H[i],hj=s.H[j];R delta=0;
        for(int r=0;r<3;r++){
          if(sim2d && r==1)continue;
          R vi=0,vj=0;
          for(int c=0;c<3;c++){
            if(sim2d && c==1)continue;
            vi+=hi.v[3*r+c]*ref.v[c];vj+=hj.v[3*r+c]*ref.v[c];
          }
          delta+=((du[r]-vi)+(du[r]-vj))*d[r]/rr;
        }
        // Mixed keeps the original float HG denominator; extended keeps it double.
        const R rr0=ref.v[0]*ref.v[0]+ref.v[1]*ref.v[1]+ref.v[2]*ref.v[2];
        const R scale=R(mat.hgfactor)*R(vol0p1)*(sizeof(R)==sizeof(double) && t.restart.Volume64?R(t.restart.Volume64[j]):R(t.restart.Volume?t.restart.Volume[j]:vol0p1))*wab*R(mat.youngmod)
          /(rr0*rr*R(mass0p1))*R(.5)*delta;
        for(int r=0;r<3;r++)acc[r]-=scale*d[r];
      }
    }
    float3 a=t.ace[p1];a.x=float(R(a.x)+acc[0]);a.y=float(R(a.y)+acc[1]);a.z=float(R(a.z)+acc[2]);
    t.ace[p1]=a;
    if(csp1>t.flexstrucdt[i])t.flexstrucdt[i]=csp1;
  }
};

// Thin CUDA entry points; all physics lives in the policy-independent class.
template<class Math,int stage,TpKernel kernel,bool sim2d,bool corrected>
__global__ void KerUserGFlexPrecision(StInterParmsFlexStrucg t,typename Math::State s){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=t.vnpfs)return;
  using Core=UserGFlexPrecisionCore<Math,corrected>;
  if(stage==0)Core::Reference(t,s,i);
  if(stage==1)Core::template Pairs<kernel>(t,s,i);
  if(stage==2)Core::template Correction<kernel,sim2d>(t,s,i);
  if(stage==3)Core::Displacement(t,s,i);
  if(stage==4)Core::template Geometry<kernel,sim2d>(t,s,i);
  if(stage==5)Core::Normals(t,s,i);
}
template<class Math,TpKernel kernel,TpVisco visco,TpMdbc2Mode mdbc,bool sim2d,bool corrected>
__global__ void KerUserGFlexPrecisionForces(StInterParmsFlexStrucg t,typename Math::State s){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<t.vnpfs)UserGFlexPrecisionCore<Math,corrected>::template Forces<kernel,visco,mdbc,sim2d>(t,s,i);
}


//==============================================================================
// Elastic-solid contact (CPU: UserInteractionForcesFlexStruc.cpp).

// The same physical law is used by all precision policies.
// Mode 0: CPU float-style particle contact, double analytic-wall geometry.
// Mode 1: double separation/radius/overlap, float subsequent particle force.
// Mode 2: double contact geometry, coefficients, force and accumulation.
template<unsigned mode> class UserGFlexContactMath{
public:
  using Real=typename std::conditional<mode==2,double,float>::type;
  using Geometry=typename std::conditional<mode==0,float,double>::type;
  using Config=UserGParameters::StContactConfig;

  template<class T> static __host__ __device__ T Root(T v){ return sqrt(v); }
  static __host__ __device__ Real Stiffness(const Config& c,double dp,float e,float nu){
    return c.KnMode==1? Real(c.Kn):
      Real(2)*Real(e)*Root(Real(dp))/(Real(3)*(Real(1)-Real(nu)*Real(nu)));
  }
  static __host__ __device__ Real Damping(const Config& c,double dp,Real mass,float e){
    return c.DampingMode==1? Real(c.Damping):
      Real(mode==2? 1.4e-5: double(1.4e-5f))*Root(Real(6)*mass*Real(e)*Root(Real(dp)/Real(2)));
  }
  template<class T> static __host__ __device__ T Force(const Config& c,T overlap,T vn,T kn,T damping){
    if(overlap<=T(0))return T(0);
    const T elastic=c.Linearity==1? kn*overlap: kn*overlap*Root(overlap);
    const T f=elastic-damping*Root(Root(overlap))*vn;
    return f>T(0)? f:T(0); // CPU: repulsion only, no adhesive force.
  }
  static __host__ __device__ bool ParticleGeometry(double dx,double dy,double dz
    ,double dp,float kernelsize2,Real& overlap,Real& nx,Real& ny,Real& nz)
  {
    const Geometry x=Geometry(dx),y=Geometry(dy),z=Geometry(dz);
    const Geometry rr2=x*x+y*y+z*z;
    if(rr2>Geometry(kernelsize2) || rr2<Geometry(ALMOSTZERO))return false;
    const Geometry r=Root(rr2);
    overlap=Real(Geometry(dp)-r);
    if(overlap<=Real(0))return false;
    nx=Real(x/r); ny=Real(y/r); nz=Real(z/r);
    return true;
  }
};

// Device-only view, independent of the host user object and precision owner.
class UserGFlexContactData{
public:
  unsigned count,axis;
  double dp;
  StDivDataGpu cells;
  const unsigned* dcell;
  const unsigned* ridp;
  const typecode* code;
  const double2* xy;
  const double* z;
  const float4* velocity;
  const StFlexStrucData* material;
  const float* mass; // Optional stable-index mass: later restart/continuation hook.
  float3* acceleration;
};

template<unsigned mode,bool particle> class UserGFlexContactCore{
public:
  static __device__ void Apply(const UserGFlexContactData& d
    ,const UserGParameters::StContactConfig& c,unsigned pfs)
  {
    using Math=UserGFlexContactMath<mode>;
    using R=typename Math::Real;
    const unsigned p=d.ridp[pfs];
    if(!CODE_IsFlexStrucFlex(d.code[p]))return;
    const auto& mat=d.material[CODE_GetIbodyFlexStruc(d.code[p])];
    const R mass=d.mass? R(d.mass[pfs]): R(mat.vol0)*R(mat.rho0);
    const R kn=Math::Stiffness(c,d.dp,mat.youngmod,mat.poissratio);
    const R damping=Math::Damping(c,d.dp,mass,mat.youngmod);
    const double2 xy=d.xy[p];
    const double z=d.z[p];
    const float4 vel=d.velocity[p]; // Actual velocity, not mDBC tangential velocity.
    R ax=0,ay=0,az=0;
    if(particle){
      int i1,f1,i2,f2,i3,f3;
      cunsearch::Initsp(d.dcell[p],d.axis,d.cells.domcellcode,d.cells.scelldiv
        ,d.cells.nc,d.cells.cellzero,i1,f1,i2,f2,i3,f3);
      for(int c3=i3;c3<f3;c3+=d.cells.nc.w){
        for(int c2=i2;c2<f2;c2+=d.cells.nc.x){
          unsigned pini,pfin=0;
          cunsearch::ParticleRange(c2,c3,i1,f1,d.cells.beginendcell,pini,pfin);
          for(unsigned q=pini;q<pfin;q++){
            if(!CODE_IsFixed(d.code[q]) || CODE_IsFlexStrucAny(d.code[q]))continue;
            R overlap,nx,ny,nz;
            if(Math::ParticleGeometry(xy.x-d.xy[q].x,xy.y-d.xy[q].y,z-d.z[q]
              ,d.dp,d.cells.kernelsize2,overlap,nx,ny,nz)){
              const R vn=R(vel.x)*nx+R(vel.y)*ny+R(vel.z)*nz;
              const R a=Math::Force(c,overlap,vn,kn,damping)/mass;
              ax+=a*nx; ay+=a*ny; az+=a*nz;
            }
          }
        }
      }
    }
    else{
      // Analytic surfaces retain the CPU double geometry in EVERY mode.
      double overlap=0,nx=0,nz=1;
      if(c.Surface==1 || (c.Surface==2 && c.NormalMode==1)){
        const double localz=c.Surface==1? c.WallZ:c.Z1+c.Slope*(xy.x-c.X1);
        overlap=localz-(z-0.5*d.dp);
      }
      else if(c.Surface==2 && c.NormalMode==2){
        nx=c.Nx; nz=c.Nz;
        overlap=0.5*d.dp-((xy.x-c.X1)*nx+(z-c.Z1)*nz);
      }
      const double vn=double(vel.x)*nx+double(vel.z)*nz;
      const double force=Math::Force(c,overlap,vn,double(kn),double(damping));
      ax=R(force*nx/double(mass)); az=R(force*nz/double(mass));
    }
    // One writer per elastic particle. Existing solver acceleration remains float.
    float3 a=d.acceleration[p];
    a.x+=float(ax); a.y+=float(ay); a.z+=float(az);
    d.acceleration[p]=a;
  }
};

template<unsigned mode,bool particle> __global__ void UserGFlexContactKernel(
  UserGFlexContactData d,UserGParameters::StContactConfig c)
{
  const unsigned pfs=blockIdx.x*blockDim.x+threadIdx.x;
  if(pfs<d.count)UserGFlexContactCore<mode,particle>::Apply(d,c,pfs);
}

class UserGFlexContactLaunch{
public:
  template<unsigned mode> static void Run(const UserGFlexContactData& d
    ,const UserGParameters::StContactConfig& c,cudaStream_t stream)
  {
    const unsigned blocks=(d.count+127)/128;
    if(c.Mode==1)UserGFlexContactKernel<mode,true><<<blocks,128,0,stream>>>(d,c);
    else UserGFlexContactKernel<mode,false><<<blocks,128,0,stream>>>(d,c);
  }
};


} // namespace usergflex
} // namespace cusph

//==============================================================================
// UserGFlexBackend (CPU-corresponding elastic implementation).
enum class UserGFlexStage{Reference,Geometry,Forces};

template<class Math,bool corrected>
class UserGFlexBackend{
private:
  UserGFlexBackend()=delete;

  template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2>
  static void Interaction_ForcesFlexStrucT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker,TpVisco tvisco>
  static void Interaction_ForcesFlexStruc_gt1(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void Interaction_ForcesFlexStruc_gt0(const StInterParmsFlexStrucg& tfs);

  // CPU counterpart: UserInteractionForcesFlexStruc.cpp.
  template<TpKernel tker,bool simulate2d>
  static void CalcFlexStrucKerCorrT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void CalcFlexStrucKerCorr_gt0(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker,bool simulate2d>
  static void CalcFlexStrucDefGradT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void CalcFlexStrucDefGrad_ct0(const StInterParmsFlexStrucg& tfs);

public:
  template<UserGFlexStage stage>
  static void Run(const StInterParmsFlexStrucg& tfs){
    if(stage==UserGFlexStage::Reference)CalcFlexStrucKerCorr(tfs);
    else if(stage==UserGFlexStage::Geometry)UpdateFlexStrucGeometry(tfs);
    else Interaction_ForcesFlexStruc(tfs);
  }

  static void CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs);
  static void UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs);
  static void Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs);
};

template<class Math,bool corrected>
template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> void UserGFlexBackend<Math,corrected>::Interaction_ForcesFlexStrucT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    const StDivDataGpu& dvd=tfs.divdatag;
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerInteractionForcesFlexStruc<Math,tker,tvisco,mdbc2,corrected> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.viscob,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell+dvd.cellfluid,tfs.dcell,tfs.poscell,tfs.velrhop,tfs.code,tfs.boundmode,tfs.tangenvel,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad,tfs.flexstrucdt,tfs.ace,tfs.restart);
  }
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<class Math,bool corrected>
template<TpKernel tker,TpVisco tvisco> void UserGFlexBackend<Math,corrected>::Interaction_ForcesFlexStruc_gt1(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.mdbc2!=MDBC2_None)throw "Extra mDBC options are disabled for FastCompilation...";
  Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_None>(tfs);
#else
  if(tfs.mdbc2==MDBC2_None)      Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_None> (tfs);
  else if(tfs.mdbc2==MDBC2_Std)  Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_Std>  (tfs);
  else if(tfs.mdbc2==MDBC2_NoPen)Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_NoPen>(tfs);
#endif
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<class Math,bool corrected>
template<TpKernel tker> void UserGFlexBackend<Math,corrected>::Interaction_ForcesFlexStruc_gt0(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tvisco!=VISCO_Artificial)throw "Extra viscosity options are disabled for FastCompilation...";
  Interaction_ForcesFlexStruc_gt1<tker,VISCO_Artificial> (tfs);
#else
  if(tfs.tvisco==VISCO_Artificial)     Interaction_ForcesFlexStruc_gt1<tker,VISCO_Artificial>(tfs);
  else if(tfs.tvisco==VISCO_Laminar)   Interaction_ForcesFlexStruc_gt1<tker,VISCO_Laminar>   (tfs);
  else if(tfs.tvisco==VISCO_LaminarSPS)Interaction_ForcesFlexStruc_gt1<tker,VISCO_LaminarSPS>(tfs);
#endif
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<class Math,bool corrected>
void UserGFlexBackend<Math,corrected>::Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  Interaction_ForcesFlexStruc_gt0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)  Interaction_ForcesFlexStruc_gt0<KERNEL_Wendland>(tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)Interaction_ForcesFlexStruc_gt0<KERNEL_Cubic>   (tfs);
#endif
#endif
}

template<class Math,bool corrected>
template<TpKernel tker,bool simulate2d> void UserGFlexBackend<Math,corrected>::CalcFlexStrucKerCorrT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucKerCorr<Math,tker,simulate2d,corrected> <<<sgridb,SPHBSIZE>>>
        (tfs.vnpfs,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,const_cast<tmatrix3f*>(tfs.kercorr),tfs.restart);
  }
}

template<class Math,bool corrected>
template<TpKernel tker> void UserGFlexBackend<Math,corrected>::CalcFlexStrucKerCorr_gt0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucKerCorrT<tker,true>  (tfs);
  else              CalcFlexStrucKerCorrT<tker,false> (tfs);
}

template<class Math,bool corrected>
void UserGFlexBackend<Math,corrected>::CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  CalcFlexStrucKerCorr_gt0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)     CalcFlexStrucKerCorr_gt0<KERNEL_Wendland> (tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)   CalcFlexStrucKerCorr_gt0<KERNEL_Cubic   > (tfs);
#endif
#endif
}

template<class Math,bool corrected>
template<TpKernel tker,bool simulate2d> void UserGFlexBackend<Math,corrected>::CalcFlexStrucDefGradT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucDefGrad<Math,tker,simulate2d,corrected> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.poscell,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad,tfs.restart);
  }
}

template<class Math,bool corrected>
template<TpKernel tker> void UserGFlexBackend<Math,corrected>::CalcFlexStrucDefGrad_ct0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucDefGradT<tker,true>  (tfs);
  else              CalcFlexStrucDefGradT<tker,false> (tfs);
}

template<class Math,bool corrected>
void UserGFlexBackend<Math,corrected>::UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  CalcFlexStrucDefGrad_ct0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)  CalcFlexStrucDefGrad_ct0<KERNEL_Wendland>(tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)CalcFlexStrucDefGrad_ct0<KERNEL_Cubic>   (tfs);
#endif
#endif
  if(tfs.vnpfs&&tfs.mdbc2>=MDBC2_Std){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucNormals<Math,corrected> <<<sgridb,SPHBSIZE,0,tfs.stm>>>(tfs.vnpfs,tfs.flexstrucridp,tfs.defgrad,tfs.boundnor0,tfs.boundnor);
  }
}

//==============================================================================
// Precision cache ownership and host dispatch.
class UserGFlexPrecision::Impl{
public:
  unsigned Mode=0,Particles=0;
  size_t Bytes=0;
  void* WorkBuffer=nullptr;
  size_t WorkCapacity=0;
  std::vector<void*> Buffers;
  UserGFlexRestartView Restart;
  std::vector<tmatrix3d> HostHstar64,HostNatural64;
  UserGFlexState<float> Mixed={};
  UserGFlexState<double> Extended={};
  static void Check(cudaError_t err){
    if(err!=cudaSuccess)throw std::runtime_error(std::string("GPU elastic precision: ")+cudaGetErrorString(err));
  }
  template<class T> void Alloc(T*& p,size_t n){
    if(!n){p=nullptr;return;}
    Check(cudaMalloc(reinterpret_cast<void**>(&p),sizeof(T)*n));
    Buffers.push_back(p);Bytes+=sizeof(T)*n;
  }
  template<class R> void Allocate(UserGFlexState<R>& s,unsigned n,unsigned pairs){
    Alloc(s.Reference,n);Alloc(s.Displacement,n);
    Alloc(s.PairReference,pairs);Alloc(s.PairNorm,pairs);
    if(sizeof(R)==sizeof(double))Alloc(s.PairGradient,pairs);
    Alloc(s.Correction,n);Alloc(s.H,n);Alloc(s.PK,n);Alloc(s.PB,n);Alloc(s.Error,1);
    Check(cudaMemset(s.Error,0,sizeof(unsigned)));
  }
  ~Impl(){for(void* p:Buffers)cudaFree(p);if(WorkBuffer)cudaFree(WorkBuffer);}
};
UserGFlexPrecision::UserGFlexPrecision():Data(nullptr){}
UserGFlexPrecision::~UserGFlexPrecision(){delete Data;}
void UserGFlexPrecision::Reset(){delete Data;Data=nullptr;}
void UserGFlexPrecision::Allocate(unsigned mode,unsigned particles,unsigned pairs){
  Reset();
  if(!mode)return;
  if(mode>2)throw std::runtime_error("Invalid GPU elastic precision mode.");
  Data=new Impl;Data->Mode=mode;Data->Particles=particles;
  try{
    if(mode==1)Data->Allocate(Data->Mixed,particles,pairs);
    else Data->Allocate(Data->Extended,particles,pairs);
  }
  catch(...){Reset();throw;}
}
void UserGFlexPrecision::AllocatePairs(unsigned pairs){
  if(!Data || !Data->Mode || Data->Restart.Corrected)return;
  if(Data->Mixed.PairReference || Data->Extended.PairReference)throw std::runtime_error("Pair cache already allocated.");
  if(Data->Mode==1){Data->Alloc(Data->Mixed.PairReference,pairs);Data->Alloc(Data->Mixed.PairNorm,pairs);}
  else{Data->Alloc(Data->Extended.PairReference,pairs);Data->Alloc(Data->Extended.PairNorm,pairs);Data->Alloc(Data->Extended.PairGradient,pairs);}
}
void UserGFlexPrecision::SetNaturalCorrection64(const std::vector<tmatrix3d>& b){
  if(!Data)Data=new Impl;Data->HostNatural64=b;
}
const std::vector<tmatrix3d>& UserGFlexPrecision::NaturalCorrection64()const{
  static const std::vector<tmatrix3d> empty;return Data?Data->HostNatural64:empty;
}
void UserGFlexPrecision::DownloadCorrection64(unsigned n,std::vector<tmatrix3d>& b)const{
  if(!Data || Data->Mode!=2 || Data->Particles!=n)throw std::runtime_error("Natural double correction cache mismatch.");
  static_assert(sizeof(UserGFlexMatrix<double>)==sizeof(tmatrix3d),"Correction layout mismatch");
  b.resize(n);Impl::Check(cudaMemcpy(b.data(),Data->Extended.Correction,n*sizeof(tmatrix3d),cudaMemcpyDeviceToHost));
}
void UserGFlexPrecision::UploadCorrections(unsigned n,unsigned pairs,const tdouble3* gi,const tdouble3* gj,
  const tdouble3* mi,const tdouble3* mj,const double* weight,const tmatrix3d* h0){
  if(!Data || !Data->Restart.Fstar || Data->Restart.Corrected)throw std::runtime_error("Invalid optional correction upload order.");
  auto& v=Data->Restart;
  if(Data->Mode==2){
    {tdouble3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,gi,pairs*sizeof(tdouble3),cudaMemcpyHostToDevice));v.GradI64=p;}
    {tdouble3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,gj,pairs*sizeof(tdouble3),cudaMemcpyHostToDevice));v.GradJ64=p;}
    {tdouble3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,mi,pairs*sizeof(tdouble3),cudaMemcpyHostToDevice));v.MapI64=p;}
    {tdouble3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,mj,pairs*sizeof(tdouble3),cudaMemcpyHostToDevice));v.MapJ64=p;}
    {double* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,weight,pairs*sizeof(double),cudaMemcpyHostToDevice));v.HgWeight64=p;}
    {tmatrix3d* p=nullptr;Data->Alloc(p,n);Impl::Check(cudaMemcpy(p,h0,n*sizeof(tmatrix3d),cudaMemcpyHostToDevice));v.Hoffset64=p;}
  }
  else{
    {std::vector<tfloat3> a(pairs);for(unsigned i=0;i<pairs;i++)a[i]=TFloat3(float(gi[i].x),float(gi[i].y),float(gi[i].z));
      tfloat3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,a.data(),pairs*sizeof(tfloat3),cudaMemcpyHostToDevice));v.GradI=p;}
    {std::vector<tfloat3> a(pairs);for(unsigned i=0;i<pairs;i++)a[i]=TFloat3(float(gj[i].x),float(gj[i].y),float(gj[i].z));
      tfloat3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,a.data(),pairs*sizeof(tfloat3),cudaMemcpyHostToDevice));v.GradJ=p;}
    {std::vector<tfloat3> a(pairs);for(unsigned i=0;i<pairs;i++)a[i]=TFloat3(float(mi[i].x),float(mi[i].y),float(mi[i].z));
      tfloat3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,a.data(),pairs*sizeof(tfloat3),cudaMemcpyHostToDevice));v.MapI=p;}
    {std::vector<tfloat3> a(pairs);for(unsigned i=0;i<pairs;i++)a[i]=TFloat3(float(mj[i].x),float(mj[i].y),float(mj[i].z));
      tfloat3* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,a.data(),pairs*sizeof(tfloat3),cudaMemcpyHostToDevice));v.MapJ=p;}
    {std::vector<float> a(pairs);for(unsigned i=0;i<pairs;i++)a[i]=float(weight[i]);
      float* p=nullptr;Data->Alloc(p,pairs);Impl::Check(cudaMemcpy(p,a.data(),pairs*sizeof(float),cudaMemcpyHostToDevice));v.HgWeight=p;}
    {std::vector<tmatrix3f> a(n);for(unsigned i=0;i<n;i++)for(int k=0;k<9;k++)(&a[i].a11)[k]=float((&h0[i].a11)[k]);
      tmatrix3f* p=nullptr;Data->Alloc(p,n);Impl::Check(cudaMemcpy(p,a.data(),n*sizeof(tmatrix3f),cudaMemcpyHostToDevice));v.Hoffset=p;}
  }
  v.Corrected=true;
}
void* UserGFlexPrecision::State(unsigned mode,unsigned particles)const{
  if(!Data || Data->Mode!=mode || Data->Particles!=particles)
    throw std::runtime_error("GPU elastic precision cache mismatch. Reinitialize before changing mode or topology.");
  return mode==1?static_cast<void*>(&Data->Mixed):static_cast<void*>(&Data->Extended);
}
UserGFlexRestartView UserGFlexPrecision::Restart()const{
  return Data?Data->Restart:UserGFlexRestartView();
}
void UserGFlexPrecision::UploadRestart(unsigned n,const tmatrix3f* f,const float* j,
    const float* volume,const float* density,const float* mass,
    const float* naturalVolume,const float* naturalDensity){
  if(!Data)Data=new Impl;
  if(Data->Restart.Fstar)throw std::runtime_error("Elastic restart is already uploaded.");
  tmatrix3f* fg=nullptr;float *jg=nullptr,*vg=nullptr,*rg=nullptr,*mg=nullptr;
  Data->Alloc(fg,n);Data->Alloc(jg,n);Data->Alloc(vg,n);Data->Alloc(rg,n);Data->Alloc(mg,n);
  Impl::Check(cudaMemcpy(fg,f,n*sizeof(tmatrix3f),cudaMemcpyHostToDevice));
  Impl::Check(cudaMemcpy(jg,j,n*sizeof(float),cudaMemcpyHostToDevice));
  Impl::Check(cudaMemcpy(vg,volume,n*sizeof(float),cudaMemcpyHostToDevice));
  Impl::Check(cudaMemcpy(rg,density,n*sizeof(float),cudaMemcpyHostToDevice));
  Impl::Check(cudaMemcpy(mg,mass,n*sizeof(float),cudaMemcpyHostToDevice));
  Data->Restart.Fstar=fg;Data->Restart.Jstar=jg;Data->Restart.Volume=vg;
  Data->Restart.Density=rg;Data->Restart.Mass=mg;
  if(Data->Mode==2){
    if(Data->HostHstar64.size()!=n)throw std::runtime_error("Missing double elastic reference state.");
    std::vector<double> j64(n),v64(n),r64(n);
    for(unsigned i=0;i<n;i++){
      const auto& h=Data->HostHstar64[i];
      const double a=1+h.a11,b=h.a12,c=h.a13,d=h.a21,e=1+h.a22,
          f=h.a23,g=h.a31,k=h.a32,l=1+h.a33;
      j64[i]=a*(e*l-f*k)-b*(d*l-f*g)+c*(d*k-e*g);
      v64[i]=double(naturalVolume[i])*j64[i];
      r64[i]=double(naturalDensity[i])/j64[i];
      if(!std::isfinite(j64[i]) || j64[i]<=0 || !std::isfinite(v64[i]) || v64[i]<=0 ||
         !std::isfinite(r64[i]) || r64[i]<=0)throw std::runtime_error("Invalid double elastic reference material.");
    }
    tmatrix3d* hg=nullptr;double *jd=nullptr,*vd=nullptr,*rd=nullptr;
    Data->Alloc(hg,n);Data->Alloc(jd,n);Data->Alloc(vd,n);Data->Alloc(rd,n);
    Impl::Check(cudaMemcpy(hg,Data->HostHstar64.data(),n*sizeof(tmatrix3d),cudaMemcpyHostToDevice));
    Impl::Check(cudaMemcpy(jd,j64.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    Impl::Check(cudaMemcpy(vd,v64.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    Impl::Check(cudaMemcpy(rd,r64.data(),n*sizeof(double),cudaMemcpyHostToDevice));
    Data->Restart.Hstar64=hg;Data->Restart.Jstar64=jd;
    Data->Restart.Volume64=vd;Data->Restart.Density64=rd;
  }
}
// Host methods used only at initialisation and saved frames.
void UserGFlexPrecision::SetRestartH64(unsigned n,const tmatrix3d* h){
  if(!Data || Data->Mode!=2 || Data->Particles!=n || Data->Restart.Fstar)
    throw std::runtime_error("Invalid double restart setup order.");
  for(unsigned i=0;i<n;i++)for(int k=0;k<9;k++)
    if(!std::isfinite((&h[i].a11)[k]))throw std::runtime_error("Non-finite double restart H.");
  Data->HostHstar64.assign(h,h+n);
}
tmatrix3d UserGFlexPrecision::TotalH64(unsigned i,const tmatrix3d& relative)const{
  if(!Data || Data->Mode!=2 || i>=Data->Particles)
    throw std::runtime_error("Double elastic output cache mismatch.");
  if(Data->HostHstar64.empty())return relative;
  const auto& star=Data->HostHstar64[i];tmatrix3d total;
  // (I+Hr)(I+Hs)-I = Hs+Hr+Hr*Hs. Never subtract I from rounded F.
  for(int r=0;r<3;r++)for(int c=0;c<3;c++){
    double v=(&star.a11)[3*r+c]+(&relative.a11)[3*r+c];
    for(int k=0;k<3;k++)v+=(&relative.a11)[3*r+k]*(&star.a11)[3*k+c];
    (&total.a11)[3*r+c]=v;
  }
  return total;
}
void UserGFlexPrecision::DownloadTotalH64(unsigned n,std::vector<tmatrix3d>& total)const{
  if(!Data || Data->Mode!=2 || Data->Particles!=n)
    throw std::runtime_error("Double elastic save cache mismatch.");
  static_assert(sizeof(UserGFlexMatrix<double>)==sizeof(tmatrix3d),"Double H layout mismatch");
  total.resize(n);
  Impl::Check(cudaMemcpy(total.data(),Data->Extended.H,n*sizeof(tmatrix3d),cudaMemcpyDeviceToHost));
  for(unsigned i=0;i<n;i++)total[i]=TotalH64(i,total[i]);
}
size_t UserGFlexPrecision::GetAllocMemoryGpu()const{return Data?Data->Bytes+Data->WorkCapacity:0;}
void* UserGFlexPrecision::WorkScratch(size_t bytes){
  if(!bytes)return nullptr;
  if(!Data)Data=new Impl;
  if(bytes>Data->WorkCapacity){
    void* replacement=nullptr;
    Impl::Check(cudaMalloc(&replacement,bytes));
    if(Data->WorkBuffer)cudaFree(Data->WorkBuffer);
    Data->WorkBuffer=replacement;Data->WorkCapacity=bytes;
  }
  return Data->WorkBuffer;
}

// Same binary max tree as ReduMaxFloat. Separate lengths preserve each
// channel's padding/tree, including non-finite values; no sums are reordered.
__global__ void KerUserTimeStepMaxPair(unsigned na,const float* a,unsigned nb,const float* b
  ,float* oa,float* ob){
  __shared__ float va[256],vb[256];
  const unsigned t=threadIdx.x,i=blockIdx.x*256+t;
  va[t]=(i<na?a[i]:-FLT_MAX);vb[t]=(i<nb?b[i]:-FLT_MAX);
  __syncthreads();
  for(unsigned stride=128;stride;stride>>=1){
    if(t<stride){va[t]=max(va[t],va[t+stride]);vb[t]=max(vb[t],vb[t+stride]);}
    __syncthreads();
  }
  if(!t){
    if(blockIdx.x<(na+255)/256)oa[blockIdx.x]=(na==1?a[0]:va[0]);
    if(blockIdx.x<(nb+255)/256)ob[blockIdx.x]=(nb==1?b[0]:vb[0]);
  }
}
float2 UserGFlexPrecision::TimeStepMaxPair(unsigned na,const float* a,unsigned nb,const float* b){
  // Per-solver scratch is independent of cell-sort workspace capacity. Each channel
  // has ping-pong storage; final scalars occupy adjacent scratch entries.
  const unsigned nmax=(na>nb?na:nb),cap=(nmax+255)/256;
  if(!nmax)return make_float2(0,0);
  float* scratch=static_cast<float*>(WorkScratch((size_t(cap)*4+2)*sizeof(float)));
  float* a0=scratch+2;float* a1=a0+cap;
  float* b0=a1+cap;float* b1=b0+cap;
  unsigned ca=na,cb=nb;const float *pa=a,*pb=b;bool flip=false;
  while(ca>1 || cb>1){
    const unsigned ba=(ca+255)/256,bb=(cb+255)/256;
    float* qa=(ca<=256?scratch:(flip?a1:a0));
    float* qb=(cb<=256?scratch+1:(flip?b1:b0));
    KerUserTimeStepMaxPair<<<ba>bb?ba:bb,256>>>(ca,pa,cb,pb,qa,qb);
    ca=ba;cb=bb;pa=qa;pb=qb;flip=!flip;
  }
  // Handle singleton/empty arrays explicitly without changing max semantics.
  if(na<=1){if(na)cudaMemcpy(scratch,a,sizeof(float),cudaMemcpyDeviceToDevice);else cudaMemset(scratch,0,sizeof(float));}
  if(nb<=1){if(nb)cudaMemcpy(scratch+1,b,sizeof(float),cudaMemcpyDeviceToDevice);else cudaMemset(scratch+1,0,sizeof(float));}
  float2 result;Impl::Check(cudaMemcpy(&result,scratch,sizeof(result),cudaMemcpyDeviceToHost));
  return result;
}

// Host launch selection only. No precision switch is evaluated on the GPU.
template<class Math,bool corrected> class UserGFlexPrecisionBackend{
  using State=typename Math::State;
  static void Check(){
    const auto e=cudaGetLastError();
    if(e!=cudaSuccess)throw std::runtime_error(std::string("GPU elastic precision launch: ")+cudaGetErrorString(e));
  }
  template<TpKernel kernel,TpVisco visco,TpMdbc2Mode mdbc,bool sim2d>
  static void Force(const StInterParmsFlexStrucg& t,State s){
    const unsigned blocks=(t.vnpfs+SPHBSIZE-1)/SPHBSIZE;
    cusph::usergflex::KerUserGFlexPrecisionForces<Math,kernel,visco,mdbc,sim2d,corrected>
      <<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
  }
  template<TpKernel kernel,TpVisco visco,bool sim2d>
  static void Boundary(const StInterParmsFlexStrucg& t,State s){
#ifdef FAST_COMPILATION
    if(t.mdbc2!=MDBC2_None)throw "Extra mDBC options are disabled for FastCompilation...";
    Force<kernel,visco,MDBC2_None,sim2d>(t,s);
#else
    if(t.mdbc2==MDBC2_None)Force<kernel,visco,MDBC2_None,sim2d>(t,s);
    else if(t.mdbc2==MDBC2_Std)Force<kernel,visco,MDBC2_Std,sim2d>(t,s);
    else if(t.mdbc2==MDBC2_NoPen)Force<kernel,visco,MDBC2_NoPen,sim2d>(t,s);
    else throw "Unsupported elastic mDBC mode.";
#endif
  }
  template<UserGFlexStage stage,TpKernel kernel,bool sim2d>
  static void Launch(const StInterParmsFlexStrucg& t,State s){
    const unsigned blocks=(t.vnpfs+SPHBSIZE-1)/SPHBSIZE;
    if(stage==UserGFlexStage::Reference){
      cusph::usergflex::KerUserGFlexPrecision<Math,0,kernel,sim2d,corrected><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      if(!corrected)cusph::usergflex::KerUserGFlexPrecision<Math,1,kernel,sim2d,corrected><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      cusph::usergflex::KerUserGFlexPrecision<Math,2,kernel,sim2d,corrected><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      Check();
      // One 4-byte validation download at initialization, never in the time loop.
      unsigned error=0;
      auto e=cudaMemcpyAsync(&error,s.Error,sizeof(error),cudaMemcpyDeviceToHost,t.stm);
      if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
      e=cudaStreamSynchronize(t.stm);
      if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
      if(error)throw std::runtime_error("Singular elastic reference correction matrix.");
    }
    else if(stage==UserGFlexStage::Geometry){
      cusph::usergflex::KerUserGFlexPrecision<Math,3,kernel,sim2d,corrected><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      cusph::usergflex::KerUserGFlexPrecision<Math,4,kernel,sim2d,corrected><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      if(t.mdbc2>=MDBC2_Std)
        cusph::usergflex::KerUserGFlexPrecision<Math,5,kernel,sim2d,corrected><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
    }
    else{
#ifdef FAST_COMPILATION
      if(t.tvisco!=VISCO_Artificial)throw "Extra viscosity options are disabled for FastCompilation...";
      Boundary<kernel,VISCO_Artificial,sim2d>(t,s);
#else
      if(t.tvisco==VISCO_Artificial)Boundary<kernel,VISCO_Artificial,sim2d>(t,s);
      else if(t.tvisco==VISCO_Laminar)Boundary<kernel,VISCO_Laminar,sim2d>(t,s);
      else if(t.tvisco==VISCO_LaminarSPS)Boundary<kernel,VISCO_LaminarSPS,sim2d>(t,s);
      else throw "Unsupported elastic viscosity.";
#endif
    }
    Check();
  }
  template<UserGFlexStage stage,TpKernel kernel>
  static void Dimension(const StInterParmsFlexStrucg& t,State s){
    if(t.simulate2d)Launch<stage,kernel,true>(t,s);
    else Launch<stage,kernel,false>(t,s);
  }
public:
  template<UserGFlexStage stage> static void Run(const StInterParmsFlexStrucg& input){
    if(!input.vnpfs)return;
    if(!input.precision || !input.posxy || !input.posz)
      throw std::runtime_error("Missing elastic precision context or double positions.");
    const State s=*static_cast<State*>(input.precision->State(sizeof(typename Math::Real)==4?1:2,input.vnpfs));
    // No host object address is passed into the device kernels.
    auto t=input;t.user=nullptr;t.precision=nullptr;
#ifdef FAST_COMPILATION
    if(t.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
    Dimension<stage,KERNEL_Wendland>(t,s);
#else
    if(t.tkernel==KERNEL_Wendland)Dimension<stage,KERNEL_Wendland>(t,s);
#ifndef DISABLE_KERNELS_EXTRA
    else if(t.tkernel==KERNEL_Cubic)Dimension<stage,KERNEL_Cubic>(t,s);
#endif
    else throw "Unsupported elastic kernel.";
#endif
  }
};


class UserGFlexDispatch{
private:
  UserGFlexDispatch()=delete;

public:
  template<UserGFlexStage stage>
  static void Run(const StInterParmsFlexStrucg& tfs){
    using Mode=UserGParameters::TpFlexPrecisionMode;
    using OriginalMath=cusph::usergflex::UserGFlexMath<Mode::Original>;
    const auto config=UserGParameters::ReadFlex(tfs.user);

    switch(config.PrecisionMode){
      case Mode::Original:
        if(tfs.restart.Corrected)UserGFlexBackend<OriginalMath,true>::template Run<stage>(tfs);
        else UserGFlexBackend<OriginalMath,false>::template Run<stage>(tfs);
        return;
      case Mode::Mixed:
        if(tfs.restart.Corrected)UserGFlexPrecisionBackend<cusph::usergflex::UserGFlexPrecisionMath<float>,true>::template Run<stage>(tfs);
        else UserGFlexPrecisionBackend<cusph::usergflex::UserGFlexPrecisionMath<float>,false>::template Run<stage>(tfs);
        return;
      case Mode::Extended:
        if(tfs.restart.Corrected)UserGFlexPrecisionBackend<cusph::usergflex::UserGFlexPrecisionMath<double>,true>::template Run<stage>(tfs);
        else UserGFlexPrecisionBackend<cusph::usergflex::UserGFlexPrecisionMath<double>,false>::template Run<stage>(tfs);
        return;
    }
    throw "Invalid GPUFlexPrecisionMode.";
  }
};

void UserGInteractionForcesFlexStruc::Interaction_Contact(const StrInterParmsFlexStrucg& tfs){
  using namespace cusph::usergflex;
  if(!tfs.vnpfs)return;
  const auto c=UserGParameters::ReadContact(tfs.user);
  if(c.KnMode==1 && c.Kn==0 && c.DampingMode==1 && c.Damping==0)return;
  if(c.Mode==2 && c.Surface==0)return;
  const UserGFlexContactData d={tfs.vnpfs,unsigned(tfs.divdatag.axis),tfs.dp,tfs.divdatag,tfs.dcell
    ,tfs.flexstrucridp,tfs.code,tfs.posxy,tfs.posz,tfs.velrhop,tfs.flexstrucdata
    ,tfs.contactmass,tfs.ace};
  switch(unsigned(UserGParameters::ReadFlex(tfs.user).PrecisionMode)){
    case 0: UserGFlexContactLaunch::Run<0>(d,c,tfs.stm); break;
    case 1: UserGFlexContactLaunch::Run<1>(d,c,tfs.stm); break;
    case 2: UserGFlexContactLaunch::Run<2>(d,c,tfs.stm); break;
    default: throw std::runtime_error("Invalid GPUFlexPrecisionMode for contact.");
  }
}

void UserGInteractionForcesFlexStruc::CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
  UserGFlexDispatch::Run<UserGFlexStage::Reference>(tfs);
}

void UserGInteractionForcesFlexStruc::UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
  UserGFlexDispatch::Run<UserGFlexStage::Geometry>(tfs);
}

void UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
  UserGFlexDispatch::Run<UserGFlexStage::Forces>(tfs);
  Interaction_Contact(tfs);
}

void UserGInteractionForcesFlexStruc::CteInteractionUp(const StCteInteraction* cte){
  cudaMemcpyToSymbol(cusph::usergflex::CTE,cte,sizeof(StCteInteraction));
}

// Preserve the existing solver API; implementation belongs to the class above.
namespace cusph{
void CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
  UserGInteractionForcesFlexStruc::CalcFlexStrucKerCorr(tfs);
}

void UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
  UserGInteractionForcesFlexStruc::UpdateFlexStrucGeometry(tfs);
}

void Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
  UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStruc(tfs);
}
}

#undef CTE_AVAILABLE

void UserGFlexCte::Upload(const StCteInteraction* cte){
  UserGInteractionForcesFlexStruc::CteInteractionUp(cte);
}
