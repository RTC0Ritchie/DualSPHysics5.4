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
#include "JSphGpu_ker.h"
#include "Functions.h"
#include "FunctionsCuda.h"
#include <cfloat>
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
class UserGFlexStrucDevice{
public:
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
};

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> __global__ void KerInteractionForcesFlexStruc(unsigned n,float visco
    ,int scelldiv,int4 nc,int3 cellzero,const int2* beginendcellfluid,const unsigned* dcell
    ,const float4* poscell,const float4* velrhop,const typecode* code
    ,const byte* boundmode,const float3* tangenvel
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr,const tmatrix3f* defgrad
    ,float* flexstrucdt,float3* ace)
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
      const float vol0p1=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].vol0;
      const float rho0p1=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].rho0;
      const float youngmod=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].youngmod;
      const float poissratio=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].poissratio;
      const float hgfactor=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].hgfactor;
      const tmatrix6f cmat=flexstrucdata[CODE_GetIbodyFlexStruc(codep1)].cmat;
      const tmatrix3f pk1p1=UserGFlexStrucDevice::CalcPK1Stress(defgradp1,cmat);
      const tmatrix3f pk1kercorrp1=cumath::MulMatrix3x3(pk1p1,kercorrp1);

      //-Get current mass of flexible structure particle.
      const float mass0p1=vol0p1*rho0p1;
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
                  const float fac=UserGFlexStrucDevice::GetKernel_Fac<tker>(rr2);
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
        const float4 pscell0p2=poscell0[pfs2];
        float drx0=pscell0p1.x-pscell0p2.x+CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        float dry0=pscell0p1.y-pscell0p2.y+CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        float drz0=pscell0p1.z-pscell0p2.z+CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        const float fac0=UserGFlexStrucDevice::GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        //-Acceleration due to structure.
        const tmatrix3f defgradp2=defgrad[pfs2];
        const tmatrix3f pk1p2=UserGFlexStrucDevice::CalcPK1Stress(defgradp2,cmat);
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
        acep1.x+=pk1kercorrdw.x*vol0p1/rho0p1; acep1.y+=pk1kercorrdw.y*vol0p1/rho0p1; acep1.z+=pk1kercorrdw.z*vol0p1/rho0p1;
        //-Hourglass correction.
        if(hgfactor){
          const unsigned p2=flexstrucridp[pfs2];
          const float4 pscellp2=poscell[p2];
          float drx=pscellp1.x-pscellp2.x + CTE.poscellsize*(PSCEL_GetfX(pscellp1.w)-PSCEL_GetfX(pscellp2.w));
          float dry=pscellp1.y-pscellp2.y + CTE.poscellsize*(PSCEL_GetfY(pscellp1.w)-PSCEL_GetfY(pscellp2.w));
          float drz=pscellp1.z-pscellp2.z + CTE.poscellsize*(PSCEL_GetfZ(pscellp1.w)-PSCEL_GetfZ(pscellp2.w));
          const float rr2=drx*drx+dry*dry+drz*drz;
          const float wab=UserGFlexStrucDevice::GetKernel_Wab<tker>(rr2);
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
          const float mulfac=(hgfactor*vol0p1*vol0p1*wab*youngmod/(rr20*rr*mass0p1)*0.5f*(deltaij+deltaji));
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

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,bool simulate2d> __global__ void KerCalcFlexStrucKerCorr(unsigned n,const typecode* code,const StFlexStrucData* flexstrucdata
    ,const unsigned* flexstrucridp,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx
    ,tmatrix3f* kercorr)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
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
        const float4 pscell0p2=poscell0[pfs2];
        float drx0=pscell0p1.x-pscell0p2.x + CTE.poscellsize*(PSCEL_GetfX(pscell0p1.w)-PSCEL_GetfX(pscell0p2.w));
        float dry0=pscell0p1.y-pscell0p2.y + CTE.poscellsize*(PSCEL_GetfY(pscell0p1.w)-PSCEL_GetfY(pscell0p2.w));
        float drz0=pscell0p1.z-pscell0p2.z + CTE.poscellsize*(PSCEL_GetfZ(pscell0p1.w)-PSCEL_GetfZ(pscell0p2.w));
        const float rr20=drx0*drx0+dry0*dry0+drz0*drz0;
        //-Computes kernel.
        const float fac0=UserGFlexStrucDevice::GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        kercorrp1.a11-=vol0p1*drx0*frx0; kercorrp1.a12-=vol0p1*drx0*fry0; kercorrp1.a13-=vol0p1*drx0*frz0;
        kercorrp1.a21-=vol0p1*dry0*frx0; kercorrp1.a22-=vol0p1*dry0*fry0; kercorrp1.a23-=vol0p1*dry0*frz0;
        kercorrp1.a31-=vol0p1*drz0*frx0; kercorrp1.a32-=vol0p1*drz0*fry0; kercorrp1.a33-=vol0p1*drz0*frz0;
      }
      if(simulate2d){
        kercorrp1.a12=kercorrp1.a21=kercorrp1.a23=kercorrp1.a32=0.0;
        kercorrp1.a22=1.0;
      }
      kercorr[pfs1]=cumath::InverseMatrix3x3(kercorrp1);
    }
  }
}

template<TpKernel tker,bool simulate2d> __global__ void KerCalcFlexStrucDefGrad(unsigned n,const float4* poscell,const typecode* code
    ,const StFlexStrucData* flexstrucdata,const unsigned* flexstrucridp
    ,const float4* poscell0,const unsigned* numpairs,const unsigned* const* pairidx,const tmatrix3f* kercorr
    ,tmatrix3f* defgrad)
{
  const unsigned p=blockIdx.x*blockDim.x + threadIdx.x; //-Number of thread.
  if(p<n){
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
        const float fac0=UserGFlexStrucDevice::GetKernel_Fac<tker>(rr20);
        const float frx0=fac0*drx0,fry0=fac0*dry0,frz0=fac0*drz0; //-Gradients.
        defgradp1.a11-=vol0p1*drx*frx0; defgradp1.a12-=vol0p1*drx*fry0; defgradp1.a13-=vol0p1*drx*frz0;
        defgradp1.a21-=vol0p1*dry*frx0; defgradp1.a22-=vol0p1*dry*fry0; defgradp1.a23-=vol0p1*dry*frz0;
        defgradp1.a31-=vol0p1*drz*frx0; defgradp1.a32-=vol0p1*drz*fry0; defgradp1.a33-=vol0p1*drz*frz0;
      }
      if(simulate2d){
        defgradp1.a12=defgradp1.a21=defgradp1.a23=defgradp1.a32=0.0;
        defgradp1.a22=1.0;
      }
      defgrad[pfs1]=cumath::MulMatrix3x3(defgradp1,kercorrp1);
    }
  }
}

__global__ void KerCalcFlexStrucNormals(unsigned n,const unsigned* flexstrucridp
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

} // namespace usergflex
} // namespace cusph

template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> void UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStrucT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    const StDivDataGpu& dvd=tfs.divdatag;
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerInteractionForcesFlexStruc<tker,tvisco,mdbc2> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.viscob,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell+dvd.cellfluid,tfs.dcell,tfs.poscell,tfs.velrhop,tfs.code,tfs.boundmode,tfs.tangenvel,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad,tfs.flexstrucdt,tfs.ace);
  }
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<TpKernel tker,TpVisco tvisco> void UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStruc_gt1(const StInterParmsFlexStrucg& tfs){
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
template<TpKernel tker> void UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStruc_gt0(const StInterParmsFlexStrucg& tfs){
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
void UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
  const UserGParameters::StFlexConfig usercfg=
      UserGParameters::ReadFlex(tfs.user);

  // Reserved for the precision-policy dispatch.
  // All requested modes currently execute the existing implementation.
  (void)usercfg;
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

template<TpKernel tker,bool simulate2d> void UserGInteractionForcesFlexStruc::CalcFlexStrucKerCorrT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucKerCorr<tker,simulate2d> <<<sgridb,SPHBSIZE>>>
        (tfs.vnpfs,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,const_cast<tmatrix3f*>(tfs.kercorr));
  }
}

template<TpKernel tker> void UserGInteractionForcesFlexStruc::CalcFlexStrucKerCorr_gt0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucKerCorrT<tker,true>  (tfs);
  else              CalcFlexStrucKerCorrT<tker,false> (tfs);
}

void UserGInteractionForcesFlexStruc::CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
  const UserGParameters::StFlexConfig usercfg=
      UserGParameters::ReadFlex(tfs.user);

  // Reserved for the precision-policy dispatch.
  // All requested modes currently execute the existing implementation.
  (void)usercfg;
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

template<TpKernel tker,bool simulate2d> void UserGInteractionForcesFlexStruc::CalcFlexStrucDefGradT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucDefGrad<tker,simulate2d> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.poscell,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad);
  }
}

template<TpKernel tker> void UserGInteractionForcesFlexStruc::CalcFlexStrucDefGrad_ct0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucDefGradT<tker,true>  (tfs);
  else              CalcFlexStrucDefGradT<tker,false> (tfs);
}

void UserGInteractionForcesFlexStruc::UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
  const UserGParameters::StFlexConfig usercfg=
      UserGParameters::ReadFlex(tfs.user);

  // Reserved for the precision-policy dispatch.
  // All requested modes currently execute the existing implementation.
  (void)usercfg;
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
    cusph::usergflex::KerCalcFlexStrucNormals <<<sgridb,SPHBSIZE,0,tfs.stm>>>(tfs.vnpfs,tfs.flexstrucridp,tfs.defgrad,tfs.boundnor0,tfs.boundnor);
  }
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
