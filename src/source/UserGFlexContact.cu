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

#include "UserGFlexContact.h"
#include "UserGFlexTypes.h"
#include "UserGParameters.h"
#include "UserGFlexContact_iker.h"
#include "JCellSearch_iker.h"
#include <stdexcept>


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

void UserGFlexContact::Run(const StrInterParmsFlexStrucg& tfs){
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
