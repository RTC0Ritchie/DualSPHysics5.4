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

#ifndef _UserGFlexContact_iker_
#define _UserGFlexContact_iker_

#include "UserGParameters.h"
#include "DualSphDef.h"
#include <type_traits>
#include <cmath>

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
#endif
