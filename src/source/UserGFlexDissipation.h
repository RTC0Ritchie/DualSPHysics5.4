#ifndef _UserGFlexDissipation_
#define _UserGFlexDissipation_
#include "DualSphDef.h"
#include <functional>
#include <vector_types.h>
class UserGFlexPrecision;
class UserGFlexDissipation{
  class Impl;
  Impl* Data;
public:
  UserGFlexDissipation(bool twoD,unsigned n,const tdouble3* x0,
      const double* volume,const double* rho,const double* young,
      const double* poisson,const unsigned* body,const unsigned* count,
      const unsigned* const* pairs,const tmatrix3f* correction,
      bool mode5,const std::function<float(float)>& fac,
      double h,bool useCR,bool useKV,double factor,unsigned mode,int simplifiedMode=1);

  ~UserGFlexDissipation();
  UserGFlexDissipation(const UserGFlexDissipation&)=delete;
  UserGFlexDissipation& operator=(const UserGFlexDissipation&)=delete;
  void Apply(unsigned n,const double2* xy,const double* z,const float4* velocity,
      const typecode* code,const unsigned* rid,const tmatrix3f* f,
      unsigned mode,UserGFlexPrecision* precision,float3* ace,float* speed);
  size_t GetAllocMemoryGpu()const;
};
#endif
