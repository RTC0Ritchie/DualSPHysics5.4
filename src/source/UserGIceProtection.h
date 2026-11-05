#ifndef USER_G_ICE_PROTECTION_H
#define USER_G_ICE_PROTECTION_H
#include "DualSphDef.h"
#include <vector_types.h>
#include <vector>
class UserParameters;
class UserGIceProtection{
  class Impl;Impl* Data;
public:
  static bool CleanupEnabled(const UserParameters* user);
  struct Reference{double x,z;unsigned id;};
  struct Domain{tdouble3 Min,Size;float Cell,MoveLimit;unsigned CellCode;};
  UserGIceProtection(const std::vector<Reference>& ref,bool top,bool right,bool bottom,
      double dp,double h,double wettime,double hcoef,unsigned nbound,double* support,bool inout=false);
  ~UserGIceProtection();
  UserGIceProtection(const UserGIceProtection&)=delete;
  UserGIceProtection& operator=(const UserGIceProtection&)=delete;
  size_t GetAllocMemoryGpu()const;
  void ValidateBuffers(unsigned np,unsigned nb,const unsigned* ids,const typecode* code,
      const double2* xy,const double* z);
  void Capture(bool predictor,double time,unsigned np,unsigned nb,const unsigned* ids,
      const typecode* code,const double2* xy,const double* z,const float4* vel,
      const double2* oldxy,const double* oldz);
  void Apply(unsigned np,unsigned nb,const unsigned* ids,typecode* code,
      double2* xy,double* z,float4* vel,unsigned* cells,const Domain& domain);
};
#endif
