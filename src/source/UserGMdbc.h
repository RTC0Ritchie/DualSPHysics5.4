#ifndef _UserGMdbc_
#define _UserGMdbc_
#include "DualSphDef.h"
#include <vector_types.h>
class UserParameters;
// Stable bridge for the existing mDBC kernel. No UserParameters dependency.
class UserGMdbc{
public:
  struct History{
    int Step;double Head,Delta;float3 Filtered;float Weight;unsigned Calm;
  };
  struct View{History* State;int* Step;unsigned Count;float Support;double* BottomSupport;unsigned BottomCount;};
  static void Upload(const View& view);
private:
  class Impl;Impl* Data;
public:
  UserGMdbc();
  ~UserGMdbc();
  UserGMdbc(const UserGMdbc&)=delete;
  UserGMdbc& operator=(const UserGMdbc&)=delete;
  size_t GetAllocMemoryGpu()const;
  double* EnableBottomSupport(unsigned count);
  void ClearBottomSupport();
  const float3* NoPenNormals(const UserParameters* user,unsigned np,unsigned nb,double dp,
      const unsigned* id,const typecode* code,const float3* normals);
  const float3* Acceleration(const UserParameters* user,unsigned np,unsigned nb,unsigned count,
      unsigned nf,int step,bool twoD,tfloat3 gravity,const unsigned* id,const typecode* code,
      const float3* normals,const float3* acceleration);
};
#endif
