#ifndef _UserGFlexPrecision_
#define _UserGFlexPrecision_
#include <cstddef>

// Per-solver owner. Its layout does not expose precision policy or CUDA buffers.
class UserGFlexPrecision{
  class Impl;
  Impl* Data;
public:
  UserGFlexPrecision();
  ~UserGFlexPrecision();
  UserGFlexPrecision(const UserGFlexPrecision&)=delete;
  UserGFlexPrecision& operator=(const UserGFlexPrecision&)=delete;
  void Reset();
  void Allocate(unsigned mode,unsigned particles,unsigned pairs);
  void* State(unsigned mode,unsigned particles)const;
  size_t GetAllocMemoryGpu()const;
};
#endif
