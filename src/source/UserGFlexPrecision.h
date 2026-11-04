#ifndef _UserGFlexPrecision_
#define _UserGFlexPrecision_
#include <cstddef>
#include <vector_types.h>

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
  // Ephemeral GPU scratch owned by this solver; reused until growth/reset.
  // Calls are serial on the default stream. Do not retain this pointer.
  void* WorkScratch(size_t bytes);
  // Independent maxima, one synchronous host transfer. Inputs remain unchanged.
  float2 TimeStepMaxPair(unsigned na,const float* a,unsigned nb,const float* b);
};

// Shared device-state layout used by the elastic solver and user output.
// CUDA computation and allocation implementations live in
// UserGInteractionForcesFlexStruc.cu; keep them out of this public header.
// All indices below are elastic indices, independent of the particle sort order.
// State is canonical; defgrad/kercorr in the solver are one-way float mirrors.
template<class R> struct UserGFlexMatrix{ R v[9]; };
template<class R> struct UserGFlexVector{ R v[3]; };
template<class R> struct UserGFlexState{
  double3 *Reference,*Displacement;
  UserGFlexVector<R>* PairReference;
  R* PairNorm;
  UserGFlexVector<R>* PairGradient; // Only extended mode allocates this cache.
  UserGFlexMatrix<R> *Correction,*H,*PK,*PB;
  unsigned* Error;
};

#endif
