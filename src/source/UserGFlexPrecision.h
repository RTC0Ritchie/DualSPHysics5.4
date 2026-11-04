#ifndef _UserGFlexPrecision_
#define _UserGFlexPrecision_
#include <cstddef>
#include <vector>
#include "TypesDef.h"

struct UserGFlexRestartView{
  bool Corrected=false;
  const tfloat3 *GradI=nullptr,*GradJ=nullptr,*MapI=nullptr,*MapJ=nullptr;
  const float* HgWeight=nullptr;
  const tmatrix3f* Hoffset=nullptr;
  const tdouble3 *GradI64=nullptr,*GradJ64=nullptr,*MapI64=nullptr,*MapJ64=nullptr;
  const double* HgWeight64=nullptr;
  const tmatrix3d* Hoffset64=nullptr;
  const tmatrix3f* Fstar=nullptr;
  // Optional authoritative mode-2 state. Float mirrors serve modes 0/1 only.
  const tmatrix3d* Hstar64=nullptr;
  const double *Jstar64=nullptr,*Volume64=nullptr,*Density64=nullptr;
  const float *Jstar=nullptr,*Volume=nullptr,*Density=nullptr,*Mass=nullptr;
};
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
  void UploadRestart(unsigned n,const tmatrix3f* f,const float* j,
      const float* volume,const float* density,const float* mass,
      const float* naturalVolume,const float* naturalDensity);
  void SetRestartH64(unsigned n,const tmatrix3d* h);
  tmatrix3d TotalH64(unsigned i,const tmatrix3d& relative)const;
  void DownloadTotalH64(unsigned n,std::vector<tmatrix3d>& total)const;
  UserGFlexRestartView Restart()const;
  void Allocate(unsigned mode,unsigned particles,unsigned pairs);
  void AllocatePairs(unsigned pairs);
  void SetNaturalCorrection64(const std::vector<tmatrix3d>& b);
  const std::vector<tmatrix3d>& NaturalCorrection64()const;
  void DownloadCorrection64(unsigned n,std::vector<tmatrix3d>& b)const;
  void UploadCorrections(unsigned n,unsigned pairs,const tdouble3* gi,const tdouble3* gj,
      const tdouble3* mi,const tdouble3* mj,const double* weight,const tmatrix3d* h0);
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
