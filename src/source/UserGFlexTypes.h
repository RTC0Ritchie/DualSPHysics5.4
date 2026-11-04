#ifndef _UserGFlexTypes_
#define _UserGFlexTypes_

#include "JSphGpu_ker.h"

class UserGFlexPrecision;

struct StrInterParmsFlexStrucg{
  //-Configuration options.
  bool simulate2d;
  TpKernel tkernel;
  TpVisco tvisco;
  TpMdbc2Mode mdbc2;
  //-Execution values.
  float viscob;
  unsigned vnpfs;
  StDivDataGpu divdatag;
  //-Input data arrays.
  const unsigned* dcell;
  const float4* poscell;
  const float4* velrhop;
  const typecode* code;
  const byte* boundmode;
  const float3* tangenvel;
  const StFlexStrucData* flexstrucdata;
  const unsigned* flexstrucridp;
  const float4* poscell0;
  const unsigned* numpairs;
  const unsigned* const* pairidx;
  const tmatrix3f* kercorr;
  const float3* boundnor0;
  //-Output data arrays.
  tmatrix3f* defgrad;
  float3* boundnor;
  float* flexstrucdt;
  float3* ace;
  //-Other values and objects.
  cudaStream_t stm;

  // Host-side user configuration; never dereference on the GPU.
  const UserParameters* user;
  UserGFlexPrecision* precision; // Host-only owner.
  const double2* posxy;
  const double* posz;
  double dp; // Preserve the CPU double spacing for analytic surface geometry.
  // Optional device mass indexed by stable elastic index (future restart adapter).
  const float* contactmass=nullptr;

  ///Structure constructor.
  StrInterParmsFlexStrucg(
       bool simulate2d,TpKernel tkernel,TpVisco tvisco,TpMdbc2Mode mdbc2
      ,float viscob,unsigned vnpfs
      ,const StDivDataGpu& divdatag
      ,const unsigned* dcell
      ,const float4* poscell,const float4* velrhop,const typecode* code
      ,const byte* boundmode,const float3* tangenvel
      ,const StFlexStrucData* flexstrucdata
      ,const unsigned* flexstrucridp,const float4* poscell0
      ,const unsigned* numpairs,const unsigned* const* pairidx
      ,const tmatrix3f* kercorr,const float3* boundnor0
      ,tmatrix3f* defgrad,float3* boundnor,float* flexstrucdt,float3* ace
      ,cudaStream_t stm,const UserParameters* user
      ,UserGFlexPrecision* precision,const double2* posxy,const double* posz,double dp)
  {
    //-Configuration options.
    this->simulate2d=simulate2d; this->tkernel=tkernel; this->tvisco=tvisco; this->mdbc2=mdbc2;
    //-Execution values.
    this->viscob=viscob;
    this->vnpfs=vnpfs;
    this->divdatag=divdatag;
    //-Input data arrays.
    this->dcell=dcell;
    this->poscell=poscell; this->velrhop=velrhop; this->code=code;
    this->boundmode=boundmode; this->tangenvel=tangenvel;
    this->flexstrucdata=flexstrucdata;
    this->flexstrucridp=flexstrucridp; this->poscell0=poscell0;
    this->numpairs=numpairs; this->pairidx=pairidx;
    this->kercorr=kercorr; this->boundnor0=boundnor0;
    //-Output data arrays.
    this->defgrad=defgrad; this->boundnor=boundnor; this->flexstrucdt=flexstrucdt; this->ace=ace;
    //-Other values and objects.
    this->stm=stm;
    this->user=user;
    this->dp=dp;
    this->precision=precision; this->posxy=posxy; this->posz=posz;
  }
};

#endif
