#ifndef _UserGParameterOutput_
#define _UserGParameterOutput_
#include "TypesDef.h"
#include "DualSphDef.h"
#include <cuda_runtime.h>
#include <vector>
class UserGFlexPrecision;

class UserGParameterOutput{
public:
  struct FlexParticle{
    unsigned Idp,Code,Pairs;
    tmatrix3f F;
    tmatrix3d H;
  };
  struct InletParticle{
    unsigned Idp;
    tdouble3 Pos;
    tfloat4 VelRho;
  };
  static void DownloadFlex(unsigned n,unsigned np,unsigned mode,UserGFlexPrecision* context
    ,const unsigned* rid,const unsigned* idp,const typecode* code
    ,const unsigned* pairs,const tmatrix3f* f,bool tensors
    ,std::vector<FlexParticle>& out);
  static void DownloadInlet(UserGFlexPrecision* context,unsigned n,unsigned firstId,const unsigned* idp
    ,const typecode* code,const double2* xy,const double* z,const float4* velrho
    ,std::vector<InletParticle>& out);
};
#endif
