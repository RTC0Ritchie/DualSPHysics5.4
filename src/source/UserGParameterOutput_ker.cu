#include "UserGParameterOutput.h"
#include "UserGFlexPrecision.h"
#include "UserGFlexPrecisionState.h"
#include <stdexcept>
#include <string>

namespace{
class OutputCuda{
public:
  static void Check(cudaError_t e){
    if(e!=cudaSuccess)throw std::runtime_error(std::string("GPU user output: ")+cudaGetErrorString(e));
  }
};
template<class T> class OutputBuffer{
public:
  T* Ptr=nullptr;
  explicit OutputBuffer(size_t n){if(n)OutputCuda::Check(cudaMalloc((void**)&Ptr,sizeof(T)*n));}
  ~OutputBuffer(){if(Ptr)cudaFree(Ptr);}
  OutputBuffer(const OutputBuffer&)=delete;
  OutputBuffer& operator=(const OutputBuffer&)=delete;
};
template<class R,bool enhanced>
__global__ void GatherFlex(unsigned n,unsigned np,const unsigned* rid,const unsigned* idp
  ,const typecode* code,const unsigned* pairs,const tmatrix3f* f
  ,const UserGFlexMatrix<R>* h,bool tensors,UserGParameterOutput::FlexParticle* out){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const unsigned p=rid[i];UserGParameterOutput::FlexParticle a={};
  if(p>=np){a.Idp=0xffffffffu;out[i]=a;return;}
  a.Idp=idp[p];a.Code=unsigned(code[p]);a.Pairs=pairs[i];
  if(tensors){
    a.F=f[i];
    if(enhanced)for(int k=0;k<9;k++)(&a.H.a11)[k]=double(h[i].v[k]);
  }
  out[i]=a;
}
__global__ void GatherInlet(unsigned n,unsigned firstId,const unsigned* idp
  ,const typecode* code,const double2* xy,const double* z,const float4* vr
  ,unsigned* count,UserGParameterOutput::InletParticle* out){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
  if(!CODE_IsNormal(code[p]) || !CODE_IsFluid(code[p]) || idp[p]<firstId)return;
  UserGParameterOutput::InletParticle a;
  a.Idp=idp[p];a.Pos={xy[p].x,xy[p].y,z[p]};
  const float4 v=vr[p];a.VelRho={v.x,v.y,v.z,v.w};
  out[atomicAdd(count,1u)]=a;
}
}
void UserGParameterOutput::DownloadFlex(unsigned n,unsigned np,unsigned mode,UserGFlexPrecision* context
  ,const unsigned* rid,const unsigned* idp,const typecode* code
  ,const unsigned* pairs,const tmatrix3f* f,bool tensors,std::vector<FlexParticle>& out){
  out.resize(n);if(!n)return;
  OutputBuffer<FlexParticle> buffer(n);
  const unsigned blocks=(n+255)/256;
  if(mode==0 || !tensors)
    GatherFlex<float,false><<<blocks,256>>>(n,np,rid,idp,code,pairs,f,nullptr,tensors,buffer.Ptr);
  else if(mode==1){
    if(!context)throw std::runtime_error("Missing GPU precision context for output.");
    const auto s=*static_cast<UserGFlexState<float>*>(context->State(mode,n));
    GatherFlex<float,true><<<blocks,256>>>(n,np,rid,idp,code,pairs,f,s.H,tensors,buffer.Ptr);
  }
  else if(mode==2){
    if(!context)throw std::runtime_error("Missing GPU precision context for output.");
    const auto s=*static_cast<UserGFlexState<double>*>(context->State(mode,n));
    GatherFlex<double,true><<<blocks,256>>>(n,np,rid,idp,code,pairs,f,s.H,tensors,buffer.Ptr);
  }
  else throw std::runtime_error("Invalid GPU precision mode for output.");
  OutputCuda::Check(cudaGetLastError());
  OutputCuda::Check(cudaMemcpy(out.data(),buffer.Ptr,sizeof(FlexParticle)*n,cudaMemcpyDeviceToHost));
}
void UserGParameterOutput::DownloadInlet(unsigned n,unsigned firstId,const unsigned* idp
  ,const typecode* code,const double2* xy,const double* z,const float4* vr
  ,std::vector<InletParticle>& out){
  out.clear();if(!n)return;
  OutputBuffer<InletParticle> buffer(n);OutputBuffer<unsigned> count(1);
  OutputCuda::Check(cudaMemset(count.Ptr,0,sizeof(unsigned)));
  GatherInlet<<<(n+255)/256,256>>>(n,firstId,idp,code,xy,z,vr,count.Ptr,buffer.Ptr);
  OutputCuda::Check(cudaGetLastError());unsigned size=0;
  OutputCuda::Check(cudaMemcpy(&size,count.Ptr,sizeof(size),cudaMemcpyDeviceToHost));
  if(size>n)throw std::runtime_error("Invalid GPU inlet output count.");
  out.resize(size);
  if(size)OutputCuda::Check(cudaMemcpy(out.data(),buffer.Ptr,sizeof(InletParticle)*size,cudaMemcpyDeviceToHost));
}
