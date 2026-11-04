// CPU counterparts: JSphCpuSingle::MdbcBoundCorrection and selective NoPenetration.
#include "UserGMdbc.h"
#include "UserParameters.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cassert>

__global__ void UserGMdbcNormals(unsigned np,unsigned nb,unsigned begin,unsigned end,
    float disabled,const unsigned* id,const typecode* code,const float3* src,float3* dst){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=np)return;
  float3 v=make_float3(0,0,disabled);
  if(p<nb && id[p]>=begin && id[p]<end && CODE_IsFixed(code[p]) && !CODE_IsFlexStrucAny(code[p])){
    const float3 b=src[p];if(b.x!=0 || b.y!=0 || b.z!=0)v=b;
  }
  dst[p]=v;
}
__global__ void UserGMdbcHistory(unsigned nb,int step,bool twoD,double gravity,
    int mode,double threshold,const unsigned* ids,const typecode* codes,
    const float3* normals,const float3* ace,UserGMdbc::View data){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p==0)*data.Step=step;
  if(p>=nb || !CODE_IsNormal(codes[p]) || !CODE_IsFlexStrucFlex(codes[p]))return;
  const float3 b=normals[p];if(b.x==0 && b.y==0 && b.z==0)return;
  const unsigned id=ids[p];if(id>=data.Count){asm("trap;");return;}
  auto& h=data.State[id];if(h.Step==step)return;
  const float3 a=ace[p];
  const double head=(double(b.x)*a.x+double(b.z)*a.z+(twoD?0.:double(b.y)*a.y))/gravity;
  if(h.Step<0 || step!=h.Step+1){
    h={};h.Step=step;h.Head=head;h.Filtered=a;return;
  }
  const double delta=head-h.Head,amp=fabs(delta),prev=fabs(h.Delta);
  const bool temporal=(mode&1)!=0;
  const bool trigger=temporal && delta*h.Delta<0 && fmin(amp,prev)>threshold && amp>=.8*prev;
  h.Filtered=make_float3(.5f*(a.x+h.Filtered.x),.5f*(a.y+h.Filtered.y),.5f*(a.z+h.Filtered.z));
  if(!temporal){h.Weight=0;h.Calm=0;}
  else if(trigger){h.Weight=1;h.Calm=0;}
  else if(amp<.25*threshold){if(h.Calm<8)h.Calm++;if(h.Calm>=8)h.Weight=fmaxf(0,h.Weight-.125f);}
  else h.Calm=0;
  h.Step=step;h.Head=head;h.Delta=delta;
}
__global__ void UserGMdbcApply(unsigned np,unsigned nb,int step,const unsigned* ids,
    const typecode* code,const float3* src,float3* dst,UserGMdbc::View data){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=np)return;
  float3 a=src[p];
  if(p<nb && CODE_IsFlexStrucFlex(code[p]) && ids[p]<data.Count){
    const auto h=data.State[ids[p]];
    if(h.Step==step && h.Weight>0){const float w=h.Weight;
      a=make_float3((1-w)*a.x+w*h.Filtered.x,(1-w)*a.y+w*h.Filtered.y,(1-w)*a.z+w*h.Filtered.z);
    }
  }
  dst[p]=a;
}
class UserGMdbc::Impl{
public:
  View V={};float3 *Normals=nullptr,*Ace=nullptr;unsigned NormalCapacity=0,AceCapacity=0;
  static void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
  ~Impl(){cudaFree(V.BottomSupport);cudaFree(V.State);cudaFree(V.Step);cudaFree(Normals);cudaFree(Ace);}
  static void Reserve(float3*& p,unsigned& capacity,unsigned n){
    if(n<=capacity)return;
    float3* next=nullptr;Check(cudaMalloc((void**)&next,n*sizeof(float3)));
    cudaFree(p);p=next;capacity=n;
  }
};
UserGMdbc::UserGMdbc():Data(new Impl){}
UserGMdbc::~UserGMdbc(){
  if(Data->V.State || Data->V.BottomSupport){try{Upload({});}catch(...){} }
  delete Data;
}
size_t UserGMdbc::GetAllocMemoryGpu()const{return Data->V.BottomCount*sizeof(double)+Data->V.Count*sizeof(History)
  +(Data->V.Step?sizeof(int):0)+(Data->NormalCapacity+Data->AceCapacity)*sizeof(float3);}
const float3* UserGMdbc::NoPenNormals(const UserParameters* user,unsigned np,unsigned nb,double dp,
    const unsigned* id,const typecode* code,const float3* normals){
  if(!user->NoPenetrationMkEnabled)return normals;
  if(!normals)throw std::runtime_error("Selective NoPenetration: missing normals.");
  Impl::Reserve(Data->Normals,Data->NormalCapacity,np);
  if(np)UserGMdbcNormals<<<(np+255)/256,256>>>(np,nb,user->NoPenetrationMkIdBegin,
    user->NoPenetrationMkIdEnd,float(4*dp),id,code,normals,Data->Normals);
  Impl::Check(cudaGetLastError());return Data->Normals;
}
const float3* UserGMdbc::Acceleration(const UserParameters* user,unsigned np,unsigned nb,unsigned count,
    unsigned nf,int step,bool twoD,tfloat3 gravity,const unsigned* id,const typecode* code,
    const float3* normals,const float3* acceleration){
  if(!user->Mdbc2AdaptiveEnabled || !nf)return acceleration;
  if(!Data->V.State){
    std::vector<History> initial(count);for(auto& h:initial)h.Step=-1;
    Impl::Check(cudaMalloc((void**)&Data->V.State,count*sizeof(History)));Data->V.Count=count;
    Impl::Check(cudaMemcpy(Data->V.State,initial.data(),count*sizeof(History),cudaMemcpyHostToDevice));
    Impl::Check(cudaMalloc((void**)&Data->V.Step,sizeof(int)));
    Impl::Check(cudaMemset(Data->V.Step,0xff,sizeof(int)));
    Data->V.Support=(user->Mdbc2AdaptiveMode&2)?float(user->Mdbc2AdaptiveSupport):0;
    Upload(Data->V);
  }
  if(count!=Data->V.Count)throw std::runtime_error("Mdbc2Adaptive: boundary identity count changed.");
  Impl::Reserve(Data->Ace,Data->AceCapacity,np);
  const double g=sqrt(double(gravity.x)*gravity.x+double(gravity.y)*gravity.y+double(gravity.z)*gravity.z);
  if(nb)UserGMdbcHistory<<<(nb+255)/256,256>>>(nb,step,twoD,g,user->Mdbc2AdaptiveMode,
    user->Mdbc2AdaptiveHead,id,code,normals,acceleration,Data->V);
  if(np)UserGMdbcApply<<<(np+255)/256,256>>>(np,nb,step,id,code,acceleration,Data->Ace,Data->V);
  Impl::Check(cudaGetLastError());return Data->Ace;
}

double* UserGMdbc::EnableBottomSupport(unsigned count){
  if(Data->V.BottomSupport && Data->V.BottomCount!=count)
    throw std::runtime_error("Ice bottom: boundary identity count changed.");
  if(!Data->V.BottomSupport){
    Impl::Check(cudaMalloc((void**)&Data->V.BottomSupport,size_t(count)*sizeof(double)));
    Data->V.BottomCount=count;ClearBottomSupport();Upload(Data->V);
  }
  return Data->V.BottomSupport;
}
void UserGMdbc::ClearBottomSupport(){
  if(Data->V.BottomSupport)Impl::Check(cudaMemsetAsync(Data->V.BottomSupport,0,size_t(Data->V.BottomCount)*sizeof(double)));
}
