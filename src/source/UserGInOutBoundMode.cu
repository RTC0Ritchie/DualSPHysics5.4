#include "UserGInOutBoundMode.h"
#include <cuda_runtime.h>
__global__ void UserGValveTopCrossing(unsigned np,unsigned nb,double face,double eps,
    double xmin,double xmax,const double2* oldxy,const double* oldz,
    double2* xy,double* z,float4* vel,typecode* code,unsigned* cells,
    UserGIceProtection::Domain dom){
  const unsigned p=nb+blockIdx.x*blockDim.x+threadIdx.x;
  if(p>=np || !CODE_IsNormal(code[p]) || !CODE_IsFluidNotInout(code[p]))return;
  const double d1=z[p]-face;if(d1>=0)return;
  const double d0=oldz[p]-face;if(d0<0)return;
  const double alpha=d0/(d0-d1);
  const double crossx=oldxy[p].x+alpha*(xy[p].x-oldxy[p].x);
  if(crossx<xmin || crossx>xmax)return;
  const double dz=face+eps-z[p];z[p]+=dz;
  const double x=xy[p].x-dom.Min.x,y=xy[p].y-dom.Min.y,rz=z[p]-dom.Min.z;
  const bool out=x!=x||y!=y||rz!=rz||x<0||y<0||rz<0||x>=dom.Size.x||y>=dom.Size.y||rz>=dom.Size.z;
  const bool outmov=fabsf(float(dz))>dom.MoveLimit;
  if(out||outmov){code[p]=out?CODE_SetOutPos(code[p]):CODE_SetOutMov(code[p]);cells[p]=DCEL_CodeMapOut;}
  else cells[p]=DCEL_Cell(dom.CellCode,unsigned(x/dom.Cell),unsigned(y/dom.Cell),unsigned(rz/dom.Cell));
  if(vel[p].z<0)vel[p].z=0;
}
void UserGInOutBoundMode::Launch(unsigned np,unsigned nb,double face,double eps,double xmin,double xmax,
    const double2* oldxy,const double* oldz,double2* xy,double* z,float4* vel,
    typecode* code,unsigned* cells,UserGIceProtection::Domain domain){
  if(np>nb)UserGValveTopCrossing<<<(np-nb+255)/256,256>>>(np,nb,face,eps,xmin,xmax,oldxy,oldz,xy,z,vel,code,cells,domain);
  const auto error=cudaGetLastError();
  if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void UserGValveConstraint(unsigned np,unsigned nb,double face,double width,double xmin,double xmax,
    double dt,float u,float gz,bool force,const double2* xy,const double* z,const typecode* code,
    float4* vel,const float4* pre,float3* ace){
  const unsigned p=nb+blockIdx.x*blockDim.x+threadIdx.x;
  if(p>=np || !CODE_IsNormal(code[p]) || !CODE_IsFluidNotInout(code[p]))return;
  const double d=z[p]-face;
  if(d<0 || d>=width || xy[p].x<xmin || xy[p].x>xmax)return;
  const float lower=u*float(1.-d/width);
  if(!force){vel[p].z=fmaxf(vel[p].z,lower);return;}
  const float amin=float((double(lower)-pre[p].z)/dt)-gz;
  ace[p].z=fmaxf(ace[p].z,amin);
}
void UserGInOutBoundMode::ConstraintLaunch(unsigned np,unsigned nb,double face,double width,double xmin,double xmax,
    double dt,float u,float gz,bool force,const double2* xy,const double* z,const typecode* code,
    float4* vel,const float4* pre,float3* ace){
  if(np>nb)UserGValveConstraint<<<(np-nb+255)/256,256>>>(np,nb,face,width,xmin,xmax,dt,u,gz,force,xy,z,code,vel,pre,ace);
  const auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
