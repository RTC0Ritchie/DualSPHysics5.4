// CPU counterparts: JSphCpuSingle ice protection and UserIce*Protection.h.
// All per-stage state, profiles and fluid correction remain on the GPU.
#include "UserGIceProtection.h"
#include "UserIceTopProtection.h"
#include "UserIceBottomProtection.h"
#include "UserGMdbc.h"
#include "JSphGpuSingle.h"
#include "UserParameters.h"
#include "JPartsLoad4.h"
#include "JLog2.h"
#include <cuda_runtime.h>
#include <unordered_set>
#include <stdexcept>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <fstream>
#include <iomanip>
#include "Functions.h"
#include "JSphInOut.h"
#include "UserIceInOutData.h"
class UserGIceMath{
public:
  struct Point{double x,z,vx,vz;};
  struct Result{bool hit=false;double dx=0,dz=0,dvx=0,dvz=0;};
  struct View{
    const Point* p;unsigned n;bool rotated=false;
    __host__ __device__ unsigned size()const{return n;}
    __host__ __device__ Point operator[](unsigned i)const{
      const Point a=p[i];return rotated?Point{-a.z,a.x,-a.vz,a.vx}:a;
    }
    __host__ __device__ Point back()const{return (*this)[n-1];}
  };
  static __host__ __device__ size_t Min(size_t a,size_t b){return a<b?a:b;}
static __host__ __device__ Result Correct(
    View oldtop,
    View top,
    const Point& oldp,const Point& p,double dp,const unsigned char* active=nullptr)
{
  Result best;
  if(top.size()<2 || oldtop.size()!=top.size())return best;

  const double r=0.5*dp,tol=1e-6*dp;

  unsigned lo=0,hi=top.size();
  while(lo<hi){const unsigned mid=lo+(hi-lo)/2;if(p.x<top[mid].x)hi=mid;else lo=mid+1;}
  const size_t center=lo==0?0:lo==top.size()?top.size()-2:lo-1;

  const size_t first=center?center-1:0;
  const size_t last=Min(size_t(top.size()-2),center+1);

  double bestdepth=0;

  for(size_t i=first;i<=last;i++){
    if(active && (!active[i] || !active[i+1]))continue;
    const Point& a=top[i];
    const Point& b=top[i+1];

    const double tx=b.x-a.x,tz=b.z-a.z;
    const double len2=tx*tx+tz*tz;
    if(!(len2>0))continue;

    const double t=((p.x-a.x)*tx+(p.z-a.z)*tz)/len2;

    // Release beyond a material top endpoint.
    if(t<0 || t>1)continue;

    const double len=sqrt(len2);
    const double nx=-tz/len,nz=tx/len;
    const double d=(p.x-a.x)*nx+(p.z-a.z)*nz;
    if(d>=r)continue;

    const Point& oa=oldtop[i];
    const Point& ob=oldtop[i+1];

    const double otx=ob.x-oa.x,otz=ob.z-oa.z;
    const double olen=sqrt(otx*otx+otz*otz);
    if(!(olen>0))continue;

    const double oldd=
        ((oldp.x-oa.x)*(-otz)+(oldp.z-oa.z)*otx)/olen;

    // Do not grab side/underside or already internal fluid.
    if(oldd<(active?-tol:r-tol))continue;

    const double depth=r-d;
    if(best.hit && depth>=bestdepth)continue;

    const double wx=(1-t)*a.vx+t*b.vx;
    const double wz=(1-t)*a.vz+t*b.vz;
    const double vn=(p.vx-wx)*nx+(p.vz-wz)*nz;

    best.hit=true;
    bestdepth=depth;
    best.dx=(depth+tol)*nx;
    best.dz=(depth+tol)*nz;
    best.dvx=vn<0?-vn*nx:0;
    best.dvz=vn<0?-vn*nz:0;
  }

  return best;
}
static __host__ __device__ Result Corner(
    View oldtop,
    View top,
    View oldside,
    View side,
    const Point& oldp,const Point& p,double dp){
  Result c;
  if(top.size()<2 || side.size()<2 || oldtop.size()!=top.size()
      || oldside.size()!=side.size())return c;
  const auto& a=top.back();const auto& oa=oldtop.back();
  const auto& b=top[top.size()-2];const auto& d=side[1];
  const double tx=a.x-b.x,tz=a.z-b.z,sx=d.x-a.x,sz=d.z-a.z;
  const double ex=p.x-a.x,ez=p.z-a.z;
  if(tx*sz-tz*sx>=0 || ex*tx+ez*tz<0 || ex*sx+ez*sz>0)return c;
  const double r=.5*dp,tol=1e-6*dp,dist=hypot(ex,ez);
  if(dist>=r || dist<=tol)return c;
  const double ox=oldp.x-oa.x,oz=oldp.z-oa.z;
  const auto& ob=oldtop[oldtop.size()-2];const auto& od=oldside[1];
  if(ox*(oa.x-ob.x)+oz*(oa.z-ob.z)<-tol*hypot(oa.x-ob.x,oa.z-ob.z)
      || ox*(od.x-oa.x)+oz*(od.z-oa.z)>tol*hypot(od.x-oa.x,od.z-oa.z)
      || hypot(ox,oz)<r-tol)return c;
  const double nx=ex/dist,nz=ez/dist;
  const double vn=(p.vx-a.vx)*nx+(p.vz-a.vz)*nz;
  c.hit=true;c.dx=(r-dist+tol)*nx;c.dz=(r-dist+tol)*nz;
  if(vn<0){c.dvx=-vn*nx;c.dvz=-vn*nz;}
  return c;
}
static __host__ __device__ Result Combine(Result a,
    const Result& b){
  if(b.hit && (!a.hit || b.dx*b.dx+b.dz*b.dz<a.dx*a.dx+a.dz*a.dz))return b;
  return a;
}
};
class UserGIceDevice{
public:
  using Point=UserGIceMath::Point;using View=UserGIceMath::View;
  unsigned Nbound,Ntop,Nright,Nbottom;bool Top,Right,Bottom;
  double Dp,H,WetTime,HCoef;
  unsigned *Map,*TopIds,*RightIds,*BottomIds,*SeedCount;
  Point *TopNow,*TopOld,*RightNow,*RightOld,*BottomNow,*BottomOld;
  double *Support,*Age,*Arc,*Seeds;
  unsigned char* Active;
  double2* OldFluid;
  unsigned Noutline,*OutlineIds,*Invalid;
  double2* Outline;
  double4* OutlineBox;
};
__global__ void UserGIceMap(unsigned n,const unsigned* id,const typecode* code,UserGIceDevice d){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<n && CODE_IsNormal(code[p]) && CODE_IsFlexStrucFlex(code[p]) && id[p]<d.Nbound)d.Map[id[p]]=p;
}
__global__ void UserGIceProfile(unsigned n,const unsigned* identity,unsigned nb,
    const unsigned* id,const typecode* code,const double2* xy,const double* z,const float4* vel,
    UserGIceMath::Point* out,UserGIceDevice d,bool mirror,bool right){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const unsigned p=d.Map[identity[i]];
  if(p>=nb || id[p]!=identity[i] || !CODE_IsFlexStrucFlex(code[p])){asm("trap;");return;}
  const auto q=xy[p];const double rz=z[p];const auto v=vel[p];
  if(!isfinite(q.x)||!isfinite(rz)){asm("trap;");return;}
  if(i){const unsigned previous=d.Map[identity[i-1]];
    if(previous>=nb || (right?z[previous]-rz:q.x-xy[previous].x)<=1e-6*d.Dp){asm("trap;");return;}
  }
  out[i]={q.x,mirror?-rz:rz,v.x,mirror?-double(v.z):double(v.z)};
}
__global__ void UserGIceOldFluid(unsigned np,unsigned nb,const double2* xy,const double* z,double2* out){
  const unsigned p=nb+blockIdx.x*blockDim.x+threadIdx.x;if(p<np)out[p-nb]=make_double2(xy[p].x,z[p]);
}
// Preserve CPU arclength summation order. This is O(surface nodes), not O(fluid).
__global__ void UserGIceWetHistory(UserGIceDevice d,double dt){
  if(blockIdx.x || threadIdx.x)return;
  unsigned count=0;double arc=0;
  for(unsigned j=0;j<d.Nbottom;j++){
    if(j)arc+=hypot(d.BottomOld[j].x-d.BottomOld[j-1].x,d.BottomOld[j].z-d.BottomOld[j-1].z);
    d.Arc[j]=arc;const double support=d.Support[d.BottomIds[j]];
    d.Age[j]=support>=.6?fmin(d.WetTime,d.Age[j]+dt):0;
    if(support<.3)d.Seeds[count++]=arc;
  }
  *d.SeedCount=count;
}
__global__ void UserGIceActivate(UserGIceDevice d){
  const unsigned j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=d.Nbottom)return;
  const unsigned count=*d.SeedCount;unsigned lo=0,hi=count;
  while(lo<hi){const unsigned m=lo+(hi-lo)/2;if(d.Seeds[m]<d.Arc[j])lo=m+1;else hi=m;}
  double dist=1.7976931348623157e308;
  if(lo<count)dist=fabs(d.Seeds[lo]-d.Arc[j]);
  if(lo)dist=fmin(dist,fabs(d.Seeds[lo-1]-d.Arc[j]));
  d.Active[j]=((dist<=d.HCoef*d.H || d.Active[j]) && d.Age[j]<d.WetTime
      && d.Arc[j]>2*d.H && d.Arc[d.Nbottom-1]-d.Arc[j]>2*d.H);
}
// Equivalent to KerUpdatePos<false>; the protection disallows periodic domains.
__device__ void UserGIceMove(unsigned p,double dx,double dz,double2* xy,double* z,
    typecode* code,unsigned* cells,UserGIceProtection::Domain dom){
  const bool outmov=fmaxf(fabsf(float(dx)),fabsf(float(dz)))>dom.MoveLimit;
  xy[p].x+=dx;z[p]+=dz;
  const double x=xy[p].x-dom.Min.x,y=xy[p].y-dom.Min.y,rz=z[p]-dom.Min.z;
  const bool out=x!=x||y!=y||rz!=rz||x<0||y<0||rz<0||x>=dom.Size.x||y>=dom.Size.y||rz>=dom.Size.z;
  if(out || outmov){code[p]=out?CODE_SetOutPos(code[p]):CODE_SetOutMov(code[p]);cells[p]=DCEL_CodeMapOut;}
  else cells[p]=DCEL_Cell(dom.CellCode,unsigned(x/dom.Cell),unsigned(y/dom.Cell),unsigned(rz/dom.Cell));
}
__global__ void UserGIceCorrect(unsigned np,unsigned nb,UserGIceDevice d,
    double2* xy,double* z,float4* vel,typecode* code,unsigned* cells,UserGIceProtection::Domain dom){
  const unsigned p=nb+blockIdx.x*blockDim.x+threadIdx.x;if(p>=np)return;
  if(!CODE_IsNormal(code[p]) || !CODE_IsFluid(code[p]))return;
  using Math=UserGIceMath;using Point=Math::Point;
  const auto old=d.OldFluid[p-nb];
  if(d.Bottom){
    const auto c=Math::Correct({d.BottomOld,d.Nbottom},{d.BottomNow,d.Nbottom},
        {old.x,-old.y,0,0},{xy[p].x,-z[p],vel[p].x,-double(vel[p].z)},d.Dp,d.Active);
    if(c.hit){
      if(CODE_IsFluidInout(code[p])){asm("trap;");return;}
      UserGIceMove(p,c.dx,-c.dz,xy,z,code,cells,dom);vel[p].x+=float(c.dvx);vel[p].z-=float(c.dvz);
    }
  }
  if(!d.Top || !CODE_IsNormal(code[p]))return;
  const Point op={old.x,old.y,0,0},q={xy[p].x,z[p],vel[p].x,vel[p].z};
  auto c=Math::Correct({d.TopOld,d.Ntop},{d.TopNow,d.Ntop},op,q,d.Dp);
  if(d.Right){
    auto cr=Math::Correct({d.RightOld,d.Nright,true},{d.RightNow,d.Nright,true},
        {-op.z,op.x,-op.vz,op.vx},{-q.z,q.x,-q.vz,q.vx},d.Dp);
    if(cr.hit){const double dx=cr.dx,dvx=cr.dvx;cr.dx=cr.dz;cr.dz=-dx;cr.dvx=cr.dvz;cr.dvz=-dvx;}
    c=Math::Combine(c,cr);
    c=Math::Combine(c,Math::Corner({d.TopOld,d.Ntop},{d.TopNow,d.Ntop},
        {d.RightOld,d.Nright},{d.RightNow,d.Nright},op,q,d.Dp));
  }
  if(c.hit){
    if(CODE_IsFluidInout(code[p])){asm("trap;");return;}
    UserGIceMove(p,c.dx,c.dz,xy,z,code,cells,dom);vel[p].x+=float(c.dvx);vel[p].z+=float(c.dvz);
  }
}

// Same material outline, strict interior and half-Dp skin as UserIceInOutData.
// Bounds are built once per check; distant inlet buffers skip the edge scan.
__global__ void UserGIceOutline(unsigned nb,const unsigned* ids,const typecode* code,
    const double2* xy,const double* z,UserGIceDevice d){
  if(blockIdx.x || threadIdx.x)return;
  *d.Invalid=0;
  double xmin=DBL_MAX,xmax=-DBL_MAX,zmin=DBL_MAX,zmax=-DBL_MAX;
  for(unsigned i=0;i<d.Noutline;i++){
    const unsigned id=d.OutlineIds[i],p=d.Map[id];
    if(p>=nb || ids[p]!=id || !CODE_IsNormal(code[p]) || !CODE_IsFlexStrucFlex(code[p])){*d.Invalid=2;return;}
    const double x=xy[p].x,rz=z[p];
    if(!isfinite(x)||!isfinite(rz)){*d.Invalid=2;return;}
    d.Outline[i]=make_double2(x,rz);
    xmin=fmin(xmin,x);xmax=fmax(xmax,x);zmin=fmin(zmin,rz);zmax=fmax(zmax,rz);
  }
  *d.OutlineBox=make_double4(xmin,xmax,zmin,zmax);
}
__global__ void UserGIceCheckBuffers(unsigned np,unsigned nb,const typecode* code,
    const double2* xy,const double* z,UserGIceDevice d){
  const unsigned p=nb+blockIdx.x*blockDim.x+threadIdx.x;
  if(p>=np || !CODE_IsNormal(code[p]) || !CODE_IsFluidInout(code[p]))return;
  // Outline errors leave an incomplete polygon; check the validity flag first.
  if(atomicAdd(d.Invalid,0u)==2)return;
  const auto box=*d.OutlineBox;const double x=xy[p].x,rz=z[p],r=.5*d.Dp;
  if(x<box.x-r||x>box.y+r||rz<box.z-r||rz>box.w+r)return;
  bool inside=false;double nearest=DBL_MAX;
  for(unsigned i=0,j=d.Noutline-1;i<d.Noutline;j=i++){
    const auto a=d.Outline[j],b=d.Outline[i];
    const double dx=b.x-a.x,dz=b.y-a.y,l2=dx*dx+dz*dz;
    if(!(l2>0)){atomicMax(d.Invalid,2u);return;}
    const double u=fmax(0.,fmin(1.,((x-a.x)*dx+(rz-a.y)*dz)/l2));
    const double ex=x-a.x-u*dx,ez=rz-a.y-u*dz;
    nearest=fmin(nearest,ex*ex+ez*ez);
    if((a.y>rz)!=(b.y>rz)){
      const double crossx=a.x+(rz-a.y)*(b.x-a.x)/(b.y-a.y);
      if(x<crossx)inside=!inside;
    }
  }
  const double tol=1e-4*d.Dp;
  if(nearest<r*r || (inside && nearest>tol*tol))atomicMax(d.Invalid,1u);
}
class UserGIceProtection::Impl{
public:
  UserGIceDevice D={};std::vector<void*> Owned;size_t Bytes=0;unsigned Capacity=0,StageNp=0,StageNb=0;double LastTime=-1;
  static void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
  template<class T>T* Allocate(size_t n){T* p=nullptr;if(n){Check(cudaMalloc((void**)&p,n*sizeof(T)));Owned.push_back(p);Bytes+=n*sizeof(T);Check(cudaMemset(p,0,n*sizeof(T)));}return p;}
  unsigned* Ids(const std::vector<unsigned>& ids){auto p=Allocate<unsigned>(ids.size());if(p)Check(cudaMemcpy(p,ids.data(),ids.size()*sizeof(unsigned),cudaMemcpyHostToDevice));return p;}
  ~Impl(){for(void* p:Owned)cudaFree(p);cudaFree(D.OldFluid);}
  void Profiles(unsigned nb,const unsigned* id,const typecode* code,const double2* xy,const double* z,const float4* vel,bool old){
    if(nb)UserGIceMap<<<(nb+127)/128,128>>>(nb,id,code,D);
    if(D.Top)UserGIceProfile<<<(D.Ntop+127)/128,128>>>(D.Ntop,D.TopIds,nb,id,code,xy,z,vel,old?D.TopOld:D.TopNow,D,false,false);
    if(D.Right)UserGIceProfile<<<(D.Nright+127)/128,128>>>(D.Nright,D.RightIds,nb,id,code,xy,z,vel,old?D.RightOld:D.RightNow,D,false,true);
    if(D.Bottom)UserGIceProfile<<<(D.Nbottom+127)/128,128>>>(D.Nbottom,D.BottomIds,nb,id,code,xy,z,vel,old?D.BottomOld:D.BottomNow,D,true,false);
  }
};
UserGIceProtection::UserGIceProtection(const std::vector<Reference>& ref,bool top,bool right,bool bottom,
    double dp,double h,double wettime,double hcoef,unsigned nbound,double* support,bool inout):Data(nullptr){
  std::unique_ptr<Impl> next(new Impl);auto& a=*next;auto& d=a.D;
  d.Top=top;d.Right=right;d.Bottom=bottom;d.Dp=dp;d.H=h;d.WetTime=wettime;d.HCoef=hcoef;d.Nbound=nbound;d.Support=support;
  std::vector<IceTopCleanup::Point> source;for(const auto& q:ref)source.push_back({q.x,q.z,q.id});
  const auto points=IceTopCleanup::ReferenceTop(source,dp);std::vector<unsigned> ids;
  for(const auto& q:points)ids.push_back(q.id);
  d.Ntop=ids.size();d.TopIds=a.Ids(ids);d.TopNow=a.Allocate<UserGIceDevice::Point>(d.Ntop);d.TopOld=a.Allocate<UserGIceDevice::Point>(d.Ntop);
  if(d.Right){
    IceRightNoPen::State side;side.Enabled=true;IceRightNoPen::Init(side,source,dp);
    if(side.Ids.front()!=ids.back())throw std::runtime_error("Ice right side does not join material top.");
    d.Nright=side.Ids.size();d.RightIds=a.Ids(side.Ids);
    d.RightNow=a.Allocate<UserGIceDevice::Point>(d.Nright);d.RightOld=a.Allocate<UserGIceDevice::Point>(d.Nright);
  }
  if(bottom){
    if(!support)throw std::runtime_error("GPU ice bottom: mDBC support is unavailable.");
    IceBottomNoPen::State state;IceBottomNoPen::Init(state,source,dp,nbound);
    d.Nbottom=state.Ids.size();d.BottomIds=a.Ids(state.Ids);
    d.BottomNow=a.Allocate<UserGIceDevice::Point>(d.Nbottom);d.BottomOld=a.Allocate<UserGIceDevice::Point>(d.Nbottom);
    d.Age=a.Allocate<double>(d.Nbottom);d.Arc=a.Allocate<double>(d.Nbottom);d.Seeds=a.Allocate<double>(d.Nbottom);
    d.Active=a.Allocate<unsigned char>(d.Nbottom);d.SeedCount=a.Allocate<unsigned>(1);
  }
  d.Map=a.Allocate<unsigned>(nbound);Impl::Check(cudaMemset(d.Map,0xff,size_t(nbound)*sizeof(unsigned)));
  if(inout){
    const auto outline=IceTopCleanup::ReferenceOutline(source,dp);
    std::vector<unsigned> outlineids;for(const auto& q:outline)outlineids.push_back(q.id);
    d.Noutline=outlineids.size();d.OutlineIds=a.Ids(outlineids);
    d.Outline=a.Allocate<double2>(d.Noutline);d.OutlineBox=a.Allocate<double4>(1);
    d.Invalid=a.Allocate<unsigned>(1);
  }
  Data=next.release();
}
UserGIceProtection::~UserGIceProtection(){delete Data;}
size_t UserGIceProtection::GetAllocMemoryGpu()const{return Data->Bytes+size_t(Data->Capacity)*sizeof(double2);}
void UserGIceProtection::ValidateBuffers(unsigned np,unsigned nb,const unsigned* ids,
    const typecode* code,const double2* xy,const double* z){
  auto& d=Data->D;if(!d.Noutline)return;
  if(nb)UserGIceMap<<<(nb+127)/128,128>>>(nb,ids,code,d);
  UserGIceOutline<<<1,1>>>(nb,ids,code,xy,z,d);
  if(np>nb)UserGIceCheckBuffers<<<(np-nb+127)/128,128>>>(np,nb,code,xy,z,d);
  unsigned invalid=0;Impl::Check(cudaMemcpy(&invalid,d.Invalid,sizeof(invalid),cudaMemcpyDeviceToHost));
  if(invalid==2)throw std::runtime_error("Ice/InOut compatibility: missing/nonfinite outline material particle or collapsed edge.");
  if(invalid)throw std::runtime_error("Ice/InOut compatibility: inlet/outlet buffer overlaps ice (including the half-Dp surface skin); move the inlet away from ice.");
}
void UserGIceProtection::Capture(bool predictor,double time,unsigned np,unsigned nb,const unsigned* ids,
    const typecode* code,const double2* xy,const double* z,const float4* vel,const double2* oldxy,const double* oldz){
  auto& a=*Data;auto& d=a.D;
  if(!d.Top && !d.Right && !d.Bottom)return; // Cleanup-only InOut guard.
  const unsigned count=np-nb;
  if(count>a.Capacity){
    const unsigned capacity=std::max(count,a.Capacity+std::max(128u,a.Capacity/5));
    double2* next=nullptr;Impl::Check(cudaMalloc((void**)&next,size_t(capacity)*sizeof(double2)));
    cudaFree(d.OldFluid);d.OldFluid=next;a.Capacity=capacity;
  }
  a.StageNp=np;a.StageNb=nb;
  if(count)UserGIceOldFluid<<<(count+127)/128,128>>>(np,nb,oldxy,oldz,d.OldFluid);
  if(predictor){
    a.Profiles(nb,ids,code,xy,z,vel,true);
    if(d.Bottom){
      const double dt=a.LastTime<0?0:std::max(0.,time-a.LastTime);a.LastTime=time;
      UserGIceWetHistory<<<1,1>>>(d,dt);UserGIceActivate<<<(d.Nbottom+127)/128,128>>>(d);
    }
  }
  Impl::Check(cudaGetLastError());
}
void UserGIceProtection::Apply(unsigned np,unsigned nb,const unsigned* ids,typecode* code,
    double2* xy,double* z,float4* vel,unsigned* cells,const Domain& domain){
  auto& a=*Data;
  if(!a.D.Top && !a.D.Right && !a.D.Bottom){
    ValidateBuffers(np,nb,ids,code,xy,z);
    return;
  }
  if(np!=a.StageNp || nb!=a.StageNb)throw std::runtime_error("GPU ice protection: stage particle count changed.");
  ValidateBuffers(np,nb,ids,code,xy,z);
  a.Profiles(nb,ids,code,xy,z,vel,false);
  if(np>nb)UserGIceCorrect<<<(np-nb+127)/128,128>>>(np,nb,a.D,xy,z,vel,code,cells,domain);
  Impl::Check(cudaGetLastError());
}
void JSphGpuSingle::InitIceProtection(){
  delete IceProtection;IceProtection=nullptr;
  const auto& bottom=IceBottomNoPen::Get(User.get());
  if(!User->IceTopNoPenetrationEnabled && !bottom.Enabled
      && !(User->IceTopFluidCleanupEnabled && InOut))return;
  JPartsLoad4 initial(false);initial.LoadParticles(DirCase,CaseName,0,"");
  std::vector<unsigned> ids(Npb);std::vector<typecode> code(Npb);
  auto check=[](cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));};
  check(cudaMemcpy(ids.data(),Idp_g->cptr(),Npb*sizeof(unsigned),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(code.data(),Code_g->cptr(),Npb*sizeof(typecode),cudaMemcpyDeviceToHost));
  std::unordered_set<unsigned> elastic;
  for(unsigned p=0;p<Npb;p++)if(CODE_IsNormal(code[p]) && CODE_IsFlexStrucFlex(code[p]))elastic.insert(ids[p]);
  std::vector<UserGIceProtection::Reference> ref;
  for(unsigned i=0;i<initial.GetCount();i++)if(elastic.erase(initial.GetIdp()[i])){
    const auto q=initial.GetPos()[i];ref.push_back({q.x,q.z,initial.GetIdp()[i]});
  }
  if(!elastic.empty())Run_Exceptioon("GPU ice protection: initial case lacks elastic IDs.");
  if(InOut){
    std::vector<double2> xy(Npb);std::vector<double> z(Npb);
    check(cudaMemcpy(xy.data(),Posxy_g->cptr(),Npb*sizeof(double2),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(z.data(),Posz_g->cptr(),Npb*sizeof(double),cudaMemcpyDeviceToHost));
    std::vector<IceTopCleanup::Point> source;
    for(const auto& q:ref)source.push_back({q.x,q.z,q.id});
    auto outline=IceTopCleanup::ReferenceOutline(source,Dp);
    std::vector<unsigned> map(CaseNbound,UINT_MAX);
    for(unsigned p=0;p<Npb;p++)if(ids[p]<CaseNbound)map[ids[p]]=p;
    for(auto& q:outline){const unsigned p=map[q.id];
      if(p>=Npb)Run_Exceptioon("Ice/InOut compatibility: missing outline material particle.");
      q.x=xy[p].x;q.z=z[p];
    }
    IceTopCleanup::ValidateOutline(outline,Dp);
  }
  double* support=bottom.Enabled?MdbcUser->EnableBottomSupport(CaseNbound):nullptr;
  IceProtection=new UserGIceProtection(ref,User->IceTopNoPenetrationEnabled,
      IceRightNoPen::Get(User.get()).Enabled,bottom.Enabled,Dp,KernelH,bottom.WetTime,bottom.HCoef,CaseNbound,support,InOut!=nullptr);
  ValidateIceInOut();
  Log->Printf("GPU ice protection: top=%u right=%u bottom=%u; material surface IDs retained; wetting history reset as on CPU.",
      unsigned(User->IceTopNoPenetrationEnabled),unsigned(IceRightNoPen::Get(User.get()).Enabled),unsigned(bottom.Enabled));
}
void JSphGpuSingle::ValidateIceInOut(){
  if(InOut && IceProtection)IceProtection->ValidateBuffers(Np,Npb,Idp_g->cptr(),
      Code_g->cptr(),Posxy_g->cptr(),Posz_g->cptr());
}
void JSphGpuSingle::CaptureIceProtection(bool predictor){
  if(IceProtection)IceProtection->Capture(predictor,TimeStep,Np,Npb,Idp_g->cptr(),Code_g->cptr(),
      Posxy_g->cptr(),Posz_g->cptr(),Velrho_g->cptr(),
      predictor?Posxy_g->cptr():PosxyPre_g->cptr(),predictor?Posz_g->cptr():PoszPre_g->cptr());
}
void JSphGpuSingle::ApplyIceProtection(bool predictor){
  if(IceProtection)IceProtection->Apply(Np,Npb,Idp_g->cptr(),Code_g->ptr(),Posxy_g->ptr(),Posz_g->ptr(),
      Velrho_g->ptr(),Dcell_g->ptr(),{MapRealPosMin,MapRealSize,Scell,MovLimit,DomCellCode});
}

bool JSphGpuSingle::ClearIceTopFluidOnce(){
  if(!User->IceTopFluidCleanupEnabled)return false;

  if(!Simulate2D || PeriActive ||
     !FlexStruc || FlexStrucCount!=1)
  {
    Run_Exceptioon(
        "IceTopFluidCleanup requires 2-D, one elastic body, "
        "no periodic boundaries.");
  }

  // Read original geometry only to identify material top-surface IDs.
  JPartsLoad4 initial(false);
  initial.LoadParticles(DirCase,CaseName,0,"");

  // Initialization only: use the exact CPU geometry predicates and CSV audit.
  // Runtime ice/InOut checks remain on device; no per-step array download.
  std::vector<unsigned> hostIds(Np);
  std::vector<typecode> hostCode(Np);
  std::vector<double2> hostXY(Np);
  std::vector<double> hostZ(Np);
  std::vector<tdouble3> hostPos(Np);
  std::vector<tfloat4> hostVel(Np);
  const auto check=[](cudaError_t e){
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  };
  check(cudaMemcpy(hostIds.data(),Idp_g->cptr(),Np*sizeof(unsigned),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hostCode.data(),Code_g->cptr(),Np*sizeof(typecode),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hostXY.data(),Posxy_g->cptr(),Np*sizeof(double2),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hostZ.data(),Posz_g->cptr(),Np*sizeof(double),cudaMemcpyDeviceToHost));
  check(cudaMemcpy(hostVel.data(),Velrho_g->cptr(),Np*sizeof(tfloat4),cudaMemcpyDeviceToHost));
  for(unsigned p=0;p<Np;p++)hostPos[p]=TDouble3(hostXY[p].x,hostXY[p].y,hostZ[p]);
  const auto* ids=hostIds.data();
  const auto* codes=hostCode.data();
  const auto* pos=hostPos.data();

  std::unordered_map<unsigned,unsigned> current;

  for(unsigned p=0;p<Npb;p++){
    if(CODE_IsNormal(codes[p]) &&
       CODE_IsFlexStrucFlex(codes[p]))
    {
      if(!current.emplace(ids[p],p).second)
        Run_Exceptioon(
            "IceTopFluidCleanup: duplicated elastic ID.");
    }
  }

  std::vector<IceTopCleanup::Point> ref;
  ref.reserve(current.size());

  std::unordered_set<unsigned> found;

  for(unsigned i=0;i<initial.GetCount();i++){
    const unsigned id=initial.GetIdp()[i];

    if(current.find(id)==current.end())continue;

    if(!found.insert(id).second)
      Run_Exceptioon(
          "IceTopFluidCleanup: duplicated initial elastic ID.");

    const tdouble3 q=initial.GetPos()[i];
    ref.push_back({q.x,q.z,id});
  }

  if(found.size()!=current.size())
    Run_Exceptioon(
        "IceTopFluidCleanup: original case lacks current elastic IDs.");

  std::vector<IceTopCleanup::Point> top;
  std::vector<IceTopCleanup::Point> outline;

  try{
    top=IceTopCleanup::ReferenceTop(ref,Dp);

    // Keep material identity but use current, possibly deformed positions.
    for(auto& q:top){
      const auto& p=pos[current.at(q.id)];
      q.x=p.x;
      q.z=p.z;
    }

    IceTopCleanup::ValidateCurrent(top,Dp);
    outline=IceTopCleanup::ReferenceOutline(ref,Dp);
    for(auto& q:outline){
      const auto& p=pos[current.at(q.id)];
      q.x=p.x;
      q.z=p.z;
    }
    IceTopCleanup::ValidateOutline(outline,Dp);
  }
  catch(const std::exception& e){
    Run_Exceptioon(
        std::string("IceTopFluidCleanup: ")+e.what());
  }

  if(InOut){
    UserIceInOutData::ConfigureOutline(User.get(),outline);
    std::vector<unsigned> rid(CaseNbound-CaseNfixed,UINT_MAX);
    for(unsigned p=0;p<Npb;p++)
      if(ids[p]>=CaseNfixed && ids[p]<CaseNbound)
        rid[ids[p]-CaseNfixed]=p;
    UserIceInOutData::ValidateBuffers(User.get(),CaseNfixed,Npb,Np,
        rid.data(),ids,codes,pos,Dp);
    for(unsigned p=Npb;p<Np;p++)
      if(CODE_IsNormal(codes[p]) && CODE_IsFluidInout(codes[p])
          && (IceTopCleanup::Above(top,pos[p].x,pos[p].z,Dp)
            || IceTopCleanup::InsideIce(outline,pos[p].x,pos[p].z,Dp)))
        Run_Exceptioon("IceTopFluidCleanup: inlet buffer overlaps the cleanup target; move the inlet instead of deleting buffer particles.");
  }

  std::vector<unsigned> remove;
  std::vector<unsigned> reasons;
  unsigned nabove=0,ninside=0;
  unsigned nfluid=0;
  tdouble3 momentum=TDouble3(0);

  const auto* vr=hostVel.data();

  for(unsigned p=Npb;p<Np;p++){
    if(CODE_IsNormal(codes[p]) && CODE_IsFluid(codes[p])
        && !CODE_IsFluidInout(codes[p])){
      nfluid++;

      const bool above=IceTopCleanup::Above(top,pos[p].x,pos[p].z,Dp);
      const bool inside=!above &&
          IceTopCleanup::InsideIce(outline,pos[p].x,pos[p].z,Dp);
      if(above || inside){
        remove.push_back(p);
        reasons.push_back(above?1:2);
        if(above)nabove++;
        else ninside++;

        momentum.x+=double(MassFluid)*vr[p].x;
        momentum.y+=double(MassFluid)*vr[p].y;
        momentum.z+=double(MassFluid)*vr[p].z;
      }
    }
  }

  if(!remove.empty() && remove.size()==nfluid
      && !(InOut && InOut->GetCurrentNp()))
    Run_Exceptioon(
        "IceTopFluidCleanup would remove all fluid; "
        "check the case geometry.");

  // Save an audit record before changing any particle.
  const std::string file=
      fun::GetDirWithSlash(DirOut)+"IceTopFluidCleanup.csv";

  std::ofstream audit(file.c_str());

  if(!audit)
    Run_Exceptioon("Cannot open IceTopFluidCleanup.csv.");

  audit<<std::setprecision(17)
       <<"Idp;x;y;z;vx;vy;vz;Density_sph;Mass;Time;Reason\n";

  for(size_t i=0;i<remove.size();i++){
    const unsigned p=remove[i];
    audit<<ids[p]
         <<';'<<pos[p].x
         <<';'<<pos[p].y
         <<';'<<pos[p].z
         <<';'<<vr[p].x
         <<';'<<vr[p].y
         <<';'<<vr[p].z
         <<';'<<vr[p].w
         <<';'<<MassFluid
         <<';'<<TimeStep
         <<';'<<(reasons[i]==1?"above_top":"inside_ice")
         <<'\n';
  }

  audit.close();

  if(!audit)
    Run_Exceptioon("Cannot finish IceTopFluidCleanup.csv.");

  // Use the existing particle removal and array-sorting mechanism.
  // Do not renumber stable IDs.
  for(unsigned p:remove)
    hostCode[p]=CODE_SetOutIgnore(codes[p]);

  if(!remove.empty()){
    check(cudaMemcpy(Code_g->ptr(),hostCode.data(),Np*sizeof(typecode),cudaMemcpyHostToDevice));
    RunCellDivide(true);
  }

  // Intentional initialization removal must not consume the allowance
  // for subsequent accidental fluid loss.
  if(!remove.empty()){
    NpfMinimum=unsigned(
        MinFluidStop*(Np-NpbPer-NpfPer-CaseNbound));
  }

  Log->Printf(
      "IceTopFluidCleanup: time=%g; removed=%u/%u; mass=%g; "
      "momentum=(%g,%g,%g); top x=[%g,%g]; NpfMinimum=%u; "
      "above_top=%u; inside_ice=%u.",
      TimeStep,
      unsigned(remove.size()),
      nfluid,
      double(MassFluid)*remove.size(),
      momentum.x,momentum.y,momentum.z,
      top.front().x,top.back().x,
      NpfMinimum,nabove,ninside);
  return true;
}

bool UserGIceProtection::CleanupEnabled(const UserParameters* user){
  return user && user->IceTopFluidCleanupEnabled;
}
