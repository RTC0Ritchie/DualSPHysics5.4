#ifndef USER_ICE_BOTTOM_PROTECTION_H
#define USER_ICE_BOTTOM_PROTECTION_H

#include "UserIceTopProtection.h"
#include "TypesDef.h"
#include <map>

namespace IceBottomNoPen {

struct State{
  bool Enabled=false;
  double WetTime=0.25,HCoef=4,LastTime=-1;
  std::vector<unsigned> Ids;
  std::vector<double> Support,Age;
  std::vector<unsigned char> Active;
  std::vector<IceTopNoPen::Point> Old;
};

// One state per UserParameters instance.
// Access this registry outside OpenMP particle loops.
inline State& Get(const void* owner){
  static std::map<const void*,State> states;
  return states[owner];
}

inline void Reset(const void* owner){
  Get(owner)=State();
}

inline void Init(State& d,std::vector<IceTopCleanup::Point> ref,
    double dp,unsigned nbound){
  d.Ids.clear();
  d.Old.clear();
  d.Active.clear();
  d.LastTime=-1;

  for(auto& q:ref)q.z=-q.z;

  const auto bottom=IceTopCleanup::ReferenceTop(ref,dp);
  for(const auto& q:bottom)d.Ids.push_back(q.id);

  d.Age.assign(d.Ids.size(),0.0);
  d.Support.assign(nbound,0.0);
}

inline std::vector<IceTopNoPen::Point> Profile(const State& d,
    unsigned nfixed,unsigned nbound,unsigned npb,const unsigned* ridp,
    const unsigned* ids,const tdouble3* pos,const tfloat4* vel,double dp){
  std::vector<IceTopNoPen::Point> b;
  b.reserve(d.Ids.size());

  for(unsigned id:d.Ids){
    if(!ridp || id<nfixed || id>=nbound)
      throw std::runtime_error(
          "IceBottomNoPenetration: invalid material ID.");

    const unsigned i=ridp[id-nfixed];

    if(i>=npb || ids[i]!=id)
      throw std::runtime_error(
          "IceBottomNoPenetration: particle lookup failed.");

    const auto q=pos[i];
    const auto v=vel[i];

    if(!std::isfinite(q.x) || !std::isfinite(q.z)
        || (!b.empty() && q.x-b.back().x<=1e-6*dp))
      throw std::runtime_error(
          "IceBottomNoPenetration: bottom folded.");

    b.push_back({q.x,-q.z,v.x,-v.z});
  }

  return b;
}

inline void UpdateActive(const std::vector<IceTopNoPen::Point>& bottom,
    const std::vector<double>& support,double dt,double wettime,
    double reach,double endmargin,std::vector<double>& wetage,
    std::vector<unsigned char>& active){
  const size_t n=bottom.size();

  if(support.size()!=n || wetage.size()!=n)
    throw std::runtime_error(
        "IceBottomNoPenetration: inconsistent history size.");

  std::vector<double> arc(n,0),seeds;

  for(size_t j=0;j<n;j++){
    if(j)
      arc[j]=arc[j-1]+std::hypot(
          bottom[j].x-bottom[j-1].x,
          bottom[j].z-bottom[j-1].z);

    wetage[j]=support[j]>=0.6
        ?std::min(wettime,wetage[j]+dt):0;

    if(support[j]<0.3)seeds.push_back(arc[j]);
  }

  const auto previous=active;
  active.assign(n,0);

  for(size_t j=0;j<n;j++){
    auto it=std::lower_bound(seeds.begin(),seeds.end(),arc[j]);

    double dist=std::numeric_limits<double>::max();

    if(it!=seeds.end())dist=std::abs(*it-arc[j]);

    if(it!=seeds.begin())
      dist=std::min(dist,std::abs(*(it-1)-arc[j]));

    const bool wasactive=j<previous.size() && previous[j];

    active[j]=((dist<=reach || wasactive)
        && wetage[j]<wettime
        && arc[j]>endmargin
        && arc.back()-arc[j]>endmargin);
  }
}

inline IceTopNoPen::Result CorrectBottom(
    const std::vector<IceTopNoPen::Point>& oldtop,
    const std::vector<IceTopNoPen::Point>& top,
    const IceTopNoPen::Point& oldp,
    const IceTopNoPen::Point& p,double dp,
    const std::vector<unsigned char>& active){
  IceTopNoPen::Result best;

  if(top.size()<2 || oldtop.size()!=top.size()
      || active.size()!=top.size())return best;

  const double r=0.5*dp,tol=1e-6*dp;

  auto it=std::upper_bound(
      top.begin(),top.end(),p.x,
      [](double x,const IceTopNoPen::Point& q){return x<q.x;});

  const size_t center=
      it==top.begin()?0:
      it==top.end()?top.size()-2:
      size_t(it-top.begin())-1;

  const size_t first=center?center-1:0;
  const size_t last=std::min(top.size()-2,center+1);

  double bestdepth=0;

  for(size_t i=first;i<=last;i++){
    if(!active[i] || !active[i+1])continue;

    const auto& a=top[i];
    const auto& b=top[i+1];

    const double tx=b.x-a.x,tz=b.z-a.z;
    const double len2=tx*tx+tz*tz;
    if(!(len2>0))continue;

    const double t=((p.x-a.x)*tx+(p.z-a.z)*tz)/len2;
    if(t<0 || t>1)continue;

    const double len=std::sqrt(len2);
    const double nx=-tz/len,nz=tx/len;
    const double d=(p.x-a.x)*nx+(p.z-a.z)*nz;
    if(d>=r)continue;

    const auto& oa=oldtop[i];
    const auto& ob=oldtop[i+1];

    const double otx=ob.x-oa.x,otz=ob.z-oa.z;
    const double olen=std::sqrt(otx*otx+otz*otz);
    if(!(olen>0))continue;

    const double oldd=
        ((oldp.x-oa.x)*(-otz)+(oldp.z-oa.z)*otx)/olen;

    // Do not pull already internal water back through the ice.
    if(oldd<-tol)continue;

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

} // namespace IceBottomNoPen

#endif