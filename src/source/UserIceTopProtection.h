#ifndef USER_ICE_TOP_PROTECTION_H
#define USER_ICE_TOP_PROTECTION_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <limits>

namespace IceTopCleanup {

struct Point {
  double x,z;
  unsigned id;
};

// Initial geometry must be a column-sampled 2-D solid
// with one upper surface.
inline std::vector<Point> ReferenceTop(
    std::vector<Point> p,double dp)
{
  if(!(dp>0) || !std::isfinite(dp))
    throw std::runtime_error("Invalid Dp.");

  for(const auto& a:p){
    if(!std::isfinite(a.x) || !std::isfinite(a.z))
      throw std::runtime_error(
          "Non-finite initial ice coordinate.");
  }

  std::sort(
      p.begin(),p.end(),
      [](const Point& a,const Point& b){
        return a.x<b.x;
      });

  std::vector<Point> top;
  const double tol=dp*1e-4;

  for(size_t i=0;i<p.size();){
    size_t j=i+1;
    Point best=p[i];

    while(j<p.size() && p[j].x-p[i].x<=tol){
      if(p[j].z>best.z)best=p[j];
      j++;
    }

    if(!top.empty() && best.x-top.back().x>2*dp)
      throw std::runtime_error(
          "Initial ice columns are disconnected or too sparse.");

    top.push_back(best);
    i=j;
  }

  if(top.size()<2)
    throw std::runtime_error(
        "At least two ice columns are required.");

  return top;
}

inline void ValidateCurrent(
    const std::vector<Point>& top,double dp)
{
  for(size_t i=0;i<top.size();i++){
    if(!std::isfinite(top[i].x) ||
       !std::isfinite(top[i].z))
    {
      throw std::runtime_error(
          "Non-finite current ice-top coordinate.");
    }

    if(i && top[i].x-top[i-1].x<=dp*1e-6)
      throw std::runtime_error(
          "Current ice top folds or is not increasing in x.");
  }
}

inline bool Above(
    const std::vector<Point>& top,
    double x,double z,double dp)
{
  if(top.size()<2 ||
     x<top.front().x ||
     x>top.back().x)
    return false;

  auto it=std::upper_bound(
      top.begin(),top.end(),x,
      [](double x,const Point& p){
        return x<p.x;
      });

  const size_t i=(it==top.end())
      ? top.size()-2
      : size_t(it-top.begin())-1;

  const Point& a=top[i];
  const Point& b=top[i+1];

  const double slope=(b.z-a.z)/(b.x-a.x);

  // Cleanup starts at the top particle-centre polyline, including its
  // exterior half-Dp skin. The finite x range above excludes end extension.
  const double surface=a.z+slope*(x-a.x);

  return z>=surface;
}


// Material outline: initial upper and lower column extrema, joined at both ends.
// The current positions are supplied by the caller using stable particle IDs.
inline std::vector<Point> ReferenceOutline(const std::vector<Point>& ref,double dp){
  auto top=ReferenceTop(ref,dp);
  auto inverted=ref;
  for(auto& p:inverted)p.z=-p.z;
  auto bottom=ReferenceTop(inverted,dp);
  for(auto& p:bottom)p.z=-p.z;
  std::vector<Point> outline=top;
  for(auto it=bottom.rbegin();it!=bottom.rend();++it)
    if(outline.back().id!=it->id)outline.push_back(*it);
  if(outline.size()>1 && outline.front().id==outline.back().id)
    outline.pop_back();
  return outline;
}

inline double Cross(const Point& a,const Point& b,const Point& c){
  return (b.x-a.x)*(c.z-a.z)-(b.z-a.z)*(c.x-a.x);
}

inline bool OnSegment(const Point& a,const Point& b,const Point& p,double tol){
  const double dx=b.x-a.x,dz=b.z-a.z;
  const double len=std::sqrt(dx*dx+dz*dz);
  return len>0 && std::abs(Cross(a,b,p))<=tol*len
      && p.x>=std::min(a.x,b.x)-tol && p.x<=std::max(a.x,b.x)+tol
      && p.z>=std::min(a.z,b.z)-tol && p.z<=std::max(a.z,b.z)+tol;
}

// Fail before deleting anything if the current material outline is invalid.
inline void ValidateOutline(const std::vector<Point>& poly,double dp){
  if(poly.size()<3)throw std::runtime_error("Ice outline needs at least three vertices.");
  const double tol=1e-6*dp;
  double area=0;
  for(size_t i=0;i<poly.size();i++){
    const auto& a=poly[i];const auto& b=poly[(i+1)%poly.size()];
    if(!std::isfinite(a.x) || !std::isfinite(a.z))
      throw std::runtime_error("Non-finite ice outline.");
    if(std::hypot(b.x-a.x,b.z-a.z)<=tol)
      throw std::runtime_error("Collapsed ice outline edge.");
    area+=(a.x-poly[0].x)*(b.z-poly[0].z)
         -(b.x-poly[0].x)*(a.z-poly[0].z);
    for(size_t j=i+1;j<poly.size();j++){
      if(j==i+1 || (i==0 && j+1==poly.size()))continue;
      const auto& c=poly[j];const auto& e=poly[(j+1)%poly.size()];
      if(std::max(a.x,b.x)+tol<std::min(c.x,e.x)
          || std::max(c.x,e.x)+tol<std::min(a.x,b.x)
          || std::max(a.z,b.z)+tol<std::min(c.z,e.z)
          || std::max(c.z,e.z)+tol<std::min(a.z,b.z))continue;
      const double u=Cross(a,b,c),v=Cross(a,b,e);
      const double w=Cross(c,e,a),t=Cross(c,e,b);
      const double abtol=tol*std::hypot(b.x-a.x,b.z-a.z);
      const double cetol=tol*std::hypot(e.x-c.x,e.z-c.z);
      if(((u>abtol && v<-abtol)||(u<-abtol && v>abtol))
          && ((w>cetol && t<-cetol)||(w<-cetol && t>cetol)))
        throw std::runtime_error("Self-intersecting ice outline.");
      if(OnSegment(a,b,c,tol)||OnSegment(a,b,e,tol)
          ||OnSegment(c,e,a,tol)||OnSegment(c,e,b,tol))
        throw std::runtime_error("Touching or overlapping ice outline edges.");
    }
  }
  if(std::abs(area)<=tol*tol)
    throw std::runtime_error("Degenerate ice outline.");
}

// Strict interior only: no outward expansion at sides, bottom or corners.
// Above() separately includes the exterior top skin.
inline bool InsideIce(const std::vector<Point>& poly,double x,double z,double dp){
  if(poly.size()<3 || !std::isfinite(x) || !std::isfinite(z))return false;
  double xmin=poly[0].x,xmax=xmin,zmin=poly[0].z,zmax=zmin;
  for(const auto& a:poly){
    xmin=std::min(xmin,a.x);xmax=std::max(xmax,a.x);
    zmin=std::min(zmin,a.z);zmax=std::max(zmax,a.z);
  }
  const double tol=1e-4*dp;
  if(x<xmin || x>xmax || z<zmin || z>zmax)return false;
  bool inside=false;
  double nearest2=std::numeric_limits<double>::max();
  for(size_t i=0,j=poly.size()-1;i<poly.size();j=i++){
    const auto& a=poly[j];const auto& b=poly[i];
    const double dx=b.x-a.x,dz=b.z-a.z,l2=dx*dx+dz*dz;
    if(l2>0){
      const double t=std::max(0.0,std::min(1.0,((x-a.x)*dx+(z-a.z)*dz)/l2));
      const double ex=x-a.x-t*dx,ez=z-a.z-t*dz;
      nearest2=std::min(nearest2,ex*ex+ez*ez);
    }
    if((a.z>z)!=(b.z>z)){
      const double crossx=a.x+(z-a.z)*(b.x-a.x)/(b.z-a.z);
      if(x<crossx)inside=!inside;
    }
  }
  // Preserve the outline and a small inward tolerance band.
  return inside && nearest2>tol*tol;
}

} // namespace IceTopCleanup

namespace IceTopNoPen {

struct Point {
  double x,z,vx,vz;
};

struct Result {
  bool hit=false;
  double dx=0,dz=0,dvx=0,dvz=0;
};

// Finite top segments only: no side-wall/end-cap correction and no friction.
inline Result Correct(
    const std::vector<Point>& oldtop,
    const std::vector<Point>& top,
    const Point& oldp,const Point& p,double dp)
{
  Result best;
  if(top.size()<2 || oldtop.size()!=top.size())return best;

  const double r=0.5*dp,tol=1e-6*dp;

  auto it=std::upper_bound(
      top.begin(),top.end(),p.x,
      [](double x,const Point& q){return x<q.x;});

  const size_t center=
      it==top.begin()?0:
      it==top.end()?top.size()-2:
      size_t(it-top.begin())-1;

  const size_t first=center?center-1:0;
  const size_t last=std::min(top.size()-2,center+1);

  double bestdepth=0;

  for(size_t i=first;i<=last;i++){
    const Point& a=top[i];
    const Point& b=top[i+1];

    const double tx=b.x-a.x,tz=b.z-a.z;
    const double len2=tx*tx+tz*tz;
    if(!(len2>0))continue;

    const double t=((p.x-a.x)*tx+(p.z-a.z)*tz)/len2;

    // Release beyond a material top endpoint.
    if(t<0 || t>1)continue;

    const double len=std::sqrt(len2);
    const double nx=-tz/len,nz=tx/len;
    const double d=(p.x-a.x)*nx+(p.z-a.z)*nz;
    if(d>=r)continue;

    const Point& oa=oldtop[i];
    const Point& ob=oldtop[i+1];

    const double otx=ob.x-oa.x,otz=ob.z-oa.z;
    const double olen=std::sqrt(otx*otx+otz*otz);
    if(!(olen>0))continue;

    const double oldd=
        ((oldp.x-oa.x)*(-otz)+(oldp.z-oa.z)*otx)/olen;

    // Do not grab side/underside or already internal fluid.
    if(oldd<r-tol)continue;

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

} // namespace IceTopNoPen

#endif