#ifndef USER_ICE_INOUT_DATA_H
#define USER_ICE_INOUT_DATA_H

#include "DualSphDef.h"
#include "UserIceTopProtection.h"
#include <map>
#include <string>
#include <stdexcept>
#include <cfloat>

// Runtime ice/InOut geometry checks only; no checkpoint files are written.
class UserIceInOutData {
  static void Require(bool ok,const char* text){
    if(!ok)throw std::runtime_error(std::string("Ice/InOut compatibility: ")+text);
  }
  static std::map<const void*,std::vector<unsigned>>& Outlines(){
    static std::map<const void*,std::vector<unsigned>> data;
    return data;
  }
public:
  static void ConfigureOutline(const void* owner,
      const std::vector<IceTopCleanup::Point>& poly){
    auto& ids=Outlines()[owner];ids.clear();
    for(const auto& q:poly)ids.push_back(q.id);
  }
  static void ValidateBuffers(const void* owner,unsigned nfixed,unsigned npb,
      unsigned np,const unsigned* ridp,const unsigned* idp,const typecode* code,
      const tdouble3* pos,double dp){
    auto it=Outlines().find(owner);
    if(it==Outlines().end())return;
    std::vector<IceTopCleanup::Point> poly;
    double xmin=DBL_MAX,xmax=-DBL_MAX,zmin=DBL_MAX,zmax=-DBL_MAX;
    for(unsigned id:it->second){
      Require(ridp && id>=nfixed,"invalid outline material ID.");
      const unsigned p=ridp[id-nfixed];
      Require(p<npb && idp[p]==id,"outline material particle is missing.");
      const auto q=pos[p];
      Require(std::isfinite(q.x) && std::isfinite(q.z),"nonfinite ice outline.");
      poly.push_back({q.x,q.z,id});
      xmin=std::min(xmin,q.x);xmax=std::max(xmax,q.x);
      zmin=std::min(zmin,q.z);zmax=std::max(zmax,q.z);
    }
    const double radius=0.5*dp,r2=radius*radius;
    for(unsigned p=npb;p<np;p++){
      if(!CODE_IsNormal(code[p]) || !CODE_IsFluidInout(code[p]))continue;
      const double x=pos[p].x,z=pos[p].z;
      if(x<xmin-radius || x>xmax+radius || z<zmin-radius || z>zmax+radius)continue;
      bool hit=IceTopCleanup::InsideIce(poly,x,z,dp);
      for(size_t i=0;!hit && i<poly.size();i++){
        const auto& a=poly[i];const auto& b=poly[(i+1)%poly.size()];
        const double dx=b.x-a.x,dz=b.z-a.z,l2=dx*dx+dz*dz;
        Require(l2>0,"collapsed ice outline edge.");
        const double u=std::max(0.0,std::min(1.0,((x-a.x)*dx+(z-a.z)*dz)/l2));
        const double ex=x-a.x-u*dx,ez=z-a.z-u*dz;
        hit=ex*ex+ez*ez<r2;
      }
      Require(!hit,"inlet/outlet buffer overlaps ice (including the half-Dp surface skin); move the inlet away from ice.");
    }
  }
};
#endif
