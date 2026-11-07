#include "UserFlexKinematics.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// A sparse index avoids allocating a bounding-box grid for disconnected bodies.
class UserFlexNormalGrid{
  typedef std::array<std::int64_t,3> Cell;
  struct Hash{
    size_t operator()(const Cell& c)const{
      size_t h=0;
      for(unsigned k=0;k<3;k++)h^=std::hash<std::int64_t>()(c[k])
        +size_t(0x9e3779b9)+(h<<6)+(h>>2);
      return h;
    }
  };
  std::unordered_map<Cell,std::vector<unsigned>,Hash> Cells;
  const tdouble3* Pos;
  const bool TwoD;
  const double Width;
  const tdouble3 Origin;
  Cell Key(const tdouble3& p)const{
    const double q[3]={(p.x-Origin.x)/Width,TwoD?0.:(p.y-Origin.y)/Width,
      (p.z-Origin.z)/Width};
    Cell c;
    for(unsigned k=0;k<3;k++){
      if(!std::isfinite(q[k]) || std::abs(q[k])>9e18)
        throw std::runtime_error("Elastic normal repair: invalid position or spatial range.");
      c[k]=std::int64_t(std::floor(q[k]));
    }
    return c;
  }
public:
  UserFlexNormalGrid(bool twoD,double dp,unsigned n,const tdouble3* pos)
    :Pos(pos),TwoD(twoD),Width(4*dp),Origin(pos[0]){
    Cells.reserve(n/16+1);
    for(unsigned i=0;i<n;i++)Cells[Key(pos[i])].push_back(i);
  }
  // Weighted solid first moments at two scales. The direction toward missing
  // solid support is outward. Agreement rejects poorly resolved thin surfaces
  // and ambiguous corner/concavity estimates instead of guessing their side.
  void Moments(unsigned i,double dp,tdouble3* outward,double* sum)const{
    const Cell c=Key(Pos[i]);
    const double radii[2]={3*dp,4*dp};
    outward[0]=outward[1]=TDouble3(0);sum[0]=sum[1]=0;
    for(int z=-1;z<=1;z++)for(int y=(TwoD?0:-1);y<=(TwoD?0:1);y++)
    for(int x=-1;x<=1;x++){
      const Cell key={{c[0]+x,c[1]+y,c[2]+z}};
      const auto found=Cells.find(key);if(found==Cells.end())continue;
      for(unsigned j:found->second){
        const double dx=Pos[j].x-Pos[i].x,dy=TwoD?0.:Pos[j].y-Pos[i].y,
          dz=Pos[j].z-Pos[i].z;
        const double r=std::sqrt(dx*dx+dy*dy+dz*dz);
        if(r==0 || r>=radii[1])continue;
        for(unsigned k=0;k<2;k++)if(r<radii[k]){
          const double q=1-r/radii[k],w=q*q;
          outward[k].x-=w*dx;outward[k].y-=w*dy;outward[k].z-=w*dz;
          sum[k]+=w*r;
        }
      }
    }
  }
};

UserFlexKinematics::NormalReport UserFlexKinematics::RepairNormalOrientation(
    bool twoD,double dp,unsigned n,const tdouble3* pos,tfloat3* normal){
  NormalReport report;
  if(!n)return report;
  if(!(dp>0) || !std::isfinite(dp) || !pos || !normal)
    throw std::runtime_error("Elastic normal repair: invalid input.");
  UserFlexNormalGrid grid(twoD,dp,n,pos);
  for(unsigned i=0;i<n;i++){
    const tfloat3 old=normal[i];
    const double length=std::sqrt(double(old.x)*old.x+double(old.y)*old.y+double(old.z)*old.z);
    if(!std::isfinite(length))throw std::runtime_error("Elastic normal repair: non-finite normal.");
    if(length==0)continue;
    report.Checked++;
    tdouble3 a[2];double sum[2];grid.Moments(i,dp,a,sum);
    const double l0=std::sqrt(a[0].x*a[0].x+a[0].y*a[0].y+a[0].z*a[0].z);
    const double l1=std::sqrt(a[1].x*a[1].x+a[1].y*a[1].y+a[1].z*a[1].z);
    if(!(l0>0 && l1>0))continue;
    // A nearly balanced solid neighbourhood supplies no orientation evidence.
    // In particular, do not count roundoff in interior moments as a defect.
    if(l0<.1*sum[0] && l1<.1*sum[1])continue;
    const double dot0=(old.x*a[0].x+old.y*a[0].y+old.z*a[0].z)/(length*l0);
    const double dot1=(old.x*a[1].x+old.y*a[1].y+old.z*a[1].z)/(length*l1);
    if(dot0>=-.2 || dot1>=-.2)continue;
    const double agree=(a[0].x*a[1].x+a[0].y*a[1].y+a[0].z*a[1].z)/(l0*l1);
    if(l0<.1*sum[0] || l1<.1*sum[1] || agree<.9){report.Unresolved++;continue;}
    // Use the local estimate; retain the original distance to the interface.
    normal[i]=TFloat3(float(length*a[0].x/l0),float(length*a[0].y/l0),float(length*a[0].z/l0));
    report.Corrected++;
  }
  return report;
}
