#ifndef USERFLEXKINEMATICS_H
#define USERFLEXKINEMATICS_H

#include "DualSphDef.h"

#include <array>
#include <vector>
#include <functional>
#include <utility>
#include <cmath>

class UserParameters;

namespace FlexKinematics{

using V=std::array<double,3>;
using M=std::array<double,9>;

inline V add(V a,V b){
  for(int k=0;k<3;k++)a[k]+=b[k];
  return a;
}

inline V mul(V a,double s){
  for(double& v:a)v*=s;
  return a;
}

inline double dot(V a,V b){
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

inline double norm(V a){
  return std::sqrt(dot(a,a));
}

inline V vec(tdouble3 a,bool twoD){
  return {{a.x,twoD?0:a.y,a.z}};
}

inline V mv(const M& a,V b){
  V c={{0,0,0}};
  for(int k=0;k<3;k++)
    for(int l=0;l<3;l++)
      c[k]+=a[3*k+l]*b[l];
  return c;
}

inline V mtv(const M& a,V b){
  V c={{0,0,0}};
  for(int k=0;k<3;k++)
    for(int l=0;l<3;l++)
      c[k]+=a[3*l+k]*b[l];
  return c;
}

inline void outer(M& a,V b,V c,double s=1){
  for(int k=0;k<3;k++)
    for(int l=0;l<3;l++)
      a[3*k+l]+=s*b[k]*c[l];
}

struct Edge{
  unsigned j;
  V g,q,r;
  double length=0;
  double qnorm=0;
};

// Contiguous edge storage.
struct PackedRows{
  struct Index{
    size_t first=0;
    size_t len=0;
  };

  struct View{
    Edge* ptr;
    size_t len;

    Edge* begin(){return ptr;}
    Edge* end(){return len?ptr+len:ptr;}
    bool empty()const{return len==0;}
    size_t size()const{return len;}
    Edge& operator[](size_t k){return ptr[k];}
  };

  std::vector<Index> index;
  std::vector<Edge> data;

  void assign(std::vector<std::vector<Edge>>& source){
    size_t count=0;
    for(const auto& row:source)count+=row.size();

    data.reserve(count);
    index.resize(source.size());

    for(size_t i=0;i<source.size();i++){
      index[i].first=data.size();
      index[i].len=source[i].size();
      data.insert(data.end(),source[i].begin(),source[i].end());
    }

    std::vector<std::vector<Edge>>().swap(source);
  }

  size_t size()const{return index.size();}

  View operator[](size_t i){
    return {
        index[i].len?data.data()+index[i].first:nullptr,
        index[i].len};
  }
};

// All particle indices in this cache are elastic-local indices.
//
// L[a,b]=d(v_a)/d(x_b), stored in row-major order.
// g and q already contain neighbour reference-volume weights.
//
// This is an auxiliary WLS spatial gradient. It is not automatically
// the constitutive Fdot*inverse(F) if physical F uses another operator.
struct Cache{
  Cache()=default;
  Cache(const Cache&)=delete;
  Cache& operator=(const Cache&)=delete;

  bool twoD=false;
  unsigned long long revision=0;

  PackedRows rows;
  std::vector<std::vector<std::pair<unsigned,unsigned>>> incoming;

  std::vector<double> mass,volume,qsize;
  std::vector<V> X,v,sumq;
  std::vector<M> L;

  bool HasSupport(unsigned i)const{
    return rows.index.at(i).len!=0;
  }

  const M& VelocityGradient(unsigned i)const{
    return L.at(i);
  }

  M StrainRate(unsigned i)const{
    M a={{0}};
    const M& l=L.at(i);

    for(int r=0;r<3;r++)
      for(int c=0;c<3;c++)
        a[3*r+c]=.5*(l[3*r+c]+l[3*c+r]);

    return a;
  }

  M Spin(unsigned i)const{
    M a={{0}};
    const M& l=L.at(i);

    for(int r=0;r<3;r++)
      for(int c=0;c<3;c++)
        a[3*r+c]=.5*(l[3*r+c]-l[3*c+r]);

    return a;
  }
};

// Call once per force-evaluation stage, including predictor/corrector.
// Always refreshes current data: do not deduplicate by step or time.
// Consumers reuse the returned const Cache within the same stage.
//
// Reset caches if reference geometry, neighbour topology or reference
// material data change.
const Cache& Update(
    const UserParameters& user,
    bool twoD,
    unsigned n,
    const tdouble3* X,
    const tdouble3* x,
    const tfloat4* vel,
    const typecode* code,
    const unsigned* rid,
    const unsigned* count,
    const unsigned* const* pairs,
    const StFlexStrucData* material,
    const std::function<double(double)>& W);

}

#endif