#include "UserFlexDissipation.h"
#include "UserParameters.h"
#include "OmpDefs.h"

#include <algorithm>
#include <stdexcept>
#include <cstdio>

namespace FlexDissipation{

using namespace FlexKinematics;

struct DampingEdge{
  V y;
  double weight=0;
  double bound=0;
};

struct Cache{
  std::shared_ptr<const FlexKinematics::Cache> geometry;

  std::vector<DampingEdge> edges;
  std::vector<double> cp,cs,bound,adjbound;
  std::vector<V> base,force;
  std::vector<M> T;

  unsigned lastStep=~0u;
  FILE* log=nullptr;

  ~Cache(){
    if(log)std::fclose(log);
  }
};

double Apply(
    const UserParameters& u,
    const FlexKinematics::Cache& k,
    const tdouble3* x,
    const typecode* code,
    const unsigned* rid,
    const StFlexStrucData* material,
    const std::function<double(double)>& dW,
    tfloat3* ace,
    double time,
    unsigned step)
{
  if(!u.ElasticHighFrequencyDissipation || k.rows.size()==0)
    return 0;

  if(u.ElasticKinematicsCache.get()!=&k || !k.revision)
    throw std::runtime_error(
        "HF: update public kinematics before applying dissipation.");

  const unsigned n=unsigned(k.rows.size());

  // Rebuild damping data if the reference kinematics cache is replaced.
  if(!u.ElasticDissipationCache
      || u.ElasticDissipationCache->geometry.get()!=&k)
  {
    auto p=std::make_shared<Cache>();
    auto& c=*p;

    c.geometry=u.ElasticKinematicsCache;
    c.edges.resize(k.rows.data.size());
    c.cp.resize(n);
    c.cs.resize(n);
    c.bound.resize(n);
    c.adjbound.resize(n);
    c.base.resize(n);
    c.force.resize(n);
    c.T.resize(n);

    for(unsigned i=0;i<n;i++){
      const auto& m=
          material[CODE_GetIbodyFlexStruc(code[rid[i]])];

      const double mu=m.youngmod/(2*(1+m.poissratio));
      const double la=
          m.youngmod*m.poissratio
          /((1+m.poissratio)*(1-2*m.poissratio));

      c.cp[i]=std::sqrt((la+2*mu)/m.rho0);
      c.cs[i]=std::sqrt(mu/m.rho0);

      if(!(c.cp[i]>0
          && c.cs[i]>0
          && std::isfinite(c.cp[i])
          && std::isfinite(c.cs[i])))
        throw std::runtime_error(
            "HF: invalid material wave speed.");

      for(size_t a=0;a<k.rows.index[i].len;a++){
        const auto& e=k.rows.data[k.rows.index[i].first+a];
        auto& d=c.edges[&e-k.rows.data.data()];

        const V r=add(k.X[e.j],mul(k.X[i],-1));

        const double rho=
            2/(k.volume[i]/k.mass[i]
              +k.volume[e.j]/k.mass[e.j]);

        d.weight=std::max(0.,-dW(dot(r,r)));
        d.weight*=.25*rho*k.volume[i]*k.volume[e.j];
      }
    }

    c.log=std::fopen("ElasticHighFrequency.csv","w");

    if(c.log){
      std::fprintf(
          c.log,
          "time,step,power,negative_residual,"
          "fx,fy,fz,tx,ty,tz,lambda_bound,max_acceleration\n");
    }

    u.ElasticDissipationCache=p;
  }

  auto& c=*u.ElasticDissipationCache;

  const bool diagnose=
      c.log && step!=c.lastStep && step%10==0;

  double residualPower=0;

  // Compute residuals using the shared current-stage gradient.
  #ifdef OMP_USE
  #pragma omp parallel for schedule(static) reduction(+:residualPower)
  #endif
  for(unsigned i=0;i<n;i++){
    for(size_t a=0;a<k.rows.index[i].len;a++){
      const auto& e=k.rows.data[k.rows.index[i].first+a];
      auto& d=c.edges[&e-k.rows.data.data()];

      const unsigned j=e.j;
      const double length=e.length;

      if(!(length>0)){
        d.y={{0,0,0}};
        d.bound=0;
        continue;
      }

      const V jump=add(
          add(k.v[j],mul(k.v[i],-1)),
          mul(
              add(mv(k.L[i],e.r),mv(k.L[j],e.r)),
              -.5));

      const V normal=mul(e.r,1/length);
      const double cp=.5*(c.cp[i]+c.cp[j]);
      const double cs=.5*(c.cs[i]+c.cs[j]);

      d.y=mul(
          add(
              mul(jump,cs),
              mul(normal,(cp-cs)*dot(normal,jump))),
          d.weight);

      if(diagnose)
        residualPower-=dot(jump,d.y);

      d.bound=
          d.weight*cp
          *(2+.5*length*(k.qsize[i]+k.qsize[j]));
    }
  }

  // First part of the adjoint assembly.
  #ifdef OMP_USE
  #pragma omp parallel for schedule(static)
  #endif
  for(unsigned i=0;i<n;i++){
    M T={{0}};
    V base={{0,0,0}};
    double b=0,z=0;

    for(size_t a=0;a<k.rows.index[i].len;a++){
      const auto& e=k.rows.data[k.rows.index[i].first+a];
      const auto& d=c.edges[&e-k.rows.data.data()];

      base=add(base,d.y);
      outer(T,d.y,e.r,.5);
      b+=d.bound;
      z+=.5*d.bound*e.length;
    }

    for(auto p:k.incoming[i]){
      const auto& e=
          k.rows.data[k.rows.index[p.first].first+p.second];
      const auto& d=c.edges[&e-k.rows.data.data()];

      base=add(base,mul(d.y,-1));
      outer(T,d.y,e.r,.5);
      b+=d.bound;
      z+=.5*d.bound*e.length;
    }

    c.T[i]=T;
    c.base[i]=base;
    c.bound[i]=b+z*norm(k.sumq[i]);
    c.adjbound[i]=z;
  }

  double power=0;
  double fx=0,fy=0,fz=0;
  double tx=0,ty=0,tz=0;
  double rate=0,maxa=0;

  // Complete f=-R^T H R v.
  #ifdef OMP_USE
  #pragma omp parallel for schedule(static) reduction(+:power,fx,fy,fz,tx,ty,tz) reduction(max:rate,maxa)
  #endif
  for(unsigned i=0;i<n;i++){
    V f=add(
        c.base[i],
        mul(mv(c.T[i],k.sumq[i]),-1));

    double b=c.bound[i];

    for(auto p:k.incoming[i]){
      const auto& e=
          k.rows.data[k.rows.index[p.first].first+p.second];

      f=add(f,mv(c.T[p.first],e.q));
      b+=c.adjbound[p.first]*e.qnorm;
    }

    c.force[i]=f;

    if(diagnose){
      power+=dot(k.v[i],f);
      fx+=f[0];
      fy+=f[1];
      fz+=f[2];

      const V xx=vec(x[rid[i]],k.twoD);

      tx+=xx[1]*f[2]-xx[2]*f[1];
      ty+=xx[2]*f[0]-xx[0]*f[2];
      tz+=xx[0]*f[1]-xx[1]*f[0];

      maxa=std::max(maxa,norm(f)/k.mass[i]);
    }

    // The stability bound is evaluated on every call.
    rate=std::max(rate,b/k.mass[i]);

    if(CODE_IsFlexStrucFlex(code[rid[i]])){
      ace[rid[i]].x+=float(f[0]/k.mass[i]);
      ace[rid[i]].y+=float(f[1]/k.mass[i]);
      ace[rid[i]].z+=float(f[2]/k.mass[i]);
    }
  }

  if(diagnose){
    std::fprintf(
        c.log,
        "%.12g,%u,%.12g,%.12g,%.12g,%.12g,"
        "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
        time,step,power,residualPower,
        fx,fy,fz,tx,ty,tz,rate,maxa);

    std::fflush(c.log);
    c.lastStep=step;
  }

  return rate;
}

}