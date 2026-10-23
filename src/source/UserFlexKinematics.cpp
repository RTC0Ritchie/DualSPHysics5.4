#include "UserFlexKinematics.h"
#include "UserParameters.h"
#include "OmpDefs.h"

#include <algorithm>
#include <stdexcept>

namespace FlexKinematics{

static bool inverse(M a,M& b){
  b={{1,0,0,0,1,0,0,0,1}};

  double scale=0;
  for(double v:a){
    if(!std::isfinite(v))return false;
    scale=std::max(scale,std::abs(v));
  }
  if(!(scale>0))return false;

  for(int k=0;k<3;k++){
    int p=k;

    for(int j=k+1;j<3;j++){
      if(std::abs(a[3*j+k])>std::abs(a[3*p+k]))
        p=j;
    }

    if(!std::isfinite(a[3*p+k])
        || std::abs(a[3*p+k])<1e-12*scale)
      return false;

    for(int j=0;j<3;j++){
      std::swap(a[3*p+j],a[3*k+j]);
      std::swap(b[3*p+j],b[3*k+j]);
    }

    const double z=a[3*k+k];
    for(int j=0;j<3;j++){
      a[3*k+j]/=z;
      b[3*k+j]/=z;
    }

    for(int i=0;i<3;i++){
      if(i==k)continue;

      const double z=a[3*i+k];
      for(int j=0;j<3;j++){
        a[3*i+j]-=z*a[3*k+j];
        b[3*i+j]-=z*b[3*k+j];
      }
    }
  }

  return true;
}

const Cache& Update(
    const UserParameters& u,
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
    const std::function<double(double)>& W)
{
  // Build reference geometry once.
  if(!u.ElasticKinematicsCache){
    auto p=std::make_shared<Cache>();
    auto& c=*p;
    c.twoD=twoD;

    std::vector<std::vector<Edge>> buildRows(n);

    c.incoming.resize(n);
    c.mass.resize(n);
    c.volume.resize(n);
    c.qsize.resize(n);
    c.X.resize(n);
    c.v.resize(n);
    c.sumq.resize(n);
    c.L.resize(n);

    const auto& state=u.FlexRestartState;

    if(u.IsFlexRestart()
        && (!state.HasRestartStateLayout() || state.Idp.size()!=n))
      throw std::runtime_error(
          "Elastic kinematics: missing restart state.");

    for(unsigned i=0;i<n;i++){
      const auto& m=
          material[CODE_GetIbodyFlexStruc(code[rid[i]])];

      c.X[i]=vec(X[i],twoD);

      c.volume[i]=u.IsFlexRestart()
          ? state.ReferenceVolume[i]
          : m.vol0;

      c.mass[i]=u.IsFlexRestart()
          ? state.ParticleMass[i]
          : m.vol0*m.rho0;

      if(!(c.mass[i]>0
          && c.volume[i]>0
          && std::isfinite(c.mass[i])
          && std::isfinite(c.volume[i])))
        throw std::runtime_error(
            "Elastic kinematics: invalid mass or reference volume.");
    }

    for(unsigned i=0;i<n;i++){
      M moment={{0}},inv;

      for(unsigned k=0;k<count[i];k++){
        const unsigned j=pairs[i][k];

        if(j>=n || j==i)
          throw std::runtime_error(
              "Elastic kinematics: invalid neighbour index.");

        if(CODE_GetIbodyFlexStruc(code[rid[i]])
            !=CODE_GetIbodyFlexStruc(code[rid[j]]))
          continue;

        const V r=add(
            vec(X[j],twoD),
            mul(vec(X[i],twoD),-1));

        const double rr=dot(r,r);
        const double w=c.volume[j]*W(rr);

        if(!(rr>0))
          throw std::runtime_error(
              "Elastic kinematics: duplicate reference particles.");

        if(w>0){
          Edge e;
          e.j=j;
          e.g=mul(r,w);

          buildRows[i].push_back(e);
          outer(moment,r,r,w);
        }
      }

      // Isolated clamp particles may have no structural neighbours.
      if(buildRows[i].empty())continue;

      if(twoD)
        moment[4]=std::max(moment[0],moment[8]);

      if(!inverse(moment,inv))
        throw std::runtime_error(
            "Elastic kinematics: singular reference reconstruction.");

      for(unsigned k=0;k<buildRows[i].size();k++){
        auto& e=buildRows[i][k];
        e.g=mv(inv,e.g);
        c.incoming[e.j].push_back({i,k});
      }
    }

    c.rows.assign(buildRows);
    u.ElasticKinematicsCache=p;
  }

  auto& c=*u.ElasticKinematicsCache;

  if(c.rows.size()!=n || c.twoD!=twoD)
    throw std::runtime_error(
        "Elastic kinematics: layout changed; reset caches first.");

  // Refresh velocities for every force-evaluation stage.
  #ifdef OMP_USE
  #pragma omp parallel for schedule(static)
  #endif
  for(unsigned i=0;i<n;i++){
    const auto a=vel[rid[i]];
    c.v[i]={{a.x,twoD?0:a.y,a.z}};
  }

  int invalid=0;

  #ifdef OMP_USE
  #pragma omp parallel for schedule(static) reduction(+:invalid)
  #endif
  for(unsigned i=0;i<n;i++){
    M F={{0}},inv,L={{0}};
    V sum={{0,0,0}};
    double qs=0;

    // Auxiliary geometric F only; physical defgrad is not modified.
    for(auto& e:c.rows[i]){
      e.r=add(
          vec(x[rid[e.j]],twoD),
          mul(vec(x[rid[i]],twoD),-1));

      outer(F,e.r,e.g);
    }

    if(twoD)F[4]=1;

    if(!c.rows[i].empty() && !inverse(F,inv)){
      invalid++;
      continue;
    }

    for(auto& e:c.rows[i]){
      e.length=norm(e.r);
      e.q=mtv(inv,e.g);
      e.qnorm=norm(e.q);

      sum=add(sum,e.q);
      qs+=e.qnorm;

      outer(
          L,
          add(c.v[e.j],mul(c.v[i],-1)),
          e.q);
    }

    c.L[i]=L;
    c.sumq[i]=sum;
    c.qsize[i]=qs+norm(sum);
  }

  if(invalid)
    throw std::runtime_error(
        "Elastic kinematics: singular current reconstruction.");

  ++c.revision;
  return c;
}

}