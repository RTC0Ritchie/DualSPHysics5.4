#include "UserFlexDissipation.h"
#include "OmpDefs.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

JFlexDissipation::V JFlexDissipation::Sub(const V& a,const V& b){
  return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};
}

JFlexDissipation::V JFlexDissipation::Scale(V a,double s){
  for(double& q:a)q*=s;
  return a;
}

double JFlexDissipation::Dot(const V& a,const V& b){
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

JFlexDissipation::V JFlexDissipation::Mul(const M& a,const V& b){
  V c={{0,0,0}};

  for(unsigned k=0;k<3;k++)
    for(unsigned l=0;l<3;l++)
      c[k]+=a[3*k+l]*b[l];

  return c;
}

void JFlexDissipation::Outer(M& a,const V& b,const V& c){
  for(unsigned k=0;k<3;k++)
    for(unsigned l=0;l<3;l++)
      a[3*k+l]+=b[k]*c[l];
}

JFlexDissipation::M JFlexDissipation::Inverse(M a){
  M b={{1,0,0,0,1,0,0,0,1}};

  double scale=0;

  for(double q:a){
    if(!std::isfinite(q))
      throw std::runtime_error("Dissipation: invalid moment.");

    scale=std::max(scale,std::abs(q));
  }

  for(unsigned k=0;k<3;k++){
    unsigned p=k;

    for(unsigned j=k+1;j<3;j++)
      if(std::abs(a[3*j+k])>std::abs(a[3*p+k]))
        p=j;

    if(!(std::abs(a[3*p+k])>1e-12*scale))
      throw std::runtime_error(
          "Dissipation: singular reference gradient.");

    for(unsigned l=0;l<3;l++){
      std::swap(a[3*k+l],a[3*p+l]);
      std::swap(b[3*k+l],b[3*p+l]);
    }

    const double pivot=a[3*k+k];

    for(unsigned l=0;l<3;l++){
      a[3*k+l]/=pivot;
      b[3*k+l]/=pivot;
    }

    for(unsigned j=0;j<3;j++)if(j!=k){
      const double q=a[3*j+k];

      for(unsigned l=0;l<3;l++){
        a[3*j+l]-=q*a[3*k+l];
        b[3*j+l]-=q*b[3*k+l];
      }
    }
  }

  return b;
}

JFlexDissipation::V JFlexDissipation::Position(const tdouble3& a)const{
  return {{a.x,TwoD?0.:a.y,a.z}};
}

JFlexDissipation::V JFlexDissipation::Velocity(const tfloat4& a)const{
  return {{a.x,TwoD?0.:double(a.y),a.z}};
}

JFlexDissipation::M JFlexDissipation::Matrix(const tmatrix3f& a){
  return {{a.a11,a.a12,a.a13,a.a21,a.a22,a.a23,a.a31,a.a32,a.a33}};
}

JFlexDissipation::JFlexDissipation(
    bool twoD,unsigned n,const tdouble3* x0,
    const double* volume,const double* rho,const double* young,
    const double* poisson,const unsigned* body,const unsigned* count,
    const unsigned* const* pairs,const tmatrix3f* correction,
    bool mode5,const std::function<float(float)>& fac,
    double h,bool useCR,bool useKV,double factor,int simplifiedMode)
  :TwoD(twoD),CR(useCR),KV(useKV && simplifiedMode==1),
   PairKV(useKV && simplifiedMode==2),PairWeights(PairKV?n:0),
   N(n),H(h),Rows(n),Weights(CR?n:0),ViscWeights(KV?n:0),
   Grad(CR?n:0),Pv(KV?n:0),Rho(rho,rho+n),Cp(CR || PairKV?n:0),
   Eta0(useKV?n:0),SumG(KV?n:0),NormSumG(KV?n:0),
   Q(KV?n:0),ReuseF(n,0)
{
  if(simplifiedMode!=1 && simplifiedMode!=2)
    throw std::runtime_error("KV simplifiedMode must be 1 or 2.");

  const tdouble3* X=x0;

  if(!(H>0) || !std::isfinite(H)
      || factor<0 || !std::isfinite(factor))
    throw std::runtime_error("Dissipation: invalid h or factor.");

  std::vector<double> cs(CR?n:0);

  for(unsigned i=0;i<N;i++){
    if(!(volume[i]>0 && Rho[i]>0 && young[i]>0
        && poisson[i]>-1 && poisson[i]<.5)
        || !std::isfinite(volume[i])
        || !std::isfinite(Rho[i])
        || !std::isfinite(young[i]))
      throw std::runtime_error("Dissipation: invalid material.");

    if(CR || PairKV){
      Cp[i]=std::sqrt(young[i]*(1-poisson[i])
          /((1+poisson[i])*(1-2*poisson[i])*Rho[i]));

      if(CR)cs[i]=std::sqrt(
          young[i]/(2*(1+poisson[i])*Rho[i]));

      if(!std::isfinite(Rho[i]*Cp[i])
          || (CR && !std::isfinite(Rho[i]*cs[i])))
        throw std::runtime_error(
            "Dissipation: invalid wave speed.");

      MaxCp=std::max(MaxCp,Cp[i]);
    }

    if(KV || PairKV){
      const double K=young[i]/(3*(1-2*poisson[i]));

      Eta0[i]=factor*Rho[i]*std::sqrt(K/Rho[i])*H;

      if(!std::isfinite(Eta0[i]))
        throw std::runtime_error("KV: invalid viscosity.");
    }
  }

  // Cache only same-body, nonzero reference interactions.
  for(unsigned i=0;i<N;i++){
    Rows[i].reserve(count[i]);
    if(CR)Weights[i].reserve(count[i]);

    M moment={{0}};
    bool sameBody=true;

    for(unsigned k=0;k<count[i];k++){
      const unsigned j=pairs[i][k];

      if(j>=N || j==i || (k && pairs[i][k-1]>=j))
        throw std::runtime_error(
            "Dissipation: invalid or unsorted pairs.");

      const float dx=float(X[i].x-X[j].x);
      const float dy=float(X[i].y-X[j].y);
      const float dz=float(X[i].z-X[j].z);
      const float rr=dx*dx+dy*dy+dz*dz;

      if(!(rr>0) || !std::isfinite(rr))
        throw std::runtime_error(
            "Dissipation: invalid reference distance.");

      const float f=fac(rr);

      if(f>0 || !std::isfinite(f))
        throw std::runtime_error("Dissipation: invalid kernel.");

      const tfloat3 raw=TFloat3(f*dx,f*dy,f*dz);
      const bool active=body[i]==body[j];

      sameBody=sameBody && active;

      if(!active || f==0)continue;

      const V G={{raw.x,TwoD?0.:double(raw.y),raw.z}};
      const V g=Scale(G,volume[j]);

      Rows[i].push_back({j,g});
      if(PairKV){
        const double w=(TwoD?4.:5.)*(.5*Eta0[i]+.5*Eta0[j])
            *volume[j]/Rho[i]*(-double(f));

        if(w<0 || !std::isfinite(w))
          throw std::runtime_error("KV: invalid pair weight.");

        PairWeights[i].push_back(w);
      }
      Outer(moment,Sub(Position(X[j]),Position(X[i])),g);

      if(CR){
        CRWeight w={Sub(Position(X[j]),Position(X[i])),0,0};

        const auto impedance=[](double a,double b){
          const double lo=std::min(a,b);
          const double hi=std::max(a,b);
          return lo/(1+lo/hi);
        };

        const double s=std::sqrt(Dot(G,G))*volume[j]/Rho[i];

        w.p=s*impedance(Rho[i]*Cp[i],Rho[j]*Cp[j]);
        w.s=s*impedance(Rho[i]*cs[i],Rho[j]*cs[j]);

        if(!std::isfinite(w.p) || !std::isfinite(w.s))
          throw std::runtime_error("cR: invalid impedance.");

        Weights[i].push_back(w);
      }
    }

    if(!CR && !KV)continue;
    bool nonzero=false;
    for(double q:moment)nonzero=nonzero || q!=0;
    if(!nonzero)continue;

    M B;

    if(correction && sameBody){
      B=Matrix(correction[i]);
    }
    else{
      if(TwoD)
        moment[4]=std::max(
            std::abs(moment[0]),std::abs(moment[8]));

      B=Inverse(moment);
    }

    // Solver forms F = (sum dx tensor Vj*Gij) * B.
    std::swap(B[1],B[3]);
    std::swap(B[2],B[6]);
    std::swap(B[5],B[7]);

    ReuseF[i]=correction && sameBody && !mode5;

    V sumg={{0,0,0}};
    for(auto& e:Rows[i]){
      e.g=Mul(B,e.g);
      const double norm=std::sqrt(Dot(e.g,e.g));

      if(!std::isfinite(norm))
        throw std::runtime_error(
            "Dissipation: invalid gradient.");

      if(KV){
        SumG[i]+=norm;
        for(unsigned k=0;k<3;k++)sumg[k]+=e.g[k];
      }
    }
    if(KV){
      NormSumG[i]=std::sqrt(Dot(sumg,sumg));
      if(!std::isfinite(SumG[i]) || !std::isfinite(NormSumG[i]))
        throw std::runtime_error(
            "KV: invalid reference gradient norm.");
    }
  }

  if(PairKV)InitPairWeights();
  // Pay the reverse lookup once, not at every force evaluation.
  if(KV)for(unsigned i=0;i<N;i++){
    ViscWeights[i].reserve(Rows[i].size());

    for(const auto& e:Rows[i]){
      const auto& row=Rows[e.j];

      const auto it=std::lower_bound(
          row.begin(),row.end(),i,
          [](const Edge& a,unsigned j){return a.j<j;});

      if(it==row.end() || it->j!=i)
        throw std::runtime_error(
            "KV: reciprocal pairs are required.");

      const V q=Scale(it->g,-volume[e.j]/volume[i]);

      ViscWeights[i].push_back({
          q,std::sqrt(Dot(q,q))
      });
    }
  }
}

void JFlexDissipation::InitPairWeights(){
  std::vector<double> scale(N,1.);

  for(unsigned i=0;i<N;i++){
    double degree=0;
    for(double w:PairWeights[i])degree+=w;

    const double speed=H*degree;

    if(!std::isfinite(speed))
      throw std::runtime_error("KV: invalid pair degree.");

    if(speed>0)
      scale[i]=std::min(1.,.99998*Cp[i]/speed);
  }

  for(unsigned i=0;i<N;i++){
    double degree=0;

    for(unsigned k=0;k<Rows[i].size();k++){
      const unsigned j=Rows[i][k].j;
      const auto& row=Rows[j];

      const auto it=std::lower_bound(
          row.begin(),row.end(),i,
          [](const Edge& a,unsigned b){return a.j<b;});

      if(it==row.end() || it->j!=i)
        throw std::runtime_error(
            "KV: reciprocal pairs are required.");

      double& w=PairWeights[i][k];
      w*=std::min(scale[i],scale[j]);
      degree+=w;
    }

    PairSpeed=std::max(PairSpeed,H*degree);
  }

  if(!std::isfinite(PairSpeed)
      || PairSpeed>std::numeric_limits<float>::max())
    throw std::runtime_error("KV: pair speed overflow.");
}

bool JFlexDissipation::BuildKV(unsigned i,const M& F,const M& L){
  const double J=F[0]*(F[4]*F[8]-F[5]*F[7])
      -F[1]*(F[3]*F[8]-F[5]*F[6])+F[2]*(F[3]*F[7]-F[4]*F[6]);
  if(!(J>0) || !std::isfinite(J))return false;
  const double eta=Eta0[i]/J;
  M S={{0}},P={{0}};
  for(unsigned a=0;a<3;a++)for(unsigned b=0;b<3;b++)
    for(unsigned k=0;k<3;k++)
      S[3*a+b]+=.5*eta*(F[3*k+a]*L[3*k+b]+L[3*k+a]*F[3*k+b]);
  for(unsigned a=0;a<3;a++)for(unsigned b=0;b<3;b++)
    for(unsigned k=0;k<3;k++)P[3*a+b]+=F[3*a+k]*S[3*k+b];
  double n1=0,ni=0,n2=0;
  for(unsigned a=0;a<3;a++){
    double row=0,col=0;
    for(unsigned b=0;b<3;b++){
      row+=std::abs(F[3*a+b]);col+=std::abs(F[3*b+a]);
      n2+=F[3*a+b]*F[3*a+b];
    }
    ni=std::max(ni,row);n1=std::max(n1,col);
  }
  Q[i]=eta*std::min(n2,n1*ni)*(SumG[i]+NormSumG[i]);
  if(!std::isfinite(Q[i]))return false;
  for(double q:P)if(!std::isfinite(q))return false;
  Pv[i]=P;
  return true;
}

template<bool twoD,bool cr,bool kv,bool pair>
bool JFlexDissipation::PrepareT(
    unsigned i,const tdouble3* x,const tfloat4* velocity,
    const unsigned* rid,const tmatrix3f& existingF)
{
  if(kv){
    Pv[i]=M{{0}};
    Q[i]=0;
  }

  M L={{0}},F={{0}};
  const unsigned p=rid[i];

  const V vi={{
    velocity[p].x,
    twoD?0.:double(velocity[p].y),
    velocity[p].z
  }};

  const bool rebuild=kv && !ReuseF[i];

  if(rebuild){
    const auto& xi=x[p];

    for(const auto& e:Rows[i]){
      const unsigned j=rid[e.j];
      const auto& vj=velocity[j];
      const auto& xj=x[j];

      const V dv={{
        vj.x-vi[0],
        twoD?0.:vj.y-vi[1],
        vj.z-vi[2]
      }};

      const V dx={{
        xj.x-xi.x,
        twoD?0.:xj.y-xi.y,
        xj.z-xi.z
      }};

      for(unsigned a=0;a<3;a++)if(!twoD || a!=1)
        for(unsigned b=0;b<3;b++)if(!twoD || b!=1){
          L[3*a+b]+=dv[a]*e.g[b];
          F[3*a+b]+=dx[a]*e.g[b];
        }
    }

    if(twoD)F[4]=1;
  }
  else{
    for(const auto& e:Rows[i]){
      const auto& vj=velocity[rid[e.j]];

      const V dv={{
        vj.x-vi[0],
        twoD?0.:vj.y-vi[1],
        vj.z-vi[2]
      }};

      for(unsigned a=0;a<3;a++)if(!twoD || a!=1)
        for(unsigned b=0;b<3;b++)if(!twoD || b!=1)
          L[3*a+b]+=dv[a]*e.g[b];
    }
  }

  for(double q:L)if(!std::isfinite(q))return false;

  if(cr)Grad[i]=L;

  if(!kv || Eta0[i]==0 || SumG[i]==0)return true;

  if(!rebuild)F=Matrix(existingF);

  return BuildKV(i,F,L);
}

template<bool twoD,bool cr,bool kv,bool pair>
bool JFlexDissipation::AddParticleT(
    unsigned i,const tdouble3* x,const tfloat4* velocity,
    const unsigned* rid,tdouble3& acceleration,double& rate)const
{
  V a={{0}},av={{0}};
  V xi={{0}},vi={{0}};
  M Li={{0}},Pi={{0}};

  if(cr || pair){
    const unsigned p=rid[i];

    xi={{x[p].x,twoD?0.:x[p].y,x[p].z}};
    vi={{
      velocity[p].x,
      twoD?0.:double(velocity[p].y),
      velocity[p].z
    }};

    if(cr)Li=Grad[i];
  }

  if(kv)Pi=Pv[i];

  const auto& row=Rows[i];

  for(unsigned k=0;k<row.size();k++){
    const auto& e=row[k];
    const unsigned j=e.j;

    if(cr || pair){
      const unsigned p=rid[j];

      const V dv={{
        velocity[p].x-vi[0],
        twoD?0.:velocity[p].y-vi[1],
        velocity[p].z-vi[2]
      }};

      const V r={{
        x[p].x-xi[0],
        twoD?0.:x[p].y-xi[1],
        x[p].z-xi[2]
      }};

      const double rr=Dot(r,r);

      if(!(rr>0) || !std::isfinite(rr))return false;

      const V normal=Scale(r,1/std::sqrt(rr));

      if(pair){
        const double damp=PairWeights[i][k]*Dot(dv,normal);

        for(unsigned t=0;t<3;t++)if(!twoD || t!=1)
          a[t]+=damp*normal[t];
      }

      if(cr){
        const auto& w=Weights[i][k];
        const auto& Lj=Grad[j];
        V jump=dv;

        for(unsigned a=0;a<3;a++)if(!twoD || a!=1){
          double gi=0,gj=0;

          for(unsigned b=0;b<3;b++)if(!twoD || b!=1){
            gi+=Li[3*a+b]*w.R[b];
            gj+=Lj[3*a+b]*w.R[b];
          }

          jump[a]-=.5*(gi+gj);
        }

        const double un=Dot(jump,normal);

        if(!std::isfinite(un))return false;

        if(un<0)
          for(unsigned t=0;t<3;t++)if(!twoD || t!=1)
            a[t]+=w.s*jump[t]+(w.p-w.s)*un*normal[t];
      }
    }

    if(kv){
      const auto& w=ViscWeights[i][k];
      const auto& Pj=Pv[j];

      for(unsigned a=0;a<3;a++)if(!twoD || a!=1){
        double ai=0,aj=0;

        for(unsigned b=0;b<3;b++)if(!twoD || b!=1){
          ai+=Pi[3*a+b]*e.g[b];
          aj+=Pj[3*a+b]*w.q[b];
        }

        av[a]+=ai+aj;
      }

      rate+=Q[j]*w.norm;
    }
  }

  if(kv){
    const double invrho=1/Rho[i];

    for(unsigned t=0;t<3;t++)
      a[t]+=av[t]*invrho;
  }

  for(double q:a)if(!std::isfinite(q))return false;

  acceleration.x+=a[0];
  acceleration.y+=a[1];
  acceleration.z+=a[2];

  return true;
}

// Called from a compile-time-specialized force pass, not via per-particle
// runtime dispatch. Each particle writes only its own acceleration.
template<bool twoD,bool cr,bool kv,bool pair>
bool JFlexDissipation::Accumulate(
    unsigned i,const tdouble3* x,const tfloat4* velocity,
    const unsigned* rid,tfloat3* ace,double& speed)const
{
  tdouble3 da=TDouble3(0);
  double rate=kv?Q[i]*NormSumG[i]:0;
  if(!AddParticleT<twoD,cr,kv,pair>(i,x,velocity,rid,da,rate))return false;

  const double limit=std::numeric_limits<float>::max();
  if(!std::isfinite(da.x) || std::abs(da.x)>limit
      || !std::isfinite(da.y) || std::abs(da.y)>limit
      || !std::isfinite(da.z) || std::abs(da.z)>limit)return false;

  if(kv){
    const double value=H*rate/Rho[i];
    if(!std::isfinite(value) || value<0 || value>limit)return false;
    speed=std::max(speed,value);
  }

  const unsigned p=rid[i];
  ace[p].x+=float(da.x);
  ace[p].y+=float(da.y);
  ace[p].z+=float(da.z);
  return true;
}

template<bool twoD,bool cr,bool kv,bool pair>
double JFlexDissipation::ApplyT(
    const tdouble3* x,const tfloat4* velocity,
    const typecode* code,const unsigned* rid,
    const tmatrix3f* defgrad,tfloat3* ace)
{
  int invalid=0;
  const tmatrix3f zero={};

  if(cr || kv){
    #ifdef OMP_USE
    #pragma omp parallel for schedule(static) reduction(+:invalid)
    #endif
    for(int ii=0;ii<int(N);ii++){
      const unsigned i=unsigned(ii);
      const tmatrix3f& F=kv && defgrad?defgrad[i]:zero;

      if(!PrepareT<twoD,cr,kv,pair>(
          i,x,velocity,rid,F))invalid++;
    }

    if(invalid)
      throw std::runtime_error(
          "Elastic dissipation: invalid stage gradient or stress.");
  }

  double speed=0;

  if(kv){
    #ifdef OMP_USE
    #pragma omp parallel for schedule(static) reduction(+:invalid) reduction(max:speed)
    #endif
    for(int ii=0;ii<int(N);ii++){
      const unsigned i=unsigned(ii);

      if(!CODE_IsFlexStrucFlex(code[rid[i]]))continue;

      if(!Accumulate<twoD,cr,kv,pair>(
          i,x,velocity,rid,ace,speed))invalid++;
    }
  }
  else{
    #ifdef OMP_USE
    #pragma omp parallel for schedule(static) reduction(+:invalid)
    #endif
    for(int ii=0;ii<int(N);ii++){
      const unsigned i=unsigned(ii);

      if(!CODE_IsFlexStrucFlex(code[rid[i]]))continue;

      double unused=0;

      if(!Accumulate<twoD,cr,kv,pair>(
          i,x,velocity,rid,ace,unused))invalid++;
    }
  }

  if(invalid)
    throw std::runtime_error(
        "Elastic dissipation: invalid stage acceleration.");

  return std::max(
      std::max(cr?MaxCp:0.,speed),
      pair?PairSpeed:0.);
}

double JFlexDissipation::Apply(
    unsigned n,const tdouble3* x,const tfloat4* velocity,
    const typecode* code,const unsigned* rid,
    const tmatrix3f* defgrad,tfloat3* ace)
{
  if(n!=N)
    throw std::runtime_error("Elastic dissipation: rebuild SPH reference cache.");
  if(!CR && !KV && !PairKV)return 0;
  if(MaxCp>std::numeric_limits<float>::max())
    throw std::runtime_error("Elastic dissipation: reference wave speed overflow.");

  if(PairKV){
    if(TwoD){
      if(CR)
        return ApplyT<true,true,false,true>(
            x,velocity,code,rid,defgrad,ace);

      return ApplyT<true,false,false,true>(
          x,velocity,code,rid,defgrad,ace);
    }

    if(CR)
      return ApplyT<false,true,false,true>(
          x,velocity,code,rid,defgrad,ace);

    return ApplyT<false,false,false,true>(
        x,velocity,code,rid,defgrad,ace);
  }
  // Select the active operators and dimension once per force evaluation.
  if(TwoD){
    if(CR && KV)return ApplyT<true,true,true>(x,velocity,code,rid,defgrad,ace);
    if(CR)return ApplyT<true,true,false>(x,velocity,code,rid,defgrad,ace);
    return ApplyT<true,false,true>(x,velocity,code,rid,defgrad,ace);
  }
  if(CR && KV)return ApplyT<false,true,true>(x,velocity,code,rid,defgrad,ace);
  if(CR)return ApplyT<false,true,false>(x,velocity,code,rid,defgrad,ace);
  return ApplyT<false,false,true>(x,velocity,code,rid,defgrad,ace);
}
