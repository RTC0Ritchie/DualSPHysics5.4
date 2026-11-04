// CPU counterpart: UserFlexDissipation.cpp. Reference cache is uploaded once;
// both stage passes, including the time-step bound, remain on the GPU.
// Build this module with --fmad=false: cR gates the complete vector jump on
// un<0. Fusion can flip a roundoff-sized normal jump and enable a finite
// tangential force. Keep CPU-style operation ordering without a new tolerance.
#include "UserGFlexDissipation.h"
#include "UserGFlexPrecision.h"
#include "JSphGpuSingle.h"
#include "UserParameters.h"
#include "UserGParameters.h"
#include "FunSphKernel.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <stdexcept>
#include <vector>
#include <type_traits>
class UserGDissTypes{
public:
  template<unsigned n> struct Array{
    double v[n];
    __host__ __device__ double& operator[](unsigned i){return v[i];}
    __host__ __device__ const double& operator[](unsigned i)const{return v[i];}
    __host__ __device__ double* begin(){return v;}
    __host__ __device__ double* end(){return v+n;}
    __host__ __device__ const double* begin()const{return v;}
    __host__ __device__ const double* end()const{return v+n;}
  };
  using V=Array<3>;
  using M=Array<9>;
  struct Edge{unsigned j;V g;};
  struct CRWeight{V R;double p,s;};
  struct KVWeight{V q;double norm;};
};
class UserGDissCache:public UserGDissTypes{
public:
  bool TwoD,CR,KV;
  unsigned N;
  double H;
  double MaxCp=0;

  std::vector<std::vector<Edge>> Rows;
  std::vector<std::vector<CRWeight>> Weights;
  std::vector<std::vector<KVWeight>> ViscWeights;

  std::vector<M> Grad,Pv;
  std::vector<double> Rho,Cp,Eta0,SumG,NormSumG,Q;
  std::vector<unsigned char> ReuseF;

  static V Sub(const V&,const V&);
  static V Scale(V,double);
  static double Dot(const V&,const V&);
  static V Mul(const M&,const V&);
  static void Outer(M&,const V&,const V&);
  static M Inverse(M);
  static M Matrix(const tmatrix3f&);

  V Position(const tdouble3&)const;
  V Velocity(const tfloat4&)const;

public:
  UserGDissCache(bool twoD,unsigned n,const tdouble3* x0,
      const double* volume,const double* rho,const double* young,
      const double* poisson,const unsigned* body,const unsigned* count,
      const unsigned* const* pairs,const tmatrix3f* correction,
      bool mode5,const std::function<float(float)>& fac,
      double h,bool useCR,bool useKV,double factor);

};
UserGDissCache::V UserGDissCache::Sub(const V& a,const V& b){
  return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};
}

UserGDissCache::V UserGDissCache::Scale(V a,double s){
  for(double& q:a)q*=s;
  return a;
}

double UserGDissCache::Dot(const V& a,const V& b){
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

UserGDissCache::V UserGDissCache::Mul(const M& a,const V& b){
  V c={{0,0,0}};

  for(unsigned k=0;k<3;k++)
    for(unsigned l=0;l<3;l++)
      c[k]+=a[3*k+l]*b[l];

  return c;
}

void UserGDissCache::Outer(M& a,const V& b,const V& c){
  for(unsigned k=0;k<3;k++)
    for(unsigned l=0;l<3;l++)
      a[3*k+l]+=b[k]*c[l];
}

UserGDissCache::M UserGDissCache::Inverse(M a){
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

UserGDissCache::V UserGDissCache::Position(const tdouble3& a)const{
  return {{a.x,TwoD?0.:a.y,a.z}};
}

UserGDissCache::V UserGDissCache::Velocity(const tfloat4& a)const{
  return {{a.x,TwoD?0.:double(a.y),a.z}};
}

UserGDissCache::M UserGDissCache::Matrix(const tmatrix3f& a){
  return {{a.a11,a.a12,a.a13,a.a21,a.a22,a.a23,a.a31,a.a32,a.a33}};
}

UserGDissCache::UserGDissCache(
    bool twoD,unsigned n,const tdouble3* x0,
    const double* volume,const double* rho,const double* young,
    const double* poisson,const unsigned* body,const unsigned* count,
    const unsigned* const* pairs,const tmatrix3f* correction,
    bool mode5,const std::function<float(float)>& fac,
    double h,bool useCR,bool useKV,double factor)
  :TwoD(twoD),CR(useCR),KV(useKV),
   N(n),H(h),Rows(n),Weights(useCR?n:0),ViscWeights(useKV?n:0),
   Grad(useCR?n:0),Pv(useKV?n:0),
   Rho(rho,rho+n),Cp(useCR?n:0),
   Eta0(useKV?n:0),SumG(useKV?n:0),NormSumG(useKV?n:0),
   Q(useKV?n:0),ReuseF(n,0)
{
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

    if(CR){
      Cp[i]=std::sqrt(young[i]*(1-poisson[i])
          /((1+poisson[i])*(1-2*poisson[i])*Rho[i]));

      cs[i]=std::sqrt(
          young[i]/(2*(1+poisson[i])*Rho[i]));

      if(!std::isfinite(Rho[i]*Cp[i])
          || !std::isfinite(Rho[i]*cs[i]))
        throw std::runtime_error(
            "Dissipation: invalid wave speed.");

      MaxCp=std::max(MaxCp,Cp[i]);
    }

    if(KV){
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

// Precision is selected once per launch, never inside a neighbour loop.
// Reference construction remains identical to CPU; only the selected GPU cache
// is allocated. Global positions are always subtracted before conversion.
template<unsigned mode> class UserGDissPolicy{
public:
  using R=typename std::conditional<mode==0,float,double>::type;
  using B=typename std::conditional<mode==2,double,float>::type;
  template<unsigned n,class T=R> struct Array{
    T v[n];
    __host__ __device__ T& operator[](unsigned i){return v[i];}
    __host__ __device__ const T& operator[](unsigned i)const{return v[i];}
    __host__ __device__ T* begin(){return v;}
    __host__ __device__ T* end(){return v+n;}
    __host__ __device__ const T* begin()const{return v;}
    __host__ __device__ const T* end()const{return v+n;}
  };
  using V=Array<3>;using M=Array<9>;using BM=Array<9,B>;
  struct Edge{unsigned j;V g;};
  struct CRWeight{V R;B p,s;};
  struct KVWeight{Array<3,B> q;B norm;};
};
// A warp reads the same neighbour ordinal from adjacent rows. Interleave
// rows in groups of 32 without changing each particle's summation order.
template<class T> struct UserGDissSpan{
  T* p;unsigned n,stride;
  struct Iterator{
    const T* p;unsigned k,stride;
    __device__ const T& operator*()const{return p[size_t(k)*stride];}
    __device__ Iterator& operator++(){++k;return *this;}
    __device__ bool operator!=(const Iterator& b)const{return k!=b.k;}
  };
  __device__ Iterator begin()const{return {p,0,stride};}
  __device__ Iterator end()const{return {p,n,stride};}
  __device__ unsigned size()const{return n;}
  __device__ const T& operator[](unsigned i)const{return p[size_t(i)*stride];}
};
struct UserGDissPositions{
  const double2* xy;const double* z;
  __device__ double3 operator[](unsigned i)const{return make_double3(xy[i].x,xy[i].y,z[i]);}
};
template<unsigned mode> class UserGDissDevice{
public:
  using Policy=UserGDissPolicy<mode>;
  using R=typename Policy::R;using B=typename Policy::B;
  using V=typename Policy::V;using M=typename Policy::M;
  using BM=typename Policy::BM;
  using Edge=typename Policy::Edge;using CRWeight=typename Policy::CRWeight;
  using KVWeight=typename Policy::KVWeight;
  using Positions=UserGDissPositions;
  template<class T> using Span=UserGDissSpan<T>;
  __device__ static R Sqrt(R x){return mode==0?R(sqrtf(float(x))):R(sqrt(double(x)));}
  __device__ static R Abs(R x){return mode==0?R(fabsf(float(x))):R(fabs(double(x)));}
  __device__ static R Max(R x,R y){return mode==0?R(fmaxf(float(x),float(y))):R(fmax(double(x),double(y)));}
  __device__ static R Min(R x,R y){return mode==0?R(fminf(float(x),float(y))):R(fmin(double(x),double(y)));}
  unsigned N;B H,MaxCp;
  Span<Edge>* Rows;Span<CRWeight>* Weights;Span<KVWeight>* ViscWeights;
  M *Grad;BM *Pv;
  B *Rho,*Eta0,*SumG,*NormSumG,*Q;
  unsigned char* ReuseF;
  __device__ static V Scale(V a,R s){for(R& q:a)q*=s;return a;}
  __device__ static R Dot(const V& a,const V& b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
  __device__ static M Matrix(const tmatrix3f& a){return {{a.a11,a.a12,a.a13,a.a21,a.a22,a.a23,a.a31,a.a32,a.a33}};}
__device__ bool BuildKV(unsigned i,const M& F,const M& L){
  const R J=F[0]*(F[4]*F[8]-F[5]*F[7])
      -F[1]*(F[3]*F[8]-F[5]*F[6])+F[2]*(F[3]*F[7]-F[4]*F[6]);
  if(!(J>0) || !isfinite(J))return false;
  const R eta=mode==1?R(B(Eta0[i])/B(J)):R(Eta0[i])/J;
  M S={{0}},P={{0}};
  for(unsigned a=0;a<3;a++)for(unsigned b=0;b<3;b++)
    for(unsigned k=0;k<3;k++)
      S[3*a+b]+=R(.5)*eta*(F[3*k+a]*L[3*k+b]+L[3*k+a]*F[3*k+b]);
  // Symmetric strain-rate cancellation above is protected. Multiplying the
  // resulting stress and evaluating positive norm/time-step bounds is regular
  // arithmetic: mode 1 uses float here, as does its stored viscous stress.
  if(mode==1){
    BM pf={{0}};
    for(unsigned a=0;a<3;a++)for(unsigned b=0;b<3;b++)
      for(unsigned k=0;k<3;k++)pf[3*a+b]+=B(F[3*a+k])*B(S[3*k+b]);
    for(unsigned k=0;k<9;k++)P[k]=R(pf[k]);
    B n1=0,ni=0,n2=0;
    for(unsigned a=0;a<3;a++){
      B row=0,col=0;
      for(unsigned b=0;b<3;b++){
        const B f=B(F[3*a+b]);row+=fabsf(f);col+=fabsf(B(F[3*b+a]));n2+=f*f;
      }
      ni=fmaxf(ni,row);n1=fmaxf(n1,col);
    }
    Q[i]=B(eta)*fminf(n2,n1*ni)*(SumG[i]+NormSumG[i]);
  }else{
    for(unsigned a=0;a<3;a++)for(unsigned b=0;b<3;b++)
      for(unsigned k=0;k<3;k++)P[3*a+b]+=F[3*a+k]*S[3*k+b];
    R n1=0,ni=0,n2=0;
    for(unsigned a=0;a<3;a++){
      R row=0,col=0;
      for(unsigned b=0;b<3;b++){
        row+=Abs(F[3*a+b]);col+=Abs(F[3*b+a]);n2+=F[3*a+b]*F[3*a+b];
      }
      ni=Max(ni,row);n1=Max(n1,col);
    }
    Q[i]=eta*Min(n2,n1*ni)*(SumG[i]+NormSumG[i]);
  }
  if(!isfinite(Q[i]))return false;
  for(R q:P)if(!isfinite(q))return false;
  for(unsigned k=0;k<9;k++)Pv[i][k]=B(P[k]);
  return true;
}

template<bool twoD,bool cr,bool kv>
__device__ bool PrepareT(
    unsigned i,Positions x,const float4* velocity,
    const unsigned* rid,const M& existingF)
{
  if(kv){
    Pv[i]=BM{{0}};
    Q[i]=0;
  }

  M L={{0}},F={{0}};
  const unsigned p=rid[i];

  const V vi={{
    velocity[p].x,
    twoD?R(0):R(velocity[p].y),
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
        twoD?R(0):vj.y-vi[1],
        vj.z-vi[2]
      }};

      const V dx={{
        R(xj.x-xi.x),
        twoD?R(0):R(xj.y-xi.y),
        R(xj.z-xi.z)
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
        twoD?R(0):vj.y-vi[1],
        vj.z-vi[2]
      }};

      for(unsigned a=0;a<3;a++)if(!twoD || a!=1)
        for(unsigned b=0;b<3;b++)if(!twoD || b!=1)
          L[3*a+b]+=dv[a]*e.g[b];
    }
  }

  for(R q:L)if(!isfinite(q))return false;

  if(cr)Grad[i]=L;

  if(!kv || Eta0[i]==0 || SumG[i]==0)return true;

  if(!rebuild)F=existingF;

  return BuildKV(i,F,L);
}

template<bool twoD,bool cr,bool kv>
__device__ bool AddParticleT(
    unsigned i,Positions x,const float4* velocity,
    const unsigned* rid,V& acceleration,B& rate)const
{
  V a={{0}},av={{0}};
  double3 xi=make_double3(0,0,0);V vi={{0}};
  M Li={{0}};BM Pi={{0}};

  if(cr){
    const unsigned p=rid[i];

    xi=x[p];

    vi={{
      velocity[p].x,
      twoD?R(0):R(velocity[p].y),
      velocity[p].z
    }};

    Li=Grad[i];
  }

  if(kv)Pi=Pv[i];

  const auto& row=Rows[i];

  for(unsigned k=0;k<row.size();k++){
    const auto& e=row[k];
    const unsigned j=e.j;

    if(cr){
      const auto& w=Weights[i][k];
      const auto& Lj=Grad[j];
      const unsigned p=rid[j];

      const V dv={{
        velocity[p].x-vi[0],
        twoD?R(0):velocity[p].y-vi[1],
        velocity[p].z-vi[2]
      }};

      V jump=dv;

      for(unsigned a=0;a<3;a++)if(!twoD || a!=1){
        R gi=0,gj=0;

        for(unsigned b=0;b<3;b++)if(!twoD || b!=1){
          gi+=Li[3*a+b]*w.R[b];
          gj+=Lj[3*a+b]*w.R[b];
        }

        jump[a]-=R(.5)*(gi+gj);
      }

      const V r={{
        R(x[p].x-xi.x),
        twoD?R(0):R(x[p].y-xi.y),
        R(x[p].z-xi.z)
      }};

      if(mode==1){
        // n=r/|r|: (jump.n)*n == (jump.r)*r/(r.r).
        // Keep the cancellation-prone dot product and its sign in double;
        // normalization is positive scaling only. No double sqrt/division
        // is needed in this neighbour loop, and no empirical threshold is used.
        const B rx=B(r[0]),ry=B(r[1]),rz=B(r[2]);
        // Scale before conversion of the small projection: this avoids
        // overflow/underflow of jump.r and r.r in float for different units.
        const B scale=fmaxf(fabsf(rx),fmaxf(fabsf(ry),fabsf(rz)));
        if(!(scale>0) || !isfinite(scale))return false;
        const B invscale=B(1)/scale;
        const B sx=rx*invscale,sy=ry*invscale,sz=rz*invscale;
        const B rr=sx*sx+sy*sy+sz*sz;
        const R projection=Dot(jump,r)*R(invscale);
        if(!(rr>0) || !isfinite(rr) || !isfinite(projection))return false;
        if(projection<0){
          const B longitudinal=(w.p-w.s)*(B(projection)/rr);
          for(unsigned t=0;t<3;t++)if(!twoD || t!=1)
            a[t]+=R(w.s*B(jump[t])+longitudinal*(B(r[t])*invscale));
        }
      }else{
      const R rr=Dot(r,r);

      if(!(rr>0) || !isfinite(rr))return false;

      const V normal=Scale(r,R(1)/Sqrt(rr));
      const R un=Dot(jump,normal);

      if(!isfinite(un))return false;

      if(un<0)
        for(unsigned t=0;t<3;t++)if(!twoD || t!=1)
          a[t]+=R(w.s*B(jump[t])+(w.p-w.s)*B(un)*B(normal[t]));
      }
    }

    if(kv){
      const auto& w=ViscWeights[i][k];
      const auto& Pj=Pv[j];

      for(unsigned a=0;a<3;a++)if(!twoD || a!=1){
        B ai=0,aj=0;

        for(unsigned b=0;b<3;b++)if(!twoD || b!=1){
          ai+=Pi[3*a+b]*B(e.g[b]);
          aj+=Pj[3*a+b]*w.q[b];
        }

        av[a]+=ai+aj;
      }

      rate+=Q[j]*w.norm;
    }
  }

  if(kv){
    const R invrho=mode==1?R(B(1)/Rho[i]):R(1)/Rho[i];

    for(unsigned t=0;t<3;t++)
      a[t]+=av[t]*invrho;
  }

  for(R q:a)if(!isfinite(q))return false;

  acceleration[0]+=a[0];
  acceleration[1]+=a[1];
  acceleration[2]+=a[2];

  return true;
}

// Called from a compile-time-specialized force pass, not via per-particle
// runtime dispatch. Each particle writes only its own acceleration.
template<bool twoD,bool cr,bool kv>
__device__ bool Accumulate(
    unsigned i,Positions x,const float4* velocity,
    const unsigned* rid,float3* ace,R& speed)const
{
  V da={{0}};
  B rate=kv?Q[i]*NormSumG[i]:0;
  if(!AddParticleT<twoD,cr,kv>(i,x,velocity,rid,da,rate))return false;

  const R limit=FLT_MAX;
  if(!isfinite(da[0]) || Abs(da[0])>limit
      || !isfinite(da[1]) || Abs(da[1])>limit
      || !isfinite(da[2]) || Abs(da[2])>limit)return false;

  if(kv){
    const R value=H*rate/Rho[i];
    if(!isfinite(value) || value<0 || value>limit)return false;
    speed=Max(speed,value);
  }

  const unsigned p=rid[i];
  ace[p].x+=float(da[0]);
  ace[p].y+=float(da[1]);
  ace[p].z+=float(da[2]);
  return true;
}

};
template<bool twoD,bool cr,bool kv,unsigned mode>
__global__ void UserGDissPrepare(UserGDissDevice<mode> d,UserGDissPositions x,
    const float4* v,const unsigned* rid,const tmatrix3f* f,const void* state){
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.N)return;
  typename UserGDissDevice<mode>::M F={{0}};
  if(kv){
    if(mode==0)F=UserGDissDevice<mode>::Matrix(f[i]);
    else{
      for(unsigned k=0;k<9;k++)F[k]=mode==2
        ?static_cast<const UserGFlexMatrix<double>*>(state)[i].v[k]
        :double(static_cast<const UserGFlexMatrix<float>*>(state)[i].v[k]);
      F[0]+=1;F[4]+=1;F[8]+=1;
    }
  }
  if(!d.template PrepareT<twoD,cr,kv>(i,x,v,rid,F))asm("trap;");
}
template<bool twoD,bool cr,bool kv,unsigned mode>
__global__ void UserGDissForces(UserGDissDevice<mode> d,UserGDissPositions x,
    const float4* v,const typecode* code,const unsigned* rid,float3* ace,float* dt){
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d.N)return;
  typename UserGDissDevice<mode>::R speed=cr?d.MaxCp:0;
  if(CODE_IsFlexStrucFlex(code[rid[i]])){
    if(!d.template Accumulate<twoD,cr,kv>(i,x,v,rid,ace,speed))asm("trap;");
  }
  dt[i]=fmaxf(dt[i],float(speed));
}
class UserGFlexDissipation::Impl{
public:
  bool TwoD,CR,KV;unsigned N,Mode;
  UserGDissDevice<0> D0={};UserGDissDevice<1> D1={};UserGDissDevice<2> D2={};
  std::vector<void*> Memory;
  size_t Bytes=0;
  ~Impl(){for(void* p:Memory)cudaFree(p);}
  static void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
  template<class T> T* Copy(const std::vector<T>& v){
    if(v.empty())return nullptr;
    T* p=nullptr;Check(cudaMalloc((void**)&p,v.size()*sizeof(T)));
    Memory.push_back(p);Bytes+=v.size()*sizeof(T);
    Check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;
  }
  template<class T> UserGDissSpan<T>* Flatten(const std::vector<std::vector<T>>& rows){
    if(rows.empty())return nullptr;
    std::vector<T> flat;
    std::vector<size_t> offsets(rows.size());
    std::vector<unsigned> strides(rows.size(),1);
    for(size_t first=0;first<rows.size();first+=32){
      const size_t end=std::min(first+32,rows.size());
      size_t count=0,total=0;
      for(size_t i=first;i<end;i++){
        count=std::max(count,rows[i].size());total+=rows[i].size();
      }
      const size_t width=end-first;
      // Bound padding to at most the live entries. Irregular groups retain
      // compact rows; this is a storage bound, not a physics/user parameter.
      if(width*count<=2*total){
        const size_t base=flat.size();flat.resize(base+width*count);
        for(size_t i=first;i<end;i++){
          offsets[i]=base+i-first;strides[i]=unsigned(width);
          for(size_t k=0;k<rows[i].size();k++)flat[base+k*width+i-first]=rows[i][k];
        }
      }else{
        for(size_t i=first;i<end;i++){
          offsets[i]=flat.size();flat.insert(flat.end(),rows[i].begin(),rows[i].end());
        }
      }
    }
    T* p=Copy(flat);std::vector<UserGDissSpan<T>> spans;
    for(size_t i=0;i<rows.size();i++)
      spans.push_back({rows[i].empty()?nullptr:p+offsets[i],unsigned(rows[i].size()),strides[i]});
    return Copy(spans);
  }
  template<unsigned mode> void Init(UserGDissDevice<mode>& d,const UserGDissCache& c){
    using P=UserGDissPolicy<mode>;using R=typename P::R;using B=typename P::B;
    d.N=c.N;d.H=R(c.H);d.MaxCp=R(c.MaxCp);
    std::vector<std::vector<typename P::Edge>> rows(c.Rows.size());
    std::vector<std::vector<typename P::CRWeight>> weights(c.Weights.size());
    std::vector<std::vector<typename P::KVWeight>> visc(c.ViscWeights.size());
    for(unsigned i=0;i<rows.size();i++)for(const auto& e:c.Rows[i]){
      typename P::Edge a={};a.j=e.j;for(unsigned k=0;k<3;k++)a.g[k]=R(e.g[k]);rows[i].push_back(a);
    }
    for(unsigned i=0;i<weights.size();i++)for(const auto& e:c.Weights[i]){
      typename P::CRWeight a={};a.p=e.p;a.s=e.s;for(unsigned k=0;k<3;k++)a.R[k]=R(e.R[k]);weights[i].push_back(a);
    }
    for(unsigned i=0;i<visc.size();i++)for(const auto& e:c.ViscWeights[i]){
      typename P::KVWeight a={};a.norm=R(e.norm);for(unsigned k=0;k<3;k++)a.q[k]=e.q[k];visc[i].push_back(a);
    }
    d.Rows=Flatten(rows);d.Weights=Flatten(weights);d.ViscWeights=Flatten(visc);
    d.Grad=Copy(std::vector<typename P::M>(c.Grad.size()));
    d.Pv=Copy(std::vector<typename P::BM>(c.Pv.size()));
    d.Rho=Copy(std::vector<B>(c.Rho.begin(),c.Rho.end()));
    d.Eta0=Copy(std::vector<B>(c.Eta0.begin(),c.Eta0.end()));
    d.SumG=Copy(std::vector<B>(c.SumG.begin(),c.SumG.end()));
    d.NormSumG=Copy(std::vector<B>(c.NormSumG.begin(),c.NormSumG.end()));
    d.Q=Copy(std::vector<B>(c.Q.begin(),c.Q.end()));d.ReuseF=Copy(c.ReuseF);
  }
  explicit Impl(const UserGDissCache& c,unsigned mode):TwoD(c.TwoD),CR(c.CR),KV(c.KV),N(c.N),Mode(mode){
    if(mode>2 || c.MaxCp>FLT_MAX)throw std::runtime_error("Invalid dissipation precision or wave speed.");
    try{
      if(mode==0)Init(D0,c);else if(mode==1)Init(D1,c);else Init(D2,c);
    }catch(...){for(void* p:Memory)cudaFree(p);throw;}
  }
  template<bool twoD,bool cr,bool kv,unsigned mode> void Launch(UserGDissDevice<mode> d,
      UserGDissPositions x,const float4* v,const typecode* code,const unsigned* rid,
      const tmatrix3f* f,const void* h,float3* ace,float* speed){
    const unsigned blocks=(N+127)/128;
    UserGDissPrepare<twoD,cr,kv,mode><<<blocks,128>>>(d,x,v,rid,f,h);
    UserGDissForces<twoD,cr,kv,mode><<<blocks,128>>>(d,x,v,code,rid,ace,speed);
  }
  template<bool twoD,bool cr,bool kv> void Run(UserGDissPositions x,
      const float4* v,const typecode* code,const unsigned* rid,const tmatrix3f* f,
      unsigned mode,UserGFlexPrecision* precision,float3* ace,float* speed){
    if(!N)return;
    if(mode==0)Launch<twoD,cr,kv>(D0,x,v,code,rid,f,nullptr,ace,speed);
    else if(mode==1)Launch<twoD,cr,kv>(D1,x,v,code,rid,f,static_cast<const UserGFlexState<float>*>(precision->State(1,N))->H,ace,speed);
    else Launch<twoD,cr,kv>(D2,x,v,code,rid,f,static_cast<const UserGFlexState<double>*>(precision->State(2,N))->H,ace,speed);
    Check(cudaGetLastError());
  }
};
UserGFlexDissipation::UserGFlexDissipation(bool twoD,unsigned n,const tdouble3* x0,
      const double* volume,const double* rho,const double* young,
      const double* poisson,const unsigned* body,const unsigned* count,
      const unsigned* const* pairs,const tmatrix3f* correction,
      bool mode5,const std::function<float(float)>& fac,
      double h,bool useCR,bool useKV,double factor,unsigned mode)
 :Data(nullptr){
  UserGDissCache cache(twoD,n,x0,volume,rho,young,poisson,body,count,pairs,
      correction,mode5,fac,h,useCR,useKV,factor);
  Data=new Impl(cache,mode);
}
UserGFlexDissipation::~UserGFlexDissipation(){delete Data;}
size_t UserGFlexDissipation::GetAllocMemoryGpu()const{return Data?Data->Bytes:0;}
void UserGFlexDissipation::Apply(unsigned n,const double2* xy,const double* z,
    const float4* velocity,const typecode* code,const unsigned* rid,const tmatrix3f* f,
    unsigned mode,UserGFlexPrecision* precision,float3* ace,float* speed){
  if(!Data || n!=Data->N || mode!=Data->Mode)throw std::runtime_error("Invalid GPU dissipation state.");
  const UserGDissPositions x={xy,z};
  if(Data->TwoD){
    if(Data->CR && Data->KV)Data->Run<true,true,true>(x,velocity,code,rid,f,mode,precision,ace,speed);
    else if(Data->CR)Data->Run<true,true,false>(x,velocity,code,rid,f,mode,precision,ace,speed);
    else if(Data->KV)Data->Run<true,false,true>(x,velocity,code,rid,f,mode,precision,ace,speed);
  }else{
    if(Data->CR && Data->KV)Data->Run<false,true,true>(x,velocity,code,rid,f,mode,precision,ace,speed);
    else if(Data->CR)Data->Run<false,true,false>(x,velocity,code,rid,f,mode,precision,ace,speed);
    else if(Data->KV)Data->Run<false,false,true>(x,velocity,code,rid,f,mode,precision,ace,speed);
  }
}

void JSphGpuSingle::InitFlexDissipation(bool beforeCorrection){
  const bool inherit=User->UseFlexRestartRestorePairs || User->UseFlexRestartGradientCorrection;
  if(beforeCorrection!=inherit)return;
  delete FlexDissipation;FlexDissipation=nullptr;
  const bool useCR=User->ElasticHighFrequencyDissipation;
  const bool useKV=User->ElasticViscousDissipation && User->ElasticViscousFactor>0;
  if(!useCR && !useKV)return;
  const unsigned np=CaseNflexstruc;
  std::vector<unsigned> rid(np),ids(Npb),counts(np);std::vector<typecode> code(Npb);
  std::vector<StFlexStrucData> mat(FlexStrucCount);std::vector<tmatrix3f> correction(np);
  std::vector<double2> xy(Npb);std::vector<double> z(Npb);
  auto read=[](void* a,const void* b,size_t bytes){
    if(bytes){const auto e=cudaMemcpy(a,b,bytes,cudaMemcpyDeviceToHost);
      if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
  };
  read(rid.data(),FlexStrucRidpg,np*sizeof(unsigned));read(ids.data(),Idp_g->cptr(),Npb*sizeof(unsigned));
  read(counts.data(),NumPairsg,np*sizeof(unsigned));read(code.data(),Code_g->cptr(),Npb*sizeof(typecode));
  read(mat.data(),FlexStrucDatag,mat.size()*sizeof(StFlexStrucData));
  if(!inherit)read(correction.data(),KerCorrg,np*sizeof(tmatrix3f));
  read(xy.data(),Posxy_g->cptr(),Npb*sizeof(double2));read(z.data(),Posz_g->cptr(),Npb*sizeof(double));
  size_t total=0;for(unsigned n:counts)total+=n;
  std::vector<unsigned> flat(total);read(flat.data(),PairIdxBufferg,total*sizeof(unsigned));
  std::vector<const unsigned*> pairs(np);std::vector<tdouble3> x0(np);
  std::vector<double> volume(np),rho(np),young(np),poisson(np);std::vector<unsigned> body(np);
  const auto& state=User->FlexRestartState;const bool restart=User->IsFlexRestart();
  if(restart && (!state.HasRestartStateLayout() || state.Idp.size()!=np))
    Run_Exceptioon("Dissipation: incomplete SPH reference state.");
  size_t offset=0;
  for(unsigned i=0;i<np;i++){
    const unsigned p=rid[i];body[i]=CODE_GetIbodyFlexStruc(code[p]);
    if(body[i]>=mat.size())Run_Exceptioon("Dissipation: invalid body.");
    const auto& m=mat[body[i]];
    if(restart && state.Idp[i]!=ids[p])Run_Exceptioon("Dissipation: inconsistent particle identity.");
    volume[i]=restart?state.ReferenceVolume[i]:double(m.vol0);
    const double mass=restart?double(state.ParticleMass[i]):volume[i]*double(m.rho0);
    rho[i]=mass/volume[i];young[i]=m.youngmod;poisson[i]=m.poissratio;
    x0[i]=TDouble3(xy[p].x,xy[p].y,z[p]);pairs[i]=flat.data()+offset;offset+=counts[i];
  }
  if(TKernel!=KERNEL_Wendland && TKernel!=KERNEL_Cubic)Run_Exceptioon("Dissipation: unsupported kernel.");
  const auto fac=[&](float rr)->float{return TKernel==KERNEL_Wendland
    ?fsph::GetKernel_Fac<KERNEL_Wendland>(CSP,rr):fsph::GetKernel_Fac<KERNEL_Cubic>(CSP,rr);};
  if(useKV && (User->UseFlexRestartRestorePairs || User->UseFlexRestartGradientCorrection))
    Run_Exceptioon("KV cannot inherit reference kernels or pairs.");
  // Transported-pair cR rebuilds its local gradient as on CPU.

  FlexDissipation=new UserGFlexDissipation(Simulate2D,np,x0.data(),volume.data(),rho.data(),
    young.data(),poisson.data(),body.data(),counts.data(),pairs.data(),
    inherit?nullptr:correction.data(),false,fac,KernelH,useCR,useKV,User->ElasticViscousFactor,User->GPUFlexPrecisionMode);
}
