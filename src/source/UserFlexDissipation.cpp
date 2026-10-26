#include "UserFlexDissipation.h"
#include "OmpDefs.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

//==============================================================================
// Vector and matrix operations.
//==============================================================================

JFlexRiemann::V JFlexRiemann::Sub(const V& a,const V& b){
  return {{a[0]-b[0],a[1]-b[1],a[2]-b[2]}};
}

JFlexRiemann::V JFlexRiemann::Scale(V a,double s){
  for(double& q:a)q*=s;
  return a;
}

double JFlexRiemann::Dot(const V& a,const V& b){
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

JFlexRiemann::V JFlexRiemann::Mul(const M& a,const V& b){
  V c={{0,0,0}};

  for(unsigned k=0;k<3;k++)
    for(unsigned l=0;l<3;l++)
      c[k]+=a[3*k+l]*b[l];

  return c;
}

void JFlexRiemann::Outer(M& a,const V& b,const V& c){
  for(unsigned k=0;k<3;k++)
    for(unsigned l=0;l<3;l++)
      a[3*k+l]+=b[k]*c[l];
}

JFlexRiemann::M JFlexRiemann::Inverse(M a){
  M b={{1,0,0,0,1,0,0,0,1}};

  double scale=0;

  for(double q:a){
    if(!std::isfinite(q))
      throw std::runtime_error("cR: invalid moment.");

    scale=std::max(scale,std::abs(q));
  }

  for(unsigned k=0;k<3;k++){
    unsigned p=k;

    for(unsigned j=k+1;j<3;j++)
      if(std::abs(a[3*j+k])>std::abs(a[3*p+k]))
        p=j;

    if(!(std::abs(a[3*p+k])>1e-12*scale))
      throw std::runtime_error(
          "cR: singular reference gradient.");

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

JFlexRiemann::V JFlexRiemann::Position(const tdouble3& a)const{
  return {{a.x,TwoD?0.:a.y,a.z}};
}

JFlexRiemann::V JFlexRiemann::Velocity(const tfloat4& a)const{
  return {{a.x,TwoD?0.:double(a.y),a.z}};
}

//==============================================================================
// Builds the fixed-reference cR operator.
//
// Reference arrays use flexible-local particle indices.
// dW(rr) returns dW/dR evaluated at R=sqrt(rr).
//
// Cached quantities:
//   R      = Xj - Xi
//   g      = V0j * Bi * Gij
//   weight = 0.5 * V0j * (-dW/dR)
//
// The constructor does not retain pointers to the supplied reference arrays.
//==============================================================================

JFlexRiemann::JFlexRiemann(
    bool twoD,
    unsigned n,
    const tdouble3* X,
    const double* volume,
    const double* rho,
    const double* young,
    const double* poisson,
    const unsigned* body,
    const unsigned* count,
    const unsigned* const* pairs,
    const std::function<double(double)>& dW)
  :TwoD(twoD),N(n),Rows(n),Grad(n),Cp(n),Cs(n)
{
  // Numerical wave-speed scales in the supplied SPH reference.
  for(unsigned i=0;i<N;i++){
    if(!(volume[i]>0 && rho[i]>0 && young[i]>0
        && poisson[i]>-1 && poisson[i]<.5)
        || !std::isfinite(volume[i])
        || !std::isfinite(rho[i])
        || !std::isfinite(young[i]))
      throw std::runtime_error(
          "cR: invalid reference material.");

    const double mu=young[i]/(2*(1+poisson[i]));
    const double la=young[i]*poisson[i]
        /((1+poisson[i])*(1-2*poisson[i]));

    Cp[i]=std::sqrt((la+2*mu)/rho[i]);
    Cs[i]=std::sqrt(mu/rho[i]);

    if(!(Cp[i]>0 && Cs[i]>0)
        || !std::isfinite(Cp[i]) || !std::isfinite(Cs[i])
        || !std::isfinite(rho[i]*Cp[i])
        || !std::isfinite(rho[i]*Cs[i]))
      throw std::runtime_error(
          "cR: invalid reference wave speed.");

    MaxCp=std::max(MaxCp,Cp[i]);
  }

  // Fixed reference gradient and impedance weights.
  for(unsigned i=0;i<N;i++){
    M moment={{0}};
    Rows[i].reserve(count[i]);

    for(unsigned p=0;p<count[i];p++){
      const unsigned j=pairs[i][p];

      if(j>=N || j==i)
        throw std::runtime_error("cR: invalid pair.");

      // Internal stabilization only.
      if(body[i]!=body[j])continue;

      const V R=Sub(Position(X[j]),Position(X[i]));
      const double rr=Dot(R,R);

      if(!(rr>0) || !std::isfinite(rr))
        throw std::runtime_error(
            "cR: invalid reference distance.");

      const double s=-dW(rr);

      if(!std::isfinite(s) || s<0)
        throw std::runtime_error(
            "cR: non-monotone reference kernel.");

      if(s==0)continue;

      // Gij = -W'(Rij) * Rij / |Rij|.
      // Initially g = V0j * Gij.
      const V g=Scale(R,volume[j]*s/std::sqrt(rr));
      Outer(moment,R,g);

      // Stable evaluation of a*b/(a+b).
      const auto impedance=[](double a,double b){
        const double lo=std::min(a,b),hi=std::max(a,b);
        return lo/(1+lo/hi);
      };

      const double wp=volume[j]*s/rho[i]
          *impedance(rho[i]*Cp[i],rho[j]*Cp[j]);
      const double ws=volume[j]*s/rho[i]
          *impedance(rho[i]*Cs[i],rho[j]*Cs[j]);

      if(!std::isfinite(wp) || !std::isfinite(ws))
        throw std::runtime_error(
            "cR: invalid pair impedance.");

      Rows[i].push_back({j,R,g,wp,ws});
    }

    if(Rows[i].empty())continue;

    // Embed the x-z moment matrix for a 2D calculation.
    if(TwoD)
      moment[4]=std::max(
          std::abs(moment[0]),std::abs(moment[8]));

    const M B=Inverse(moment);

    // Cache V0j * Bi * Gij.
    for(auto& e:Rows[i])
      e.g=Mul(B,e.g);
  }
}

//==============================================================================
// Returns the largest reference longitudinal wave speed.
//==============================================================================

double JFlexRiemann::GetMaxWaveSpeed()const{
  return MaxCp;
}

//==============================================================================
// Adds the paper cR acceleration to the existing acceleration array.
//
// Pass 1 updates material velocity gradients for the current force stage.
// Pass 2 reconstructs midpoint jumps and directly assembles cR acceleration.
//
// rid maps flexible-local indices to current global particle indices.
//==============================================================================

void JFlexRiemann::Apply(
    unsigned n,
    const tdouble3* x,
    const tfloat4* velocity,
    const typecode* code,
    const unsigned* rid,
    tfloat3* ace)
{
  if(n!=N)
    throw std::runtime_error(
        "cR: rebuild reference cache.");

  // Pass 1:
  //   L_i^X = sum_j (vj-vi) tensor (V0j * Bi * Gij).
  //
  // Include clamp particles because they can participate in reconstruction.
  #ifdef OMP_USE
  #pragma omp parallel for schedule(static)
  #endif
  for(int ii=0;ii<int(N);ii++){
    const unsigned i=unsigned(ii);
    M L={{0}};

    const V vi=Velocity(velocity[rid[i]]);

    for(const auto& e:Rows[i])
      Outer(
          L,
          Sub(Velocity(velocity[rid[e.j]]),vi),
          e.g);

    Grad[i]=L;
  }

  // Pass 2:
  //   jump = vj-vi - 0.5*(L_i^X + L_j^X)*Rij.
  //
  // Each thread writes only its target particle acceleration.
  int invalid=0;

  #ifdef OMP_USE
  #pragma omp parallel for schedule(static) reduction(+:invalid)
  #endif
  for(int ii=0;ii<int(N);ii++){
    const unsigned i=unsigned(ii);
    const unsigned p=rid[i];

    if(!CODE_IsFlexStrucFlex(code[p]))continue;

    const V vi=Velocity(velocity[p]);
    const V xi=Position(x[p]);
    V a={{0,0,0}};

    for(const auto& e:Rows[i]){
      const unsigned j=e.j;

      // Midpoint reconstruction in the reference configuration.
      V jump=Sub(Velocity(velocity[rid[j]]),vi);

      const V gi=Mul(Grad[i],e.R);
      const V gj=Mul(Grad[j],e.R);

      for(unsigned k=0;k<3;k++)
        jump[k]-=.5*(gi[k]+gj[k]);

      // Longitudinal and tangential projections use the current direction.
      const V r=Sub(Position(x[rid[j]]),xi);
      const double rr=Dot(r,r);

      if(!(rr>0) || !std::isfinite(rr)){
        invalid++;
        continue;
      }

      const V normal=Scale(r,1/std::sqrt(rr));
      const double un=Dot(jump,normal);

      if(!std::isfinite(un)){
        invalid++;
        continue;
      }

      // Paper limiter:
      //   beta=1 for a compressive reconstructed normal jump.
      //   beta=0 otherwise, including an exactly zero normal jump.
      if(un>=0)continue;

      // Cp * jump_normal + Cs * jump_tangential.
      for(unsigned k=0;k<3;k++)
        a[k]+=e.ws*jump[k]+(e.wp-e.ws)*un*normal[k];
    }

    if(!std::isfinite(a[0])
        || !std::isfinite(a[1])
        || !std::isfinite(a[2])){
      invalid++;
      continue;
    }

    // Already an acceleration: no additional density division.
    // Preserve all previously assembled force contributions.
    ace[p].x+=float(a[0]);
    ace[p].y+=float(a[1]);
    ace[p].z+=float(a[2]);
  }

  if(invalid)
    throw std::runtime_error(
        "cR: invalid current state.");
}