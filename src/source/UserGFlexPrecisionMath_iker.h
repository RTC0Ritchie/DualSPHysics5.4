#ifndef _UserGFlexPrecisionMath_iker_
#define _UserGFlexPrecisionMath_iker_

// Geometry subtraction is ALWAYS double; the policy only selects where to narrow.
// No run-time precision branch occurs inside particle or neighbour loops.
template<class Scalar> class UserGFlexPrecisionMath{
public:
  using Real=Scalar;
  static constexpr bool OriginalCoordinates=false;
  using Matrix=UserGFlexMatrix<Real>;
  using Vector=UserGFlexVector<Real>;
  using State=UserGFlexState<Real>;

  template<TpKernel kernel> static __device__ double Fac64(double rr2){
    const double rad=sqrt(rr2),q=rad/double(CTE.kernelh);
    if(kernel==KERNEL_Wendland){
      const double a=1.0-0.5*q;
      return double(CTE.bwenh)*a*a*a;
    }
    if(rad>double(CTE.kernelh)){
      const double a=2.0-q;
      return double(CTE.cubic_c2)*a*a/rad;
    }
    // The limiting value at zero avoids 0/0 for a coincident pair.
    return rr2? (double(CTE.cubic_c1)*q+double(CTE.cubic_d1)*q*q)/rad
              : double(CTE.cubic_c1)/double(CTE.kernelh);
  }
  template<TpKernel kernel> static __device__ double Wab64(double rr2){
    const double rad=sqrt(rr2),q=rad/double(CTE.kernelh);
    if(kernel==KERNEL_Wendland){
      const double a=1.0-0.5*q;
      return double(CTE.awen)*(2.0*q+1.0)*a*a*a*a;
    }
    if(rad>double(CTE.kernelh)){
      const double a=2.0-q;return double(CTE.cubic_a24)*a*a*a;
    }
    return double(CTE.cubic_a2)*(1.0+(0.75*q-1.5)*q*q);
  }
  template<TpKernel kernel> static __device__ Vector Gradient(const State& s,size_t pair){
    if(sizeof(Real)==sizeof(double))return s.PairGradient[pair];
    const float fac=cufsph::GetKernel_Fac<kernel>(float(s.PairNorm[pair]));
    const Vector d=s.PairReference[pair];
    return {{Real(fac*float(d.v[0])),Real(fac*float(d.v[1])),Real(fac*float(d.v[2]))}};
  }
  static __device__ Matrix Multiply(const Matrix& a,const Matrix& b){
    Matrix c={};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)
      for(int k=0;k<3;k++)c.v[3*i+j]+=a.v[3*i+k]*b.v[3*k+j];
    return c;
  }
  static __device__ Matrix Stress(const Matrix& h,const tmatrix6f& material){
    // E = (H + H^T + H^T H)/2, exactly the full Green-Lagrange law.
    // Do not reconstruct H by subtracting I from the float F mirror.
    Real e[9];
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      Real v=Real(.5)*(h.v[3*i+j]+h.v[3*j+i]);
      for(int k=0;k<3;k++)v+=Real(.5)*h.v[3*k+i]*h.v[3*k+j];
      e[3*i+j]=v;
    }
    const Real ev[6]={e[0],e[4],e[8],2*e[5],2*e[6],2*e[1]};
    Real sv[6]={};const float* c=&material.a11;
    for(int i=0;i<6;i++)for(int j=0;j<6;j++)sv[i]+=Real(c[6*i+j])*ev[j];
    const Matrix stress={{sv[0],sv[5],sv[4],sv[5],sv[1],sv[3],sv[4],sv[3],sv[2]}};
    Matrix p=stress;
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)
      for(int k=0;k<3;k++)p.v[3*i+j]+=h.v[3*i+k]*stress.v[3*k+j];
    return p;
  }
  static __device__ Real DeterminantF(const Matrix& h){
    const Real a=1+h.v[0],b=h.v[1],c=h.v[2],d=h.v[3],e=1+h.v[4];
    const Real f=h.v[5],g=h.v[6],j=h.v[7],k=1+h.v[8];
    return a*(e*k-f*j)-b*(d*k-f*g)+c*(d*j-e*g);
  }
};
#endif
