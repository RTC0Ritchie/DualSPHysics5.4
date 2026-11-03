#ifndef _UserGFlexAccess_iker_
#define _UserGFlexAccess_iker_

// Shared entry for future CPU-to-GPU features. Mode differences belong here,
// not in constitutive/boundary/dissipation formulas. All branches are compile-time.
template<class Math> class UserGFlexAccess{
public:
  using R=typename Math::Real;
  using Matrix=UserGFlexMatrix<R>;
  using Vector=UserGFlexVector<R>;
  using State=UserGFlexState<R>;
  static __device__ __forceinline__ Matrix H(const StInterParmsFlexStrucg& t,const State& s,unsigned i){
    if(!Math::OriginalCoordinates)return s.H[i];
    Matrix h;const tmatrix3f f=t.defgrad[i];
    for(int k=0;k<9;k++)h.v[k]=(&f.a11)[k]-(k%4==0?1.f:0.f);
    return h;
  }
  static __device__ __forceinline__ Matrix F(const StInterParmsFlexStrucg& t,const State& s,unsigned i){
    if(Math::OriginalCoordinates){
      Matrix f;const tmatrix3f v=t.defgrad[i];for(int k=0;k<9;k++)f.v[k]=(&v.a11)[k];return f;
    }
    Matrix f=s.H[i];f.v[0]+=1;f.v[4]+=1;f.v[8]+=1;return f;
  }
  static __device__ __forceinline__ Matrix Correction(const StInterParmsFlexStrucg& t,const State& s,unsigned i){
    if(!Math::OriginalCoordinates)return s.Correction[i];
    Matrix b;const tmatrix3f v=t.kercorr[i];for(int k=0;k<9;k++)b.v[k]=(&v.a11)[k];return b;
  }
  template<TpKernel kernel>
  static __device__ __forceinline__ Vector Gradient(const StInterParmsFlexStrucg& t,const State& s,unsigned i,unsigned k){
    if(!Math::OriginalCoordinates)
      return UserGFlexPrecisionMath<R>::template Gradient<kernel>(s,t.pairidx[i]+k-t.pairidx[0]);
    const float4 a=t.poscell0[i],b=t.poscell0[t.pairidx[i][k]];
    const float x=a.x-b.x+CTE.poscellsize*(PSCEL_GetfX(a.w)-PSCEL_GetfX(b.w));
    const float y=a.y-b.y+CTE.poscellsize*(PSCEL_GetfY(a.w)-PSCEL_GetfY(b.w));
    const float z=a.z-b.z+CTE.poscellsize*(PSCEL_GetfZ(a.w)-PSCEL_GetfZ(b.w));
    const float fac=cufsph::GetKernel_Fac<kernel>(x*x+y*y+z*z);
    return {{R(fac*x),R(fac*y),R(fac*z)}};
  }
};
#endif
