#ifndef USERFLEXDISSIPATION_H
#define USERFLEXDISSIPATION_H

#include "DualSphDef.h"
#include <array>
#include <vector>
#include <functional>

class JFlexDissipation{
  using V=std::array<double,3>;
  using M=std::array<double,9>;

  struct Edge{
    unsigned j;
    V g;
  };

  struct CRWeight{
    V R;
    double p,s;
  };

  struct KVWeight{
    V q;
    double norm;
  };

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

  bool BuildKV(unsigned i,const M& F,const M& L);

  template<bool twoD,bool cr,bool kv>
  bool PrepareT(unsigned i,const tdouble3* x,const tfloat4* velocity,
      const unsigned* rid,const tmatrix3f& F);

  template<bool twoD,bool cr,bool kv>
  bool AddParticleT(unsigned i,const tdouble3* x,const tfloat4* velocity,
      const unsigned* rid,tdouble3& acceleration,double& rate)const;

  template<bool twoD,bool cr,bool kv>
  bool Accumulate(unsigned i,const tdouble3* x,const tfloat4* velocity,
      const unsigned* rid,tfloat3* ace,double& speed)const;

  template<bool twoD,bool cr,bool kv>
  double ApplyT(const tdouble3* x,const tfloat4* velocity,
      const typecode* code,const unsigned* rid,
      const tmatrix3f* defgrad,tfloat3* ace);

public:
  JFlexDissipation(bool twoD,unsigned n,const tdouble3* x0,
      const double* volume,const double* rho,const double* young,
      const double* poisson,const unsigned* body,const unsigned* count,
      const unsigned* const* pairs,const tmatrix3f* correction,
      bool mode5,const std::function<float(float)>& fac,
      double h,bool useCR,bool useKV,double factor);

  // Reference caches are constructed once. Apply updates only stage values.
  // defgrad uses flexible-local indices; rid maps to current global indices.
  // Returns the wave-speed equivalent for the existing dt3 constraint.
  double Apply(unsigned n,const tdouble3* x,const tfloat4* velocity,
      const typecode* code,const unsigned* rid,
      const tmatrix3f* defgrad,tfloat3* ace);
};

#endif
