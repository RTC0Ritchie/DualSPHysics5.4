#ifndef USERFLEXDISSIPATION_H
#define USERFLEXDISSIPATION_H

#include "DualSphDef.h"

#include <array>
#include <vector>
#include <functional>

class JFlexRiemann{
  using V=std::array<double,3>;
  using M=std::array<double,9>;

  struct Edge{
    unsigned j;
    V R,g;
    double wp,ws;
  };

  bool TwoD;
  unsigned N;
  double MaxCp=0;

  std::vector<std::vector<Edge>> Rows;
  std::vector<M> Grad;
  std::vector<double> Cp,Cs;

  static V Sub(const V& a,const V& b);
  static V Scale(V a,double s);
  static double Dot(const V& a,const V& b);
  static V Mul(const M& a,const V& b);
  static void Outer(M& a,const V& b,const V& c);
  static M Inverse(M a);

  V Position(const tdouble3& a)const;
  V Velocity(const tfloat4& a)const;


public:
  JFlexRiemann(
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
      const std::function<double(double)>& dW);

  double GetMaxWaveSpeed()const;

  void Apply(
      unsigned n,
      const tdouble3* x,
      const tfloat4* velocity,
      const typecode* code,
      const unsigned* rid,
      tfloat3* ace);
};

#endif