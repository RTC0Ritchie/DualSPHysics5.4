#ifndef _UserGFlexPrecisionState_
#define _UserGFlexPrecisionState_

// All indices below are elastic indices, independent of the particle sort order.
// State is canonical; defgrad/kercorr in the solver are one-way float mirrors.
template<class R> struct UserGFlexMatrix{ R v[9]; };
template<class R> struct UserGFlexVector{ R v[3]; };
template<class R> struct UserGFlexState{
  double3 *Reference,*Displacement;
  UserGFlexVector<R>* PairReference;
  R* PairNorm;
  UserGFlexVector<R>* PairGradient; // Only extended mode allocates this cache.
  UserGFlexMatrix<R> *Correction,*H,*PK,*PB;
  unsigned* Error;
};
#endif
