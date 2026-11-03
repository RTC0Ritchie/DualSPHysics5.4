#ifndef _UserGInteractionForcesFlexStruc_
#define _UserGInteractionForcesFlexStruc_

#include "JSphGpu_ker.h"

// Stateless GPU elastic-force module. CUDA implementation stays in the .cu file.
class UserGInteractionForcesFlexStruc{
private:
  UserGInteractionForcesFlexStruc()=delete;

  template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2>
  static void Interaction_ForcesFlexStrucT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker,TpVisco tvisco>
  static void Interaction_ForcesFlexStruc_gt1(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void Interaction_ForcesFlexStruc_gt0(const StInterParmsFlexStrucg& tfs);

  // CPU counterpart: UserInteractionForcesFlexStruc.cpp.
  template<TpKernel tker,bool simulate2d>
  static void CalcFlexStrucKerCorrT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void CalcFlexStrucKerCorr_gt0(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker,bool simulate2d>
  static void CalcFlexStrucDefGradT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void CalcFlexStrucDefGrad_ct0(const StInterParmsFlexStrucg& tfs);

public:
  static void CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs);
  static void UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs);
  static void CteInteractionUp(const StCteInteraction* cte);
  static void Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs);
};

#endif
