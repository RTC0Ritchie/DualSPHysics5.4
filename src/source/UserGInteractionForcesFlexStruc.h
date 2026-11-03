#ifndef _UserGInteractionForcesFlexStruc_
#define _UserGInteractionForcesFlexStruc_

#include "JSphGpu_ker.h"

// Stable host API. Precision implementation remains in the CUDA module.
class UserGInteractionForcesFlexStruc{
private:
  UserGInteractionForcesFlexStruc()=delete;

public:
  static void CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs);
  static void UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs);
  static void CteInteractionUp(const StCteInteraction* cte);
  static void Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs);
};

#endif
