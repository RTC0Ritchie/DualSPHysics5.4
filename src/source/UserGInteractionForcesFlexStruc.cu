//HEAD_DSPH
/*
 <DUALSPHYSICS>  Copyright (c) 2025 by Dr Jose M. Dominguez et al. (see http://dual.sphysics.org/index.php/developers/). 

 EPHYSLAB Environmental Physics Laboratory, Universidade de Vigo, Ourense, Spain.
 School of Mechanical, Aerospace and Civil Engineering, University of Manchester, Manchester, U.K.

 This file is part of DualSPHysics. 

 DualSPHysics is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License 
 as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version.
 
 DualSPHysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details. 

 You should have received a copy of the GNU Lesser General Public License along with DualSPHysics. If not, see <http://www.gnu.org/licenses/>. 
*/

#include "UserGInteractionForcesFlexStruc.h"
#include "UserGParameters.h"
#include "UserGFlexTypes.h"
#include "UserGFlexContact.h"
#include "UserGFlexCte.h"
#include "JSphGpu_ker.h"
#include "Functions.h"
#include "FunctionsCuda.h"
#include <cfloat>
#include <vector>
#include <stdexcept>
#include "UserGFlexPrecision.h"
#include "UserGFlexPrecisionState.h"
#include <math_constants.h>

namespace cusph{
namespace usergflex{

__constant__ StCteInteraction CTE;
#define CTE_AVAILABLE

#include "FunctionsBasic_iker.h"
#include "FunctionsMath_iker.h"
#include "FunctionsGeo3d_iker.h"
#include "FunSphKernel_iker.h"
#include "FunSphEos_iker.h"
#include "JCellSearch_iker.h"

//==============================================================================
/// Calculates the PK1 stress matrix for each flexible structure particle.
/// Calcula la matriz de tensión PK1 para cada partícula de estructura flexible.
#include "UserGFlexMath_iker.h"
#include "UserGFlexCore_iker.h"
#include "UserGFlexPrecisionMath_iker.h"
#include "UserGFlexAccess_iker.h"
#include "UserGFlexPrecisionCore_iker.h"

} // namespace usergflex
} // namespace cusph

#include "UserGFlexBackend_iker.h"
#include "UserGFlexPrecisionHost_iker.h"

class UserGFlexDispatch{
private:
  UserGFlexDispatch()=delete;

public:
  template<UserGFlexStage stage>
  static void Run(const StInterParmsFlexStrucg& tfs){
    using Mode=UserGParameters::TpFlexPrecisionMode;
    using OriginalMath=cusph::usergflex::UserGFlexMath<Mode::Original>;
    const auto config=UserGParameters::ReadFlex(tfs.user);

    switch(config.PrecisionMode){
      case Mode::Original:
        UserGFlexBackend<OriginalMath>::template Run<stage>(tfs);
        return;
      case Mode::Mixed:
        UserGFlexPrecisionBackend<cusph::usergflex::UserGFlexPrecisionMath<float>>::template Run<stage>(tfs);
        return;
      case Mode::Extended:
        UserGFlexPrecisionBackend<cusph::usergflex::UserGFlexPrecisionMath<double>>::template Run<stage>(tfs);
        return;
    }
    throw "Invalid GPUFlexPrecisionMode.";
  }
};

void UserGInteractionForcesFlexStruc::CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
  UserGFlexDispatch::Run<UserGFlexStage::Reference>(tfs);
}

void UserGInteractionForcesFlexStruc::UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
  UserGFlexDispatch::Run<UserGFlexStage::Geometry>(tfs);
}

void UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
  UserGFlexDispatch::Run<UserGFlexStage::Forces>(tfs);
  UserGFlexContact::Run(tfs);
}

void UserGInteractionForcesFlexStruc::CteInteractionUp(const StCteInteraction* cte){
  cudaMemcpyToSymbol(cusph::usergflex::CTE,cte,sizeof(StCteInteraction));
}

// Preserve the existing solver API; implementation belongs to the class above.
namespace cusph{
void CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
  UserGInteractionForcesFlexStruc::CalcFlexStrucKerCorr(tfs);
}

void UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
  UserGInteractionForcesFlexStruc::UpdateFlexStrucGeometry(tfs);
}

void Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
  UserGInteractionForcesFlexStruc::Interaction_ForcesFlexStruc(tfs);
}
}

#undef CTE_AVAILABLE

void UserGFlexCte::Upload(const StCteInteraction* cte){
  UserGInteractionForcesFlexStruc::CteInteractionUp(cte);
}
