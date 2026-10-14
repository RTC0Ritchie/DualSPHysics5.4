#include "JSphCpuSingle.h"
#include "UserParameters.h"
#include "FunctionsMath.h"
#include "JLog2.h"

#include <cmath>
#include <unordered_set>

//==============================================================================
/// Initialises elastic restart state from the selected source.
//==============================================================================

void JSphCpuSingle::InitFlexRestartState()
{
  User->FlexRestartState.Clear();
  User->FlexReferenceId.clear();
  User->FlexReferenceWritten=false;

  if(!User->IsFlexRestart())
    return;

  if(!FlexStruc || !CaseNflexstruc){
    Run_Exceptioon(
        "FlexRestart: flexible particles are required.");
  }

  if(User->IsFlexDirectContinuation()){
    LoadFlexDirectContinuationState();
  }
  else if(User->IsFlexPrescribedRestart()){
    InitFlexRestartFromPrescribed();
  }
  else if(User->IsFlexModelTransferRestart()){
    InitFlexRestartFromModelTransfer();
  }
  else{
    Run_Exceptioon(
        "FlexRestart: unknown state initialisation source.");
  }

  CompleteFlexRestartMaterialState();

  const bool userRestart=
      User->IsFlexPrescribedRestart() ||
      User->IsFlexModelTransferRestart();

  if(userRestart &&
     User->FlexRestartState.HasNaturalReferenceLayout() &&
     User->FlexReferenceId.empty())
  {
    User->FlexReferenceId=RunCode;
  }
}

void JSphCpuSingle::InitFlexRestartFromPrescribed()
{
  if(User->FlexRestartElasticInput==1){
    // Filename and variable type have been read from XML.
    // File parsing and material-state reconstruction are reserved.
    Run_Exceptioon(
        "FlexRestart.Prescribed: elastic file input "
        "is not implemented yet.");
    return;
  }

  if(User->FlexRestartElasticInput==2){
    switch(User->FlexRestartSpecialField){
      case 1:
        InitFlexRestartUndeformed();
        return;

      case 2:
        Run_Exceptioon(
            "FlexRestart.Prescribed: hydrostatic-pressure "
            "initialisation is not implemented yet.");
        return;

      default:
        Run_Exceptioon(
            "FlexRestart.Prescribed: invalid special field.");
        return;
    }
  }

  Run_Exceptioon(
      "FlexRestart.Prescribed: invalid elastic input source.");
}

void JSphCpuSingle::InitFlexRestartFromModelTransfer()
{
  // Future implementation:
  //
  // 1. Read particle-wise material information from the source model.
  // 2. Map source particles to current flexible/clamp particles.
  // 3. Convert source quantities to the target constitutive convention.
  // 4. Build and validate the target elastic restart state.
  // 5. Supply natural-reference data only when actually available.
  //
  // The current downstream path requires Idp and Fstar.
  // Missing natural-reference information must not be fabricated.

  Run_Exceptioon(
      "FlexRestart.ModelTransfer: "
      "initialisation is not implemented yet.");
}

void JSphCpuSingle::CompleteFlexRestartMaterialState()
{
  auto& state=User->FlexRestartState;
  const unsigned nf=CaseNflexstruc;

  if(state.Idp.size()!=nf ||
     state.Fstar.size()!=nf)
  {
    Run_Exceptioon(
        "FlexRestart: incomplete restart state.");
  }

  state.Jstar.resize(nf);
  state.NaturalVolume.resize(nf);
  state.NaturalDensity.resize(nf);
  state.ReferenceVolume.resize(nf);
  state.ReferenceDensity.resize(nf);
  state.ParticleMass.resize(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned p=FlexStrucRidpc[pfs];

    if(p>=Np ||
       state.Idp[pfs]!=Idp_c->cptr()[p])
    {
      Run_Exceptioon(
          "FlexRestart: particle ID mismatch.");
    }

    const typecode code=Code_c->cptr()[p];

    if(!CODE_IsNormal(code) ||
       !CODE_IsFlexStrucAny(code))
    {
      Run_Exceptioon(
          "FlexRestart: invalid particle in restart state.");
    }

    const tmatrix3f& f=state.Fstar[pfs];

    const float values[]={
      f.a11,f.a12,f.a13,
      f.a21,f.a22,f.a23,
      f.a31,f.a32,f.a33
    };

    for(float value:values){
      if(!std::isfinite(value)){
        Run_Exceptioon(
            "FlexRestart: non-finite Fstar.");
      }
    }

    const unsigned body=CODE_GetIbodyFlexStruc(code);
    const StFlexStrucData& material=FlexStrucDatac[body];

    const float jstar=fmath::Determinant3x3(f);
    const float voln=material.vol0;
    const float rhon=material.rho0;

    if(!std::isfinite(jstar) || jstar<=0.f ||
       !std::isfinite(voln) || voln<=0.f ||
       !std::isfinite(rhon) || rhon<=0.f)
    {
      Run_Exceptioon(
          "FlexRestart: invalid Fstar or natural material properties.");
    }

    state.Jstar[pfs]=jstar;

    state.NaturalVolume[pfs]=voln;
    state.NaturalDensity[pfs]=rhon;

    state.ReferenceVolume[pfs]=voln*jstar;
    state.ReferenceDensity[pfs]=rhon/jstar;
    state.ParticleMass[pfs]=voln*rhon;

    if(!std::isfinite(state.ReferenceVolume[pfs]) ||
       state.ReferenceVolume[pfs]<=0.f ||
       !std::isfinite(state.ReferenceDensity[pfs]) ||
       state.ReferenceDensity[pfs]<=0.f ||
       !std::isfinite(state.ParticleMass[pfs]) ||
       state.ParticleMass[pfs]<=0.f)
    {
      Run_Exceptioon(
          "FlexRestart: invalid reference material properties.");
    }
  }
}

//==============================================================================
/// Manually restart behaviour
//==============================================================================
void JSphCpuSingle::InitFlexRestartUndeformed()
{
  const unsigned nf=CaseNflexstruc;

  if(!nf || !FlexStrucRidpc){
    Run_Exceptioon(
        "FlexRestart.Undeformed: flexible particles "
        "are unavailable.");
  }

  UserParameters::StFlexRestartState state;

  state.Idp.resize(nf);
  state.Fstar.assign(nf,TMatrix3f());

  std::unordered_set<unsigned> seen;
  seen.reserve(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned p=FlexStrucRidpc[pfs];

    if(p>=Np){
      Run_Exceptioon(
          "FlexRestart.Undeformed: invalid particle index.");
    }

    const typecode code=Code_c->cptr()[p];

    if(!CODE_IsNormal(code) ||
       !CODE_IsFlexStrucAny(code))
    {
      Run_Exceptioon(
          "FlexRestart.Undeformed: invalid flexible particle.");
    }

    const unsigned id=Idp_c->cptr()[p];

    if(!seen.insert(id).second){
      Run_Exceptioon(
          "FlexRestart.Undeformed: duplicated particle ID.");
    }

    state.Idp[pfs]=id;
  }

  // Fstar=I identifies this run's reference with the natural configuration.
  // Keep DefGradc for the relative deformation calculated by the solver.
  // Material quantities are completed by CompleteFlexRestartMaterialState().
  User->FlexRestartState=state;

  Log->Printf(
      "FlexRestart.Prescribed: undeformed field initialised "
      "for %u flexible/clamp particles; Fstar=I.",
      nf);
}