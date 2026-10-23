#include "JSphCpuSingle.h"
#include "UserParameters.h"
#include "FunctionsMath.h"
#include "JLog2.h"

#include <cmath>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <vector>

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

  CompleteFlexRestartMaterialState(true);
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
        InitFlexRestartHydrostatic();
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

void JSphCpuSingle::CompleteFlexRestartMaterialState(bool validateOnly)
{
  const bool undeformed=
      User->IsFlexPrescribedRestart() &&
      User->FlexRestartElasticInput==2 &&
      User->FlexRestartSpecialField==1;

  // Direct continuation retains its existing convention.
  // All four conventions coincide for the prescribed undeformed field.
  const bool useExistingConvention=
      User->IsFlexDirectContinuation() || undeformed;

  const bool conserveMass=
      useExistingConvention || User->FlexRestartConserveMass;

  const bool densityIsNatural=
      useExistingConvention || User->FlexRestartDensityIsNatural;

  if(conserveMass && densityIsNatural){
    // Implemented: fixed natural-state mass with XML density in Xn.
  }
  else if(!conserveMass && densityIsNatural){
    // Implemented: V0=Vgeometry with XML density in Xn.
  }
  else if(!conserveMass && !densityIsNatural){
    // Implemented: V0=Vgeometry with XML density in X0.
  }
  else{
    // Reserved: conserved natural-state mass with XML density in X0.
    Run_Exceptioon(
        "FlexRestart: mass conservation with reference density "
        "is not implemented yet.");
    return;
  }

  // Validation runs before constructing the prescribed elastic field.
  if(validateOnly)return;

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

    // Base geometric particle volume.
    const float volbase=material.vol0;

    // Its configuration reference is selected below.
    const float rhoxml=material.rho0;

    if(!std::isfinite(jstar) || jstar<=0.f ||
       !std::isfinite(volbase) || volbase<=0.f ||
       !std::isfinite(rhoxml) || rhoxml<=0.f)
    {
      Run_Exceptioon(
          "FlexRestart: invalid Fstar or input material properties.");
    }

    state.Jstar[pfs]=jstar;

    if(conserveMass && densityIsNatural){
      // Existing direct-continuation convention.
      state.NaturalVolume[pfs]=volbase;
      state.NaturalDensity[pfs]=rhoxml;

      state.ReferenceVolume[pfs]=volbase*jstar;
      state.ReferenceDensity[pfs]=rhoxml/jstar;
      state.ParticleMass[pfs]=volbase*rhoxml;
    }
    else if(!conserveMass && densityIsNatural){
      // Fixed geometric volume in X0; XML density belongs to Xn.
      state.ReferenceVolume[pfs]=volbase;
      state.NaturalDensity[pfs]=rhoxml;

      state.NaturalVolume[pfs]=volbase/jstar;
      state.ReferenceDensity[pfs]=rhoxml/jstar;
      state.ParticleMass[pfs]=state.NaturalVolume[pfs]*rhoxml;
    }
    else if(!conserveMass && !densityIsNatural){
      // Both the geometric volume and XML density belong to X0.
      state.ReferenceVolume[pfs]=volbase;
      state.ReferenceDensity[pfs]=rhoxml;

      state.NaturalVolume[pfs]=volbase/jstar;
      state.NaturalDensity[pfs]=rhoxml*jstar;
      state.ParticleMass[pfs]=volbase*rhoxml;
    }
    else{
      Run_Exceptioon(
          "FlexRestart: material-state convention is unavailable.");
    }

    if(!std::isfinite(state.NaturalVolume[pfs]) ||
       state.NaturalVolume[pfs]<=0.f ||
       !std::isfinite(state.NaturalDensity[pfs]) ||
       state.NaturalDensity[pfs]<=0.f ||
       !std::isfinite(state.ReferenceVolume[pfs]) ||
       state.ReferenceVolume[pfs]<=0.f ||
       !std::isfinite(state.ReferenceDensity[pfs]) ||
       state.ReferenceDensity[pfs]<=0.f ||
       !std::isfinite(state.ParticleMass[pfs]) ||
       state.ParticleMass[pfs]<=0.f)
    {
      Run_Exceptioon(
          "FlexRestart: invalid material density, volume or mass.");
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

//==============================================================================
/// Builds a local-column hydrostatic prestress using the elastic constitutive law.
//==============================================================================
void JSphCpuSingle::InitFlexRestartHydrostatic()
{
  const unsigned nf=CaseNflexstruc;
  if(!nf || !FlexStrucRidpc || !FlexStrucDatac)
    Run_Exceptioon("FlexRestart.Hydrostatic: flexible particles are unavailable.");

  const double gx=Gravity.x,gy=Gravity.y,gz=Gravity.z;
  const double gravity=std::sqrt(gx*gx+gy*gy+gz*gz);
  if(!std::isfinite(gravity) || !std::isfinite(Dp) || Dp<=0)
    Run_Exceptioon("FlexRestart.Hydrostatic: invalid gravity or particle spacing.");
  if(Simulate2D && gy!=0.0)
    Run_Exceptioon("FlexRestart.Hydrostatic: 2D gravity must lie in the x-z plane.");

  const double up[3]={gravity>0 ? -gx/gravity : 0.0,
                      gravity>0 ? -gy/gravity : 0.0,
                      gravity>0 ? -gz/gravity : 0.0};
  const double half=0.5*double(Dp);
  const double tolerance=1e-6*double(Dp);

  const double columnGapCoef=User->FlexRestartColumnGapCoef;

  if(!std::isfinite(columnGapCoef) || columnGapCoef<0.0)
    Run_Exceptioon(
        "FlexRestart.Hydrostatic: invalid column gap coefficient.");

  const double columnGapTolerance=columnGapCoef*double(Dp);

  if(!std::isfinite(columnGapTolerance))
    Run_Exceptioon(
        "FlexRestart.Hydrostatic: column gap tolerance overflow.");

  UserParameters::StFlexRestartState state;
  state.Idp.resize(nf);
  state.Fstar.assign(nf,TMatrix3f());

  std::vector<tdouble3> positions(nf);
  std::vector<std::vector<unsigned>> bodies(FlexStrucCount);
  std::unordered_set<unsigned> seen;
  seen.reserve(nf);

  for(unsigned i=0;i<nf;i++){
    const unsigned p=FlexStrucRidpc[i];
    if(p>=Np)
      Run_Exceptioon("FlexRestart.Hydrostatic: invalid particle index.");

    const typecode code=Code_c->cptr()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFlexStrucAny(code))
      Run_Exceptioon("FlexRestart.Hydrostatic: invalid flexible particle.");

    const unsigned body=CODE_GetIbodyFlexStruc(code);
    if(body>=FlexStrucCount)
      Run_Exceptioon("FlexRestart.Hydrostatic: invalid material body.");

    const unsigned id=Idp_c->cptr()[p];
    if(!seen.insert(id).second)
      Run_Exceptioon("FlexRestart.Hydrostatic: duplicated particle ID.");

    state.Idp[i]=id;
    positions[i]=Pos_c->cptr()[p];

    const tdouble3& x=positions[i];
    if(!std::isfinite(x.x) ||
       !std::isfinite(x.y) ||
       !std::isfinite(x.z))
    {
      Run_Exceptioon("FlexRestart.Hydrostatic: invalid particle position.");
    }

    bodies[body].push_back(i);
  }

  struct ColumnInterval{
    double begin,end,head;
  };

  std::vector<ColumnInterval> intervals;
  double pressuremax=0.0,jmin=1.0;

  for(unsigned body=0;body<FlexStrucCount;body++){
    const auto& ids=bodies[body];
    if(ids.empty())continue;

    const StFlexStrucData& material=FlexStrucDatac[body];
    const double rhoinput=material.rho0;

    // Use the same stiffness coefficients as the constitutive calculation.
    const double modulus=Simulate2D
        ? double(material.cmat.a11)+double(material.cmat.a13)
        : double(material.cmat.a11)+double(material.cmat.a12)
          +double(material.cmat.a13);

    if(!std::isfinite(rhoinput) || rhoinput<=0 ||
       !std::isfinite(modulus) || modulus<=0)
    {
      Run_Exceptioon(
          "FlexRestart.Hydrostatic: invalid material properties.");
    }

    intervals.reserve(ids.size());

    for(unsigned i:ids){
      double depth=0.0;

      if(gravity>0){
        intervals.clear();
        const tdouble3& xi=positions[i];

        // Intersect the upward ray with particle cells of this body.
        for(unsigned j:ids){
          const tdouble3& xj=positions[j];
          const double delta[3]={
            xj.x-xi.x,
            xj.y-xi.y,
            xj.z-xi.z
          };

          double begin=-std::numeric_limits<double>::infinity();
          double end=std::numeric_limits<double>::infinity();
          bool intersects=true;

          for(unsigned axis=0;axis<3;axis++){
            if(Simulate2D && axis==1)continue;

            if(std::fabs(up[axis])<1e-12){
              if(std::fabs(delta[axis])>half+tolerance){
                intersects=false;
                break;
              }
            }
            else{
              double a=(delta[axis]-half)/up[axis];
              double b=(delta[axis]+half)/up[axis];

              if(a>b)std::swap(a,b);
              begin=std::max(begin,a);
              end=std::min(end,b);

              if(end-begin<=tolerance){
                intersects=false;
                break;
              }
            }
          }

          if(intersects && end>=0.0){
            const double head=
                delta[0]*up[0]+
                delta[1]*up[1]+
                delta[2]*up[2];

            intervals.push_back({begin,end,head});
          }
        }

        std::sort(intervals.begin(),intervals.end(),
            [](const ColumnInterval& a,const ColumnInterval& b){
              return a.begin<b.begin;
            });

        // Bridge permitted discretization gaps.
        // Use the upper particle centre as the pressure datum.
        double reach=0.0;
        for(const auto& interval:intervals){
          const double gap=interval.begin-reach;

          if(gap>tolerance &&
             gap-tolerance>columnGapTolerance)
            break;

          reach=std::max(reach,interval.end);
          depth=std::max(depth,interval.head);
        }
      }

      // Natural-density input: retain the existing pressure approximation.
      // Reference-density input: use the prescribed X0 density directly.
      // Do not replace rhoinput by the reconstructed NaturalDensity here.
      const double pressure=rhoinput*gravity*depth;
      const double ratio=pressure/modulus;

      if(!std::isfinite(ratio) || ratio<0)
        Run_Exceptioon("FlexRestart.Hydrostatic: invalid prescribed pressure.");

      double stretch;

      if(Simulate2D){
        // F*=diag(a,1,a), J*=a^2.
        // sigma_xx=sigma_zz=modulus*(J*-1)/2.
        const double jacobian=1.0-2.0*ratio;

        if(jacobian<=0.0){
          Run_Exceptioon(
              "FlexRestart.Hydrostatic: pressure exceeds "
              "the admissible 2D elastic compression.");
        }

        stretch=std::sqrt(jacobian);
      }
      else{
        // 3D SVK: F*=a I, sigma=modulus*(a-1/a)/2 I.
        stretch=1.0/(std::hypot(1.0,ratio)+ratio);
      }

      const float a=float(stretch);
      state.Fstar[i]=TMatrix3f(
          a,0,0,
          0,Simulate2D ? 1.f : a,0,
          0,0,a);

      const double jacobian=
          fmath::Determinant3x3(state.Fstar[i]);

      if(!std::isfinite(jacobian) || jacobian<=0.0)
        Run_Exceptioon("FlexRestart.Hydrostatic: invalid stored Fstar.");

      pressuremax=std::max(pressuremax,pressure);
      jmin=std::min(jmin,jacobian);
    }
  }

  // CompleteFlexRestartMaterialState() supplies rho, volume and conserved mass.
  // Do not fabricate natural positions or overwrite fluid-EOS state arrays.
  User->FlexRestartState=state;
  Log->Printf(
      "FlexRestart.Hydrostatic: columnGapCoef=%g; "
      "columnGapTolerance=%g m.",
      columnGapCoef,columnGapTolerance);
  Log->Printf(
      "FlexRestart.Prescribed: local-column hydrostatic field initialised "
      "for %u flexible/clamp particles; pmax=%g Pa; Jstar_min=%g.",
      nf,pressuremax,jmin);
}