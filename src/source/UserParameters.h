#ifndef USERPARAMETERS_H
#define USERPARAMETERS_H

#include <string>
#include <vector>

#include "JObject.h"
#include "TypesDef.h"

class JXml;
class JLog2;
class TiXmlElement;

class UserParameters : protected JObject
{
public:
  struct StPoint2d{
    double x=0.0;
    double z=0.0;
  };

  struct StSurface2d{
    // 0: unused; 1: horizontal wall; 2: two-point geometry.
    int value=0;

    // Take effect when value==1
    double WallZ=0.0;

    // Take effect when value==2
    int NormalDirectionMode=1; // 1: normal in +z direction; 2: normal in surface
    StPoint2d point1;
    StPoint2d point2;
  };

  // 1: particle contact; 2: surface contact.
  int FlexBoundContactMode;

  StSurface2d SurfaceType;

  int KnContactMode; // 1: manual; 2: automatic.
  double KnContact;
  int KnContactLinearity; // 1: String (linear); 2: Hertzian (3/2 power).

  int unContactMode; // 1: manual; 2: automatic.
  double unContact;

  bool SaveFlexData; // Enable PartFlex elastic-state output.


  // Explicit User restart request; mutually exclusive with SPH restart.
  bool FlexRestartRequested;
  // XML restartsource: 0=unspecified, 1=prescribed, 2=model_transfer.
  // Required when FlexRestartRequested is true.
  int FlexRestartSourceRequested;


  // Elastic information source: 0=unspecified, 1=file, 2=special_field.
  int FlexRestartElasticInput;
  // File input configuration.
  std::string FlexRestartElasticFile;
  // 0=unspecified, 1=deformation_gradient, 2=cauchy_stress.
  int FlexRestartFileVariableType;

  // Special field: 0=unspecified, 1=undeformed, 2=hydrostatic_pressure.
  int FlexRestartSpecialField;

  bool FlexRestartRestorePairs;
  bool FlexRestartGradientCorrection;
  bool FlexRestartHourglassCorrection;

  // Output
  // Time-varying output configuration.
  struct StOutputDynamic{
    bool DensityMaterial=false;
    bool DensitySPH=false;
    bool PressureSPH=false;
    bool VolumeMaterial=false;
    bool VolumeSPH=false;
    bool Velocity=false;

    // If empty, no output is generated.
    std::vector<std::string> StrainComponents;
    std::vector<std::string> StressComponents;
    std::vector<std::string> StressInvariants;
  };

  // Time-invariant output configuration.
  struct StOutputStatic{
    bool Mass=false;
  };

  struct StOutputObject{
    std::string Name;
    std::vector<unsigned> Mk;
    std::string OutputFolder;

    bool Csv=false;
    bool Vtk=false;

    StOutputDynamic Dynamic;
    StOutputStatic Static;
  };

  // if empty, no output is generated.
  std::vector<StOutputObject> OutputObjects;


  UserParameters();

  void Reset();
  void LoadXml(const JXml* xml,JLog2* log);
  void VisuConfig(JLog2* log)const;




  // Inner variables and functions, not directly accessible to users.
  // Runtime restart state, indexed by flexible-particle index.
  struct StFlexRestartState{
    // All per-particle arrays use the current flexible-particle index.
    std::vector<unsigned> Idp;

    // F*: Xn -> this run's fixed reference X0.
    std::vector<tmatrix3f> Fstar;
    std::vector<float> Jstar;

    // Natural configuration Xn.
    std::vector<float> NaturalVolume;
    std::vector<float> NaturalDensity;

    // Fixed reference configuration X0 of this run.
    std::vector<float> ReferenceVolume;
    std::vector<float> ReferenceDensity;

    // Invariant particle mass.
    std::vector<float> ParticleMass;

    // Natural-reference geometry and discretization data.
    // Supplied by the selected restart source when available.
    // Never replace these data with the current reference geometry X0.
    bool HasNaturalReferenceData=false;

    unsigned ReferenceKernel=0;
    float ReferenceH=0;
    bool Reference2D=false;

    std::vector<tdouble3> NaturalPos;
    std::vector<tmatrix3f> NaturalKerCorr;

    // CSR representation of the original ordered neighbour lists.
    //
    // Row pfs occupies:
    //   [PairOffset[pfs], PairOffset[pfs+1])
    //
    // PairIdp stores stable particle IDs, NOT flexible-array indices.
    std::vector<unsigned> PairOffset;
    std::vector<unsigned> PairIdp;

    // Runtime caches in the CURRENT active pair ordering.
    // These are rebuilt at every restart and are not written to files.
    std::vector<size_t> CorrectionOffset;
    std::vector<unsigned> CorrectionNeighbour;

    // Corrected gradient coefficients for the two endpoints.
    // Both use the same i-minus-j direction.
    std::vector<tfloat3> Grad0I;
    std::vector<tfloat3> Grad0J;

    // F*_i Rn_ij and F*_j Rn_ij.
    std::vector<tfloat3> HgMappedI;
    std::vector<tfloat3> HgMappedJ;

    // Original HG reference coefficient.
    std::vector<float> HgWeight;

    bool GradientReady=false;
    bool HourglassReady=false;

    void Clear(){
      Idp.clear();
      Fstar.clear();
      Jstar.clear();

      NaturalVolume.clear();
      NaturalDensity.clear();
      ReferenceVolume.clear();
      ReferenceDensity.clear();
      ParticleMass.clear();

      HasNaturalReferenceData=false;
      ReferenceKernel=0;
      ReferenceH=0;
      Reference2D=false;

      NaturalPos.clear();
      NaturalKerCorr.clear();
      PairOffset.clear();
      PairIdp.clear();

      CorrectionOffset.clear();
      CorrectionNeighbour.clear();

      Grad0I.clear();
      Grad0J.clear();

      HgMappedI.clear();
      HgMappedJ.clear();
      HgWeight.clear();

      GradientReady=false;
      HourglassReady=false;
    }

    // Checks array layout only; numerical validity is checked separately.
    bool HasRestartStateLayout()const{
      const size_t n=Idp.size();

      return n &&
          Fstar.size()==n &&
          Jstar.size()==n &&
          NaturalVolume.size()==n &&
          NaturalDensity.size()==n &&
          ReferenceVolume.size()==n &&
          ReferenceDensity.size()==n &&
          ParticleMass.size()==n;
    }

    // Checks reference layout only; this is not full reference validation.
    bool HasNaturalReferenceLayout()const{
      return HasNaturalReferenceData &&
          NaturalPos.size()==Idp.size() &&
          NaturalKerCorr.size()==Idp.size() &&
          PairOffset.size()==Idp.size()+1 &&
          !PairOffset.empty() &&
          PairOffset.front()==0 &&
          PairOffset.back()==PairIdp.size();
    }
  };

  StFlexRestartState FlexRestartState;

  // Retrieve Fstar by flexible index and verify particle identity.
  const tmatrix3f& GetFlexRestartFstar(
      unsigned pfs,unsigned idp)const;

  // Requests qualified by the current execution mode.
  // Enabled corrections still require suitable natural-reference data.
  bool UseFlexRestartRestorePairs;
  bool UseFlexRestartGradientCorrection;
  bool UseFlexRestartHourglassCorrection;

  // Output-directory state; reset for every run.
  bool FlexReferenceWritten;

  // Identifies the reference file associated with PartFlex.
  // Preserved through successive restarts.
  std::string FlexReferenceId;

  void ConfigFlexRestartCorrections(
      bool isDirectContinuation,
      bool hasflex,
      JLog2* log);


  // Resolved elastic restart source for this run.
  // XML source values are mapped explicitly; do not cast them to this enum.
  enum class TpFlexRestartSource{
    None,
    DirectContinuation,
    Prescribed,
    ModelTransfer
  };

  TpFlexRestartSource FlexRestartSource;

  bool IsFlexRestart()const{
    return FlexRestartSource!=TpFlexRestartSource::None;
  }

  bool IsFlexDirectContinuation()const{
    return FlexRestartSource==
        TpFlexRestartSource::DirectContinuation;
  }

  bool IsFlexPrescribedRestart()const{
    return FlexRestartSource==
        TpFlexRestartSource::Prescribed;
  }

  bool IsFlexModelTransferRestart()const{
    return FlexRestartSource==
        TpFlexRestartSource::ModelTransfer;
  }

  void ConfigFlexRestartMode(
      bool particleContinuation,
      bool hasflex,
      JLog2* log);

  

private:
  void LoadOutputXml(const JXml* xml,const TiXmlElement* userNode);
  void LoadRestartElasticInputXml(
      const JXml* xml,const TiXmlElement* restart);

};

#endif