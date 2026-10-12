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
  struct StFlexRestartHistory{
    // All arrays use the flexible-particle index.
    std::vector<unsigned> Idp;

    // F*: Xn -> this run's X0.
    std::vector<tmatrix3f> DefGrad;
    std::vector<float> Jstar;

    // Natural configuration Xn.
    std::vector<float> VolN;
    std::vector<float> RhoN;

    // Fixed reference configuration X0 of this run.
    std::vector<float> Vol0;
    std::vector<float> Rho0;

    // Invariant particle mass.
    std::vector<float> Mass;

    void Clear(){
      Idp.clear();
      DefGrad.clear();
      Jstar.clear();
      VolN.clear();
      RhoN.clear();
      Vol0.clear();
      Rho0.clear();
      Mass.clear();
    }

    bool Available()const{
      const size_t n=Idp.size();
      return n &&
          DefGrad.size()==n &&
          Jstar.size()==n &&
          VolN.size()==n &&
          RhoN.size()==n &&
          Vol0.size()==n &&
          Rho0.size()==n &&
          Mass.size()==n;
    }
  };

  StFlexRestartHistory FlexRestartHistory;

  // Retrieve history by flexible index and verify particle identity.
  const tmatrix3f& GetFlexRestartDefGrad(
      unsigned pfs,unsigned idp)const;
  

private:
  void LoadOutputXml(const JXml* xml,const TiXmlElement* userNode);

};

#endif