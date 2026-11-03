#ifndef USERPARAMETEROUTPUT_H
#define USERPARAMETEROUTPUT_H

#include "JObject.h"
#include "TypesDef.h"
#include "UserParameters.h"
#include "DualSphDef.h"

#include <string>
#include <vector>

class UserParameterOutput : protected JObject
{
public:
  struct StVtkParticle{
    unsigned Idp=0;
    unsigned Mk=0;
    unsigned Boundary=0;

    tdouble3 Pos=TDouble3(0);
    tfloat3 Velocity=TFloat3(0);

    double DensityMaterial=0;
    double DensitySPH=0;
    double PressureSPH=0;
    double VolumeMaterial=0;
    double VolumeSPH=0;
    double Mass=0;

    tmatrix3f Strain={};
    tmatrix3f Stress={};
    // Optional high-precision output tensors; CPU callers retain their existing path.
    bool PreciseTensors=false;
    tmatrix3d Strain64={},Stress64={};
  };

  UserParameterOutput();

  void SaveFiles(
    const std::string& basename,
    double time,
    const UserParameters::StOutputObject& config,
    const std::vector<StVtkParticle>& particles,
    bool staticOnly);

  tmatrix3f CalcCauchyStressOutput(
    const tmatrix3f& defgrad,
    const tmatrix6f& cmat,
    TpConstitModel model,
    double young,
    double poisson)const;
};


#endif