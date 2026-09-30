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

    tdouble3 Pos=TDouble3(0);
    tfloat3 Velocity=TFloat3(0);

    double Density=0;
    double Pressure=0;
    double Volume=0;
    double Mass=0;

    tmatrix3f Strain={};
    tmatrix3f Stress={};
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