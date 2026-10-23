#ifndef USERFLEXBOUNDARYCORRECTION_H
#define USERFLEXBOUNDARYCORRECTION_H

#include "TypesDef.h"
#include "DualSphDef.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

class UserParameters;

namespace FlexBoundaryCorrection{

struct Options{
  int Layers=0;
  double InwardReach=0.0;
  unsigned MaxIterations=2000;
  double QuadraticError=0.0;
  int BoundaryForm=1;
  double KernelH=0; // Solver kernel length; not an XML parameter.


  // Borrowed during Initialize only; indexed by flexible-particle index.
  const float* ParticleMass=nullptr;
  unsigned ParticleMassCount=0;
};

struct ContactGeometry{
  // 0: no fixed support.
  // 1: existing fixed boundary particles.
  // 2: existing analytical wall.
  int Kind=0;

  std::vector<tdouble3> Fixed;
  tdouble3 Point=TDouble3(0);
  tdouble3 Normal=TDouble3(0,0,1);
};

struct Cache;

struct Diagnostics{
  double AreaError=0;
  double AffineError=0;
  unsigned Iterations=0;
  unsigned QuadraticTried=0;
  unsigned QuadraticAccepted=0;
  unsigned QuadraticAboveThreshold=0;
  std::string Approximation;
};

// Structural loads on constrained particles; opposite to support-on-body loads.
struct AuxiliaryLoads{
  double HourglassEnergy=0;
  std::vector<tdouble3> ClampForce; // Flexible-array indexing; zero for free nodes.
  tdouble3 ClampTotal=TDouble3(0);
  tdouble3 ClampMoment=TDouble3(0); // About the global origin.
};

const AuxiliaryLoads& GetAuxiliaryLoads(const std::shared_ptr<Cache>& cache);

void ElasticForceWithHourglass(
    const std::shared_ptr<Cache>& cache,unsigned n,
    const tdouble3* reference,const tdouble3* position,const unsigned* ridp,
    const tmatrix3f* deformation,const tmatrix3f* referencePK,
    const double* hgModulus,const byte* clamp,
    const std::function<double(double)>& referenceKernel,
    std::vector<tdouble3>& force);

std::string ValidateOptions(const Options& options);

Diagnostics Initialize(
    const Options& options,bool twoD,double dp,unsigned n,
    const tdouble3* reference,const unsigned* body,const byte* selected,
    const float* volume,const unsigned* count,const unsigned* const* pairs,
    const std::function<double(double)>& kernelFac,
    std::shared_ptr<Cache>& cache);

void DefGrad(
    const std::shared_ptr<Cache>& cache,unsigned n,
    const tdouble3* position,const unsigned* ridp,
    tmatrix3f* result,const ContactGeometry* contact=nullptr);

void ElasticForce(
    const std::shared_ptr<Cache>& cache,unsigned n,
    const tmatrix3f* referencePK,
    std::vector<tdouble3>& force);

void BeginAcceptedStep(const UserParameters& user,double time);

void DefGradForSolver(
    const UserParameters& user,unsigned np,unsigned npb,
    const tdouble3* pos,const typecode* code,
    const unsigned* ridp,tmatrix3f* result);

void ElasticForceForSolver(
    const UserParameters& user,unsigned np,
    const tdouble3* reference,const tdouble3* position,
    const typecode* code,const StFlexStrucData* materialData,
    const unsigned* ridp,const tmatrix3f* deformation,
    const tmatrix3f* referencePK,
    const std::function<double(double)>& kernel,
    std::vector<tdouble3>& force);

}

#endif