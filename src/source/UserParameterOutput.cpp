#include "UserParameterOutput.h"
#include "JDataArrays.h"
#include "JSpVtkData.h"
#include "FunctionsMath.h"

#include "JSphCpuSingle.h"
#include "JSphMk.h"
#include "JSphFlexStruc.h"
#include "Functions.h"
#include "JOutputCsv.h"
#include "FunSphEos.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

UserParameterOutput::UserParameterOutput(){
  ClassName="UserParameterOutput";
}

void UserParameterOutput::SaveFiles(
  const std::string& basename,
  double time,
  const UserParameters::StOutputObject& config,
  const std::vector<StVtkParticle>& particles,
  bool staticOnly)
{
  const unsigned n=unsigned(particles.size());

  // Arrays own their buffers.
  JDataArrays arrays;

  tdouble3* pos=arrays.CreateArrayPtrDouble3("Pos",n);
  unsigned* idp=arrays.CreateArrayPtrUint("Idp",n);
  unsigned* mk=arrays.CreateArrayPtrUint("Mk",n);
  arrays.CreateArrayDouble("Time",n,true,time);

  for(unsigned p=0;p<n;p++){
    pos[p]=particles[p].Pos;
    idp[p]=particles[p].Idp;
    mk[p]=particles[p].Mk;
  }

  // Copy a scalar particle field.
  const auto AddScalar=[&](
    const std::string& name,
    double StVtkParticle::*member)
  {
    double* values=arrays.CreateArrayPtrDouble(name,n);

    for(unsigned p=0;p<n;p++){
      values[p]=particles[p].*member;
    }
  };

  // Read a symmetric tensor component.
  const auto Component=[&](
    const tmatrix3f& tensor,
    const std::string& field)->double
  {
    if(field=="xx")return tensor.a11;
    if(field=="yy")return tensor.a22;
    if(field=="zz")return tensor.a33;
    if(field=="xy")return tensor.a12;
    if(field=="xz")return tensor.a13;
    if(field=="yz")return tensor.a23;

    Run_Exceptioon("Unsupported tensor component: "+field);
    return 0.0;
  };

  if(staticOnly){
    if(config.Static.Mass){
      AddScalar("Mass",&StVtkParticle::Mass);
    }
  }
  else{
    const auto& dynamic=config.Dynamic;

    if(dynamic.DensityMaterial){
      AddScalar("Density",&StVtkParticle::Density);
    }

    if(dynamic.PressureSPH){
      AddScalar("Pressure",&StVtkParticle::Pressure);
    }

    if(dynamic.VolumeMaterial){
      AddScalar("Volume",&StVtkParticle::Volume);
    }

    if(dynamic.Velocity){
      tfloat3* velocity=
        arrays.CreateArrayPtrFloat3("Velocity",n);

      for(unsigned p=0;p<n;p++){
        velocity[p]=particles[p].Velocity;
      }
    }

    // Green-Lagrange tensor components.
    for(const std::string& field:dynamic.StrainComponents){
      double* values=arrays.CreateArrayPtrDouble(
        "Strain_"+field,n);

      for(unsigned p=0;p<n;p++){
        values[p]=Component(particles[p].Strain,field);
      }
    }

    // Cauchy stress components.
    for(const std::string& field:dynamic.StressComponents){
      double* values=arrays.CreateArrayPtrDouble(
        "Stress_"+field,n);

      for(unsigned p=0;p<n;p++){
        values[p]=Component(particles[p].Stress,field);
      }
    }

    // Derived stress scalars.
    for(const std::string& field:dynamic.StressInvariants){
      if(field!="mean"
          && field!="mech_pressure"
          && field!="von_mises"){
        Run_Exceptioon("Unsupported stress invariant: "+field);
      }

      double* values=arrays.CreateArrayPtrDouble(
        "Stress_"+field,n);

      for(unsigned p=0;p<n;p++){
        const tmatrix3f& s=particles[p].Stress;

        const double xx=s.a11;
        const double yy=s.a22;
        const double zz=s.a33;
        const double xy=s.a12;
        const double xz=s.a13;
        const double yz=s.a23;

        const double mean=(xx+yy+zz)/3.0;

        if(field=="mean"){
          values[p]=mean;
        }
        else if(field=="mech_pressure"){
          values[p]=-mean;
        }
        else{
          const double vm2=
            0.5*((xx-yy)*(xx-yy)
              +(yy-zz)*(yy-zz)
              +(zz-xx)*(zz-xx))
            +3.0*(xy*xy+xz*xz+yz*yz);

          values[p]=std::sqrt(vm2);
        }
      }
    }
  }

  if(config.Vtk){
    JSpVtkData::Save(basename+".vtk",arrays,"Pos");
  }

  if(config.Csv){
    JOutputCsv csv(true);
    csv.SaveCsv(basename+".csv",arrays);
  }
}

// Same as UserInteractionForcesFlexStruc.cpp
tmatrix3f UserParameterOutput::CalcCauchyStressOutput(
    const tmatrix3f& defgrad,
    const tmatrix6f& cmat,
    TpConstitModel model,
    double young,
    double poisson)const
{
  if(model!=CONSTITMODEL_PlaneStrain
      && model!=CONSTITMODEL_PlaneStress
      && model!=CONSTITMODEL_SVK){
    Run_Exceptioon("Unsupported constitutive model.");
  }

  // Check the current volume ratio.
  const double j=fmath::Determinant3x3(defgrad);

  if(!std::isfinite(j) || j<=0.0){
    Run_Exceptioon(
        "Cannot compute Cauchy stress: invalid det(F).");
  }

  // Compute Green-Lagrange strain.
  tmatrix3f gl=fmath::MulMatrix3x3(
      fmath::TrasMatrix3x3(defgrad),defgrad);

  gl.a11-=1.0f;gl.a22-=1.0f;gl.a33-=1.0f;
  gl.a11*=0.5f;gl.a12*=0.5f;gl.a13*=0.5f;
  gl.a21*=0.5f;gl.a22*=0.5f;gl.a23*=0.5f;
  gl.a31*=0.5f;gl.a32*=0.5f;gl.a33*=0.5f;

  // Use a local constitutive matrix.
  tmatrix6f c=cmat;

  // Recover the omitted plane-strain PK2 component.
  if(model==CONSTITMODEL_PlaneStrain){
    if(!std::isfinite(young) || young<=0.0
        || !std::isfinite(poisson)
        || poisson<=-1.0 || poisson>=0.5){
      Run_Exceptioon("Invalid plane-strain material parameters.");
    }

    const double lambda=
        young*poisson/
        ((1.0+poisson)*(1.0-2.0*poisson));

    const double mu=young/(2.0*(1.0+poisson));

    c.a21=float(lambda);
    c.a22=float(lambda+2.0*mu);
    c.a23=float(lambda);
  }

  // Voigt order: xx, yy, zz, yz, xz, xy.
  // Shear entries use engineering strain.
  const float strain[6]={
      gl.a11,
      gl.a22,
      gl.a33,
      gl.a23+gl.a32,
      gl.a31+gl.a13,
      gl.a12+gl.a21
  };

  const float stiffness[6][6]={
      {c.a11,c.a12,c.a13,c.a14,c.a15,c.a16},
      {c.a21,c.a22,c.a23,c.a24,c.a25,c.a26},
      {c.a31,c.a32,c.a33,c.a34,c.a35,c.a36},
      {c.a41,c.a42,c.a43,c.a44,c.a45,c.a46},
      {c.a51,c.a52,c.a53,c.a54,c.a55,c.a56},
      {c.a61,c.a62,c.a63,c.a64,c.a65,c.a66}
  };

  // Compute PK2 stress.
  float s[6]={};

  for(unsigned row=0;row<6;row++){
    for(unsigned col=0;col<6;col++){
      s[row]+=stiffness[row][col]*strain[col];
    }
  }

  const tmatrix3f pk2={
      s[0],s[5],s[4],
      s[5],s[1],s[3],
      s[4],s[3],s[2]
  };

  // P = F * S.
  const tmatrix3f pk1=
      fmath::MulMatrix3x3(defgrad,pk2);

  // sigma = P * transpose(F) / J.
  const tmatrix3f numerator=fmath::MulMatrix3x3(
      pk1,fmath::TrasMatrix3x3(defgrad));

  tmatrix3f stress={};

  stress.a11=float(double(numerator.a11)/j);
  stress.a22=float(double(numerator.a22)/j);
  stress.a33=float(double(numerator.a33)/j);

  // Remove round-off asymmetry.
  stress.a12=stress.a21=float(
      0.5*(double(numerator.a12)+numerator.a21)/j);

  stress.a13=stress.a31=float(
      0.5*(double(numerator.a13)+numerator.a31)/j);

  stress.a23=stress.a32=float(
      0.5*(double(numerator.a23)+numerator.a32)/j);

  const float values[6]={
      stress.a11,stress.a22,stress.a33,
      stress.a12,stress.a13,stress.a23
  };

  for(float value:values){
    if(!std::isfinite(value)){
      Run_Exceptioon("Non-finite Cauchy stress.");
    }
  }

  return stress;
}

void JSphCpuSingle::SaveUserOutput(
  unsigned npsave,
  const unsigned* svidp,
  const tdouble3* svpos,
  const tfloat3* svvel,
  const float* svrho)
{
  const UserParameters& User = *this->User;
  if(User.OutputObjects.empty())return;

  // First output frame of this run, including restart runs.
  const bool firstFrame=(Part==PartIni);

  const auto HasStress=[](
    const UserParameters::StOutputDynamic& d)->bool
  {
    return !d.StressComponents.empty()
      || !d.StressInvariants.empty();
  };

  const auto HasMaterialFields=[&](
    const UserParameters::StOutputDynamic& d)->bool
  {
    return d.DensityMaterial
      || d.VolumeMaterial
      || !d.StrainComponents.empty()
      || HasStress(d);
  };

  bool needFlexMap=false;

  for(const auto& config:User.OutputObjects){
    if(!config.Vtk && !config.Csv)continue;

    if(HasMaterialFields(config.Dynamic)
        || (firstFrame && config.Static.Mass)){
      needFlexMap=true;
      break;
    }
  }

  // Stable particle ID -> flexible particle index.
  std::unordered_map<unsigned,unsigned> flexById;

  if(needFlexMap){
    flexById.reserve(CaseNflexstruc);

    for(unsigned pfs=0;pfs<CaseNflexstruc;pfs++){
      const unsigned p=FlexStrucRidpc[pfs];

      if(p>=Np){
        Run_Exceptioon("Invalid flexible particle mapping.");
      }

      const typecode code=Code_c->cptr()[p];

      if(CODE_IsNormal(code)){
        if(!CODE_IsFlexStrucAny(code)){
          Run_Exceptioon("Invalid flexible particle code.");
        }

        flexById.emplace(Idp_c->cptr()[p],pfs);
      }
    }
  }

  UserParameterOutput userOutput;

  for(const auto& config:User.OutputObjects){
    if(!config.Vtk && !config.Csv)continue;

    const auto& dynamic=config.Dynamic;

    const bool outputStress=HasStress(dynamic);
    const bool outputStrain=!dynamic.StrainComponents.empty();
    const bool outputMaterial=HasMaterialFields(dynamic);

    const bool outputDynamic=
      outputMaterial || dynamic.PressureSPH || dynamic.Velocity;

    const bool outputMass=firstFrame && config.Static.Mass;

    if(!outputDynamic && !outputMass)continue;

    for(unsigned mk:config.Mk){
      if(MkInfo->GetMkBlockByMk(word(mk))>=MkInfo->Size()){
        Run_Exceptioon(
          fun::PrintStr("Unknown output MK: %u",mk));
      }
    }

    std::vector<UserParameterOutput::StVtkParticle> particles;

    for(unsigned q=0;q<npsave;q++){
      const unsigned id=svidp[q];
      const unsigned block=MkInfo->GetMkBlockById(id);

      if(block>=MkInfo->Size()){
        Run_Exceptioon(
          fun::PrintStr("Cannot identify MK for Idp=%u.",id));
      }

      const JSphMkBlock* mkblock=MkInfo->Mkblock(block);
      const unsigned mk=mkblock->Mk;

      if(std::find(config.Mk.begin(),config.Mk.end(),mk)
          ==config.Mk.end()){
        continue;
      }

      UserParameterOutput::StVtkParticle particle;

      particle.Idp=id;
      particle.Mk=mk;
      particle.Pos=svpos[q];

      if(dynamic.Velocity){
        particle.Velocity=svvel[q];
      }

      // Use SPH density from this output frame.
      if(dynamic.PressureSPH){
        particle.Pressure=fsph::ComputePress(svrho[q],CSP);
      }

      const auto it=flexById.find(id);

      if(outputMaterial && it==flexById.end()){
        Run_Exceptioon(fun::PrintStr(
          "Material output requires a flexible or clamp "
          "particle: Idp=%u, MK=%u.",
          id,mk));
      }

      if(it!=flexById.end()){
        const unsigned pfs=it->second;
        const unsigned p=FlexStrucRidpc[pfs];

        if(p>=Np || !CODE_IsFlexStrucAny(Code_c->cptr()[p])){
          Run_Exceptioon("Invalid flexible particle mapping.");
        }

        const unsigned body=
          CODE_GetIbodyFlexStruc(Code_c->cptr()[p]);

        const JSphFlexStrucBody* bodyData=FlexStruc->GetBody(body);

        if(!bodyData){
          Run_Exceptioon("Invalid flexible body index.");
        }

        const StFlexStrucData& material=FlexStrucDatac[body];

        // Reference mass does not require F.
        if(outputMass){
          particle.Mass=
            double(material.rho0)*double(material.vol0);
        }

        if(outputMaterial){
          if(!NumPairsc[pfs]){
            Run_Exceptioon(
              "Cannot compute material fields without "
              "structural neighbours.");
          }

          const tmatrix3f& f=DefGradc[pfs];
          const double j=fmath::Determinant3x3(f);

          if(!std::isfinite(j) || j<=0.0){
            Run_Exceptioon(fun::PrintStr(
              "Invalid output F: Part=%u Step=%u Time=%.17g "
              "Idp=%u MK=%u p=%u pfs=%u kind=%s pairs=%u "
              "detF=%.17g detKerCorr=%.17g "
              "F=[%.9g %.9g %.9g; %.9g %.9g %.9g; %.9g %.9g %.9g]",
              unsigned(Part),unsigned(Nstep),TimeStep,
              id,mk,p,pfs,
              CODE_IsFlexStrucFlex(Code_c->cptr()[p])?"flex":"clamp",
              NumPairsc[pfs],
              j,double(fmath::Determinant3x3(KerCorrc[pfs])),
              double(f.a11),double(f.a12),double(f.a13),
              double(f.a21),double(f.a22),double(f.a23),
              double(f.a31),double(f.a32),double(f.a33)));
          }

          if(dynamic.DensityMaterial){
            particle.Density=double(material.rho0)/j;
          }

          if(dynamic.VolumeMaterial){
            particle.Volume=double(material.vol0)*j;
          }

          if(outputStrain){
            const tmatrix3f c=fmath::MulMatrix3x3(
              fmath::TrasMatrix3x3(f),f);

            tmatrix3f& e=particle.Strain;

            e.a11=0.5f*(c.a11-1.0f);
            e.a22=0.5f*(c.a22-1.0f);
            e.a33=0.5f*(c.a33-1.0f);

            e.a12=e.a21=0.25f*(c.a12+c.a21);
            e.a13=e.a31=0.25f*(c.a13+c.a31);
            e.a23=e.a32=0.25f*(c.a23+c.a32);
          }

          if(outputStress){
            particle.Stress=userOutput.CalcCauchyStressOutput(
              f,
              material.cmat,
              bodyData->GetConstitModel(),
              material.youngmod,
              material.poissratio);
          }
        }
      }
      else if(outputMass){
        // Mass of ordinary SPH and rigid-body particles.
        const typecode code=mkblock->Code;

        if(CODE_IsFloating(code)){
          const unsigned body=CODE_GetTypeValue(code);

          if(body>=FtCount){
            Run_Exceptioon("Invalid floating body index.");
          }

          particle.Mass=FtObjs[body].massp;
        }
        else if(CODE_IsFluid(code)){
          particle.Mass=MassFluid;
        }
        else{
          particle.Mass=MassBound;
        }
      }

      particles.push_back(particle);
    }

    std::string outputDir;

    if(config.OutputFolder.empty()){
      outputDir=DirOut;
    }
    else if(config.OutputFolder[0]=='$'){
      outputDir=
        fun::GetDirWithSlash(DirOut)+config.OutputFolder.substr(1);
    }
    else{
      outputDir=config.OutputFolder;
    }

    outputDir=fun::GetDirWithSlash(outputDir);

    // Remove the temporary extension after adding the frame number.
    const std::string frameFile=fun::FileNameSec(
      "User_"+config.Name+".vtk",Part);

    const std::string basename=
      outputDir+frameFile.substr(0,frameFile.size()-4);

    if(outputDynamic){
      userOutput.SaveFiles(
        basename,TimeStep,config,particles,false);
    }

    if(outputMass){
      userOutput.SaveFiles(
        basename+"_static",TimeStep,config,particles,true);
    }
  }
}

