#include "JSphGpuSingle.h"
#include "UserGParameterOutput.h"
#include "UserGFlexPrecision.h"
#include "UserParameterOutput.h"
#include "UserGParameters.h"
#include "JSphGpu_ker.h"
#include "JSphMk.h"
#include "JSphFlexStruc.h"
#include "JSphCfgRun.h"
#include "Functions.h"
#include "FunctionsMath.h"
#include "FunSphEos.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// Output postprocessing runs only at saved frames. Keep the canonical H in
// double here, even for mode1; never subtract I from the float solver mirror.
class UserGOutputMath{
public:
  static double Calculate(const tmatrix3d& hh,const StFlexStrucData& mat
    ,TpConstitModel model,bool strain,bool stress,UserParameterOutput::StVtkParticle& out){
    if(model!=CONSTITMODEL_PlaneStrain && model!=CONSTITMODEL_PlaneStress && model!=CONSTITMODEL_SVK)
      throw std::runtime_error("Unsupported GPU output constitutive model.");
    const double* h=&hh.a11;double f[9],e[9]={};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){
      const int q=3*i+j;f[q]=h[q]+(i==j?1.:0.);
      e[q]=.5*(h[q]+h[3*j+i]);
      for(int k=0;k<3;k++)e[q]+=.5*h[3*k+i]*h[3*k+j];
    }
    const double det=f[0]*(f[4]*f[8]-f[5]*f[7])-f[1]*(f[3]*f[8]-f[5]*f[6])+f[2]*(f[3]*f[7]-f[4]*f[6]);
    if(!std::isfinite(det) || det<=0)throw std::runtime_error("Invalid GPU output deformation determinant.");
    out.PreciseTensors=true;
    if(strain)for(int k=0;k<9;k++)(&out.Strain64.a11)[k]=e[k];
    if(stress){
      tmatrix6f c=mat.cmat;
      if(model==CONSTITMODEL_PlaneStrain){
        const double young=mat.youngmod,poisson=mat.poissratio;
        if(!std::isfinite(young)||young<=0||!std::isfinite(poisson)||poisson<=-1||poisson>=.5)
          throw std::runtime_error("Invalid GPU output plane-strain material.");
        const double lambda=young*poisson/((1+poisson)*(1-2*poisson)),mu=young/(2*(1+poisson));
        c.a21=float(lambda);c.a22=float(lambda+2*mu);c.a23=float(lambda);
      }
      const double ev[6]={e[0],e[4],e[8],e[5]+e[7],e[6]+e[2],e[1]+e[3]};
      double sv[6]={};const float* cc=&c.a11;
      for(int i=0;i<6;i++)for(int j=0;j<6;j++)sv[i]+=double(cc[6*i+j])*ev[j];
      const double s[9]={sv[0],sv[5],sv[4],sv[5],sv[1],sv[3],sv[4],sv[3],sv[2]};
      double p[9]={},sigma[9]={};
      for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)p[3*i+j]+=f[3*i+k]*s[3*k+j];
      for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)sigma[3*i+j]+=p[3*i+k]*f[3*j+k];
      for(int i=0;i<3;i++)for(int j=0;j<3;j++){
        const double v=.5*(sigma[3*i+j]+sigma[3*j+i])/det;
        if(!std::isfinite(v))throw std::runtime_error("Non-finite GPU output Cauchy stress.");
        (&out.Stress64.a11)[3*i+j]=v;
      }
    }
    return det;
  }
};

void JSphGpuSingle::SaveUserOutput(
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
        || config.Dynamic.VolumeSPH
        || (firstFrame && (config.Static.Mass || config.Static.Boundary))){
      needFlexMap=true;
      break;
    }
  }

  const unsigned precisionMode=unsigned(UserGParameters::ReadFlex(this->User.get()).PrecisionMode);
  bool needTensors=false,needLive=false;
  for(const auto& c:User.OutputObjects){
    if(!c.Vtk && !c.Csv)continue;
    needTensors|=HasMaterialFields(c.Dynamic);
    needLive|=std::find(c.Mk.begin(),c.Mk.end(),65536u)!=c.Mk.end();
  }


  std::vector<UserGParameterOutput::FlexParticle> flex;
  std::vector<StFlexStrucData> materials;
  std::unordered_map<unsigned,unsigned> flexById;
  if(needFlexMap){
    UserGParameterOutput::DownloadFlex(CaseNflexstruc,Np,precisionMode,FlexPrecision
      ,FlexStrucRidpg,Idp_g->cptr(),Code_g->cptr(),NumPairsg,DefGradg,needTensors,flex);
    materials.resize(FlexStrucCount);
    if(FlexStrucCount){
      const auto err=cudaMemcpy(materials.data(),FlexStrucDatag
        ,sizeof(StFlexStrucData)*FlexStrucCount,cudaMemcpyDeviceToHost);
      if(err!=cudaSuccess)Run_Exceptioon(cudaGetErrorString(err));
    }
    flexById.reserve(flex.size());
    for(unsigned i=0;i<flex.size();i++){
      if(flex[i].Idp==0xffffffffu)Run_Exceptioon("Invalid GPU elastic output particle index.");
      const typecode code=typecode(flex[i].Code);
      if(CODE_IsNormal(code)){
        if(!CODE_IsFlexStrucAny(code))Run_Exceptioon("Invalid GPU flexible output mapping.");
        flexById.emplace(flex[i].Idp,i);
      }
    }
  }
  std::vector<UserGParameterOutput::InletParticle> inlet;
  if(needLive && InOut){
    UserGParameterOutput::DownloadInlet(FlexPrecision,Np,CaseNp,Idp_g->cptr(),Code_g->cptr()
      ,Posxy_g->cptr(),Posz_g->cptr(),Velrho_g->cptr(),inlet);
    // Stable output order despite the compact GPU gather's atomic allocation.
    std::sort(inlet.begin(),inlet.end(),[](const auto& a,const auto& b){return a.Idp<b.Idp;});
  }

  UserParameterOutput userOutput;

  for(const auto& config:User.OutputObjects){
    if(!config.Vtk && !config.Csv)continue;

    const auto& dynamic=config.Dynamic;

    const bool outputStress=HasStress(dynamic);
    const bool outputStrain=!dynamic.StrainComponents.empty();
    const bool outputMaterial=HasMaterialFields(dynamic);

    const bool outputDynamic=
      outputMaterial
      || dynamic.DensitySPH
      || dynamic.VolumeSPH
      || dynamic.PressureSPH
      || dynamic.Velocity;

    const bool outputMass=firstFrame && config.Static.Mass;
    const bool outputBoundary=firstFrame && config.Static.Boundary;
    const bool outputStatic=outputMass || outputBoundary;

    const bool boundaryCorrectionActive=
        outputBoundary &&
        User.FlexRestartBoundaryCorrectionMode!=0;

    // SPH volume requires mass at every output frame.
    const bool needParticleMass=outputMass || dynamic.VolumeSPH;

    if(!outputDynamic && !outputStatic)continue;

    const unsigned inletMarker=65536u;
    const bool selectInlet=
        std::find(config.Mk.begin(),config.Mk.end(),inletMarker)
        !=config.Mk.end();

    if(selectInlet && !InOut)
      Run_Exceptioon("Output selector 'inlet' requires InOut.");

    #ifdef _WITHMR
    if(selectInlet && CfgRun && CfgRun->VRes>0)
      Run_Exceptioon("Output selector 'inlet' does not support VRes particle creation.");
    #endif

    for(unsigned mk:config.Mk){
      if(mk==inletMarker)continue;
      if(MkInfo->GetMkBlockByMk(word(mk))>=MkInfo->Size()){
        Run_Exceptioon(
          fun::PrintStr("Unknown output MK: %u",mk));
      }
    }

    std::vector<UserParameterOutput::StVtkParticle> particles;

    std::unordered_set<unsigned> writtenIds;
    const size_t count=size_t(npsave)+(selectInlet? inlet.size():0);

    for(size_t q=0;q<count;q++){
      const bool live=q>=npsave;
      const unsigned p=live? unsigned(q-npsave):0;
      const unsigned id=live? inlet[p].Idp:svidp[q];

      const bool inletBorn=(InOut && id>=CaseNp);
      const JSphMkBlock* mkblock=nullptr;
      unsigned mk=inletMarker;

      if(inletBorn){
        if(!selectInlet)continue;
      }
      else{
        const unsigned block=MkInfo->GetMkBlockById(id);
        if(block>=MkInfo->Size()){
          Run_Exceptioon(
              fun::PrintStr("Cannot identify MK for Idp=%u.",id));
        }
        mkblock=MkInfo->Mkblock(block);
        mk=mkblock->Mk;
        if(std::find(config.Mk.begin(),config.Mk.end(),mk)
            ==config.Mk.end())continue;
      }

      // Union of selectors and snapshot/live sources: one record per ID.
      if(!writtenIds.insert(id).second)continue;
      const float rhoSPH=live? inlet[p].VelRho.w:svrho[q];

      UserParameterOutput::StVtkParticle particle;

      particle.Idp=id;
      particle.Mk=mk;
      particle.Pos=live? inlet[p].Pos:svpos[q];

      if(dynamic.Velocity){
        if(live){
          const tfloat4 vr=inlet[p].VelRho;
          particle.Velocity=TFloat3(vr.x,vr.y,vr.z);
        }
        else particle.Velocity=svvel[q];
      }

      // Use SPH density from this output frame.
      if(dynamic.PressureSPH){
        particle.PressureSPH=fsph::ComputePress(rhoSPH,CSP);
      }

      if(dynamic.DensitySPH || dynamic.VolumeSPH){
        const double rho=double(rhoSPH);

        if(!std::isfinite(rho) || rho<=0.0){
          Run_Exceptioon(fun::PrintStr(
            "Invalid SPH density: Idp=%u MK=%u rho=%.17g.",
            id,mk,rho));
        }

        if(dynamic.DensitySPH){
          particle.DensitySPH=rho;
        }
      }

      const auto it=flexById.find(id);

      if(outputBoundary && it==flexById.end()){
        Run_Exceptioon(fun::PrintStr(
          "Boundary output requires a flexible or clamp "
          "particle: Idp=%u, MK=%u.",
          id,mk));
      }

      if(outputMaterial && it==flexById.end()){
        Run_Exceptioon(fun::PrintStr(
          "Material output requires a flexible or clamp "
          "particle: Idp=%u, MK=%u.",
          id,mk));
      }

      if(it!=flexById.end()){
        const unsigned pfs=it->second;
        const auto& snapshot=flex[pfs];
        const unsigned body=CODE_GetIbodyFlexStruc(typecode(snapshot.Code));
        if(body>=materials.size())Run_Exceptioon("Invalid GPU flexible body index for output.");
        if(boundaryCorrectionActive)
          Run_Exceptioon("GPU Boundary output requires GPU boundary correction support, which is not migrated yet.");
        const auto* bodyData=FlexStruc->GetBody(body);
        if(!bodyData)Run_Exceptioon("Invalid GPU flexible material for output.");
        const StFlexStrucData& material=materials[body];
        const double naturalDensity=User.IsFlexRestart()?User.FlexRestartState.NaturalDensity[pfs]:double(material.rho0);
        const double naturalVolume=User.IsFlexRestart()?User.FlexRestartState.NaturalVolume[pfs]:double(material.vol0);
        if(needParticleMass)particle.Mass=User.IsFlexRestart()?User.FlexRestartState.ParticleMass[pfs]:naturalDensity*naturalVolume;
        if(outputMaterial){
          if(!snapshot.Pairs)Run_Exceptioon("Cannot compute material fields without structural neighbours.");
          double j=0;
          if(precisionMode){
            tmatrix3d totalH=snapshot.H;
            if(precisionMode==2)totalH=FlexPrecision->TotalH64(pfs,snapshot.H);
            else if(User.IsFlexRestart()){
              const auto& star=User.GetFlexRestartFstar(pfs,id);
              tmatrix3d composed={};
              for(int r=0;r<3;r++)for(int c=0;c<3;c++){
                double h=double((&star.a11)[3*r+c])-(r==c?1.:0.);
                for(int k=0;k<3;k++)h+=(&totalH.a11)[3*r+k]*double((&star.a11)[3*k+c]);
                (&composed.a11)[3*r+c]=h;
              }
              totalH=composed;
            }
            j=UserGOutputMath::Calculate(totalH,material,bodyData->GetConstitModel()
              ,outputStrain,outputStress,particle);
          }
          else{
            const tmatrix3f f=User.IsFlexRestart()?fmath::MulMatrix3x3(snapshot.F,User.GetFlexRestartFstar(pfs,id)):snapshot.F;
            j=fmath::Determinant3x3(f);
            if(outputStrain){
              const tmatrix3f c=fmath::MulMatrix3x3(fmath::TrasMatrix3x3(f),f);
              auto& e=particle.Strain;
              e.a11=.5f*(c.a11-1.f);e.a22=.5f*(c.a22-1.f);e.a33=.5f*(c.a33-1.f);
              e.a12=e.a21=.25f*(c.a12+c.a21);
              e.a13=e.a31=.25f*(c.a13+c.a31);
              e.a23=e.a32=.25f*(c.a23+c.a32);
            }
            if(outputStress)particle.Stress=userOutput.CalcCauchyStressOutput(f,material.cmat
              ,bodyData->GetConstitModel(),material.youngmod,material.poissratio);
          }
          if(!std::isfinite(j) || j<=0)Run_Exceptioon(fun::PrintStr(
            "Invalid GPU user output det(F): Part=%u Idp=%u detF=%.17g.",unsigned(Part),id,j));
          if(dynamic.DensityMaterial)particle.DensityMaterial=naturalDensity/j;
          if(dynamic.VolumeMaterial)particle.VolumeMaterial=naturalVolume*j;
        }
      }
      else if(needParticleMass){
        // Mass of ordinary SPH and rigid-body particles.
        const typecode code=inletBorn? MkInfo->GetCodeNewFluid():mkblock->Code;

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

      if(dynamic.VolumeSPH){
        if(!std::isfinite(particle.Mass) || particle.Mass<=0.0){
          Run_Exceptioon(fun::PrintStr(
            "Invalid particle mass: Idp=%u MK=%u mass=%.17g.",
            id,mk,particle.Mass));
        }

        particle.VolumeSPH=particle.Mass/double(rhoSPH);
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

    if(outputStatic){
      userOutput.SaveFiles(
        basename+"_static",TimeStep,config,particles,true);
    }
  }
}


//==============================================================================
// Device gathers and download adapters.
#include "UserGParameterOutput.h"
#include "UserGFlexPrecision.h"
#include "UserGFlexPrecision.h"
#include <stdexcept>
#include <string>

namespace{
class OutputCuda{
public:
  static void Check(cudaError_t e){
    if(e!=cudaSuccess)throw std::runtime_error(std::string("GPU user output: ")+cudaGetErrorString(e));
  }
};
template<class R,bool enhanced>
__global__ void GatherFlex(unsigned n,unsigned np,const unsigned* rid,const unsigned* idp
  ,const typecode* code,const unsigned* pairs,const tmatrix3f* f
  ,const UserGFlexMatrix<R>* h,bool tensors,UserGParameterOutput::FlexParticle* out){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const unsigned p=rid[i];UserGParameterOutput::FlexParticle a={};
  if(p>=np){a.Idp=0xffffffffu;out[i]=a;return;}
  a.Idp=idp[p];a.Code=unsigned(code[p]);a.Pairs=pairs[i];
  if(tensors){
    a.F=f[i];
    if(enhanced)for(int k=0;k<9;k++)(&a.H.a11)[k]=double(h[i].v[k]);
  }
  out[i]=a;
}
__global__ void GatherInlet(unsigned n,unsigned firstId,const unsigned* idp
  ,const typecode* code,const double2* xy,const double* z,const float4* vr
  ,unsigned* count,UserGParameterOutput::InletParticle* out){
  const unsigned p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
  if(!CODE_IsNormal(code[p]) || !CODE_IsFluid(code[p]) || idp[p]<firstId)return;
  UserGParameterOutput::InletParticle a;
  a.Idp=idp[p];a.Pos={xy[p].x,xy[p].y,z[p]};
  const float4 v=vr[p];a.VelRho={v.x,v.y,v.z,v.w};
  out[atomicAdd(count,1u)]=a;
}
}
void UserGParameterOutput::DownloadFlex(unsigned n,unsigned np,unsigned mode,UserGFlexPrecision* context
  ,const unsigned* rid,const unsigned* idp,const typecode* code
  ,const unsigned* pairs,const tmatrix3f* f,bool tensors,std::vector<FlexParticle>& out){
  out.resize(n);if(!n)return;
  if(!context)throw std::runtime_error("Missing GPU output scratch owner.");
  FlexParticle* buffer=static_cast<FlexParticle*>(context->WorkScratch(sizeof(FlexParticle)*n));
  const unsigned blocks=(n+255)/256;
  if(mode==0 || !tensors)
    GatherFlex<float,false><<<blocks,256>>>(n,np,rid,idp,code,pairs,f,nullptr,tensors,buffer);
  else if(mode==1){
    if(!context)throw std::runtime_error("Missing GPU precision context for output.");
    const auto s=*static_cast<UserGFlexState<float>*>(context->State(mode,n));
    GatherFlex<float,true><<<blocks,256>>>(n,np,rid,idp,code,pairs,f,s.H,tensors,buffer);
  }
  else if(mode==2){
    if(!context)throw std::runtime_error("Missing GPU precision context for output.");
    const auto s=*static_cast<UserGFlexState<double>*>(context->State(mode,n));
    GatherFlex<double,true><<<blocks,256>>>(n,np,rid,idp,code,pairs,f,s.H,tensors,buffer);
  }
  else throw std::runtime_error("Invalid GPU precision mode for output.");
  OutputCuda::Check(cudaGetLastError());
  OutputCuda::Check(cudaMemcpy(out.data(),buffer,sizeof(FlexParticle)*n,cudaMemcpyDeviceToHost));
}
void UserGParameterOutput::DownloadInlet(UserGFlexPrecision* context,unsigned n,unsigned firstId,const unsigned* idp
  ,const typecode* code,const double2* xy,const double* z,const float4* vr
  ,std::vector<InletParticle>& out){
  out.clear();if(!n)return;
  if(!context)throw std::runtime_error("Missing GPU output scratch owner.");
  InletParticle* buffer=static_cast<InletParticle*>(context->WorkScratch(sizeof(InletParticle)*size_t(n)+sizeof(unsigned)));
  unsigned* count=reinterpret_cast<unsigned*>(buffer+n);
  OutputCuda::Check(cudaMemset(count,0,sizeof(unsigned)));
  GatherInlet<<<(n+255)/256,256>>>(n,firstId,idp,code,xy,z,vr,count,buffer);
  OutputCuda::Check(cudaGetLastError());unsigned size=0;
  OutputCuda::Check(cudaMemcpy(&size,count,sizeof(size),cudaMemcpyDeviceToHost));
  if(size>n)throw std::runtime_error("Invalid GPU inlet output count.");
  out.resize(size);
  if(size)OutputCuda::Check(cudaMemcpy(out.data(),buffer,sizeof(InletParticle)*size,cudaMemcpyDeviceToHost));
}
