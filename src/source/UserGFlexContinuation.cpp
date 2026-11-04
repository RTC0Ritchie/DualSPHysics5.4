#include "JSphGpuSingle.h"
#include "UserParameters.h"
#include "UserGFlexPrecision.h"
#include "UserGParameters.h"
#include "JBinaryData.h"
#include "JLog2.h"
#include "Functions.h"
#include "FunctionsMath.h"
#include "FunSphKernel.h"
#include "JDsExtraData.h"
#include "JSphFlexStruc.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// CPU counterpart: UserFlexContinuation.cpp. All downloads occur at
// initialisation or saved frames, never in the time-step force/geometry loop.
class UserGFlexContinuationSnapshot{
public:
  std::vector<unsigned> Id,Ridp;
  std::vector<typecode> Code;
  std::vector<StFlexStrucData> Material;
  std::vector<tmatrix3f> F;
  template<class T> static void Copy(std::vector<T>& out,unsigned n,const T* device){
    out.resize(n);if(!n)return;
    const auto e=cudaMemcpy(out.data(),device,n*sizeof(T),cudaMemcpyDeviceToHost);
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  }
  UserGFlexContinuationSnapshot(unsigned np,const unsigned* ids,const typecode* codes,
      unsigned nb,const StFlexStrucData* mat,unsigned nf=0,const unsigned* rid=nullptr,
      const tmatrix3f* f=nullptr){
    Copy(Id,np,ids);Copy(Code,np,codes);Copy(Material,nb,mat);
    if(rid)Copy(Ridp,nf,rid);if(f)Copy(F,nf,f);
  }
};
void JSphGpuSingle::SaveFlexData(){
  if(!User->SaveFlexData || !FlexStruc)
    return;
  if(!(SvData & SDAT_Binx))
    return;

  const unsigned nf=CaseNflexstruc;
  if(!nf)
    return;


  const bool saveDouble=unsigned(UserGParameters::ReadFlex(User.get()).PrecisionMode)==2;
  std::vector<tmatrix3d> totalH64;
  if(saveDouble)FlexPrecision->DownloadTotalH64(nf,totalH64);
  UserGFlexContinuationSnapshot snapshot(Npb,Idp_g->cptr(),Code_g->cptr(),
      FlexStrucCount,FlexStrucDatag,CaseNflexstruc,FlexStrucRidpg,saveDouble?nullptr:DefGradg);


  const std::string file=
      fun::GetDirWithSlash(DirDataOut)
      +fun::PrintStr("PartFlex_%04u.bi4",unsigned(Part));

  // A v3 frame contains only a source marker, not per-frame material arrays.
  const std::string materialStorage=
      "material";
  if(materialStorage=="material")SaveFlexMaterialDataOnce();

  JBinaryData data("JPartFlexBi4");
  data.SetvUint("FormatVer",3);
  data.SetvText("MaterialStorage",materialStorage);
  data.SetvText("WriterRunCode",RunCode);
  data.SetvUint("Cpart",unsigned(Part));
  data.SetvDouble("TimeStep",TimeStep);
  data.SetvUint("Nflex",nf);

  unsigned* ids=data.CreateArrayUint("Idp",nf,false);
  // Optional metadata: old readers ignore these arrays; old files remain readable.
  unsigned* bodies=data.CreateArrayUint("FlexBody",nf,false);
  unsigned* roles=data.CreateArrayUint("FlexRole",nf,false); // 0=free, 1=clamp
  tfloat3* row1=data.CreateArrayFloat3("Frow1",nf,false);
  tfloat3* row2=data.CreateArrayFloat3("Frow2",nf,false);
  tfloat3* row3=data.CreateArrayFloat3("Frow3",nf,false);

  tdouble3 *hrow1=nullptr,*hrow2=nullptr,*hrow3=nullptr;
  if(saveDouble){
    data.SetvUint("ElasticDoubleStateVer",1);
    data.SetvText("ElasticDoubleStateKind","Htotal_FminusI_natural");
    hrow1=data.CreateArrayDouble3("Hrow1_64",nf,false);
    hrow2=data.CreateArrayDouble3("Hrow2_64",nf,false);
    hrow3=data.CreateArrayDouble3("Hrow3_64",nf,false);
  }
  std::unordered_set<unsigned> seen;
  seen.reserve(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    // Flexible-array index -> main-array index -> stable particle ID.
    const unsigned p=snapshot.Ridp[pfs];

    if(p>=Np)
      Run_Exceptioon("Invalid flexible particle index.");

    const typecode code=snapshot.Code.data()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFlexStrucAny(code))
      Run_Exceptioon("Invalid flexible particle in PartFlex output.");

    const unsigned id=snapshot.Id.data()[p];
    if(!seen.insert(id).second)
      Run_Exceptioon("Duplicated flexible particle ID.");

    ids[pfs]=id;
    bodies[pfs]=CODE_GetIbodyFlexStruc(code);
    roles[pfs]=CODE_IsFlexStrucClamp(code) ? 1u : 0u;

    tmatrix3f f;
    if(saveDouble){
      const auto& h=totalH64[pfs];
      for(int k=0;k<9;k++){
        if(!std::isfinite((&h.a11)[k]))Run_Exceptioon("Non-finite mode-2 H in PartFlex output.");
        (&f.a11)[k]=float((&h.a11)[k]+(k%4==0?1.:0.));
        if(!std::isfinite((&f.a11)[k]))
          Run_Exceptioon("Mode-2 F cannot be represented by the compatible float PartFlex fields.");
      }
      tmatrix3d f64=h;f64.a11+=1;f64.a22+=1;f64.a33+=1;
      const double determinant=fmath::Determinant3x3(f64);
      if(!std::isfinite(determinant) || determinant<=0)
        Run_Exceptioon("Invalid mode-2 deformation determinant in PartFlex output.");
      hrow1[pfs]=TDouble3(h.a11,h.a12,h.a13);
      hrow2[pfs]=TDouble3(h.a21,h.a22,h.a23);
      hrow3[pfs]=TDouble3(h.a31,h.a32,h.a33);
    }
    else f=snapshot.F[pfs];

    if(!saveDouble && User->IsFlexRestart()){
      const tmatrix3f& fstar=
          User->GetFlexRestartFstar(pfs,id);

      f=fmath::MulMatrix3x3(f,fstar);
    }
    row1[pfs]=TFloat3(f.a11,f.a12,f.a13);
    row2[pfs]=TFloat3(f.a21,f.a22,f.a23);
    row3[pfs]=TFloat3(f.a31,f.a32,f.a33);
  }

  data.SaveFile(file,false,true);
}

void JSphGpuSingle::RestoreFlexContinuationClampCodes(){
  if(!PartBegin || !FlexStruc)return;

  UserGFlexContinuationSnapshot snapshot(Npb,Idp_g->cptr(),Code_g->cptr(),
      FlexStrucCount,FlexStrucDatag);
  const std::string file=fun::GetDirWithSlash(PartBeginDir)
      +fun::PrintStr("PartFlex_%04u.bi4",unsigned(PartBegin));
  JBinaryData data;
  data.LoadFile(file);
  const unsigned format=data.GetvUint("FormatVer");
  if(format!=1 && format!=2 && format!=3)
    Run_ExceptioonFile("Unsupported PartFlex format.",file);
  if(data.GetvUint("Cpart")!=unsigned(PartBegin))
    Run_ExceptioonFile("PartFlex frame number does not match.",file);
  const double savedtime=data.GetvDouble("TimeStep");
  if(!std::isfinite(savedtime) ||
      std::fabs(savedtime-TimeStep)>1e-9*(1.0+std::fabs(TimeStep)))
    Run_ExceptioonFile("PartFlex time does not match restart time.",file);
  const unsigned nf=data.GetvUint("Nflex");
  if(nf>Npb)
    Run_ExceptioonFile("PartFlex count exceeds current boundary count.",file);
  const unsigned* ids=static_cast<const unsigned*>(data.GetArrayTpSize(
      "Idp",JBinaryDataDef::DatUint,nf,file)->GetDataPointer());
  const bool hasbody=data.GetArray("FlexBody")!=nullptr;
  const bool hasrole=data.GetArray("FlexRole")!=nullptr;
  if(hasbody!=hasrole)
    Run_ExceptioonFile("Incomplete PartFlex membership metadata.",file);
  const unsigned* bodies=hasbody ? static_cast<const unsigned*>(
      data.GetArrayTpSize("FlexBody",JBinaryDataDef::DatUint,nf,file)
          ->GetDataPointer()) : nullptr;
  const unsigned* roles=hasrole ? static_cast<const unsigned*>(
      data.GetArrayTpSize("FlexRole",JBinaryDataDef::DatUint,nf,file)
          ->GetDataPointer()) : nullptr;

  std::unordered_map<unsigned,unsigned> saved;
  saved.reserve(nf);
  for(unsigned i=0;i<nf;i++){
    if(!saved.emplace(ids[i],i).second)
      Run_ExceptioonFile("Duplicated particle ID in PartFlex.",file);
    if(hasbody && (bodies[i]>=FlexStrucCount || roles[i]>1))
      Run_ExceptioonFile("Invalid PartFlex membership metadata.",file);
  }
  const unsigned* currentids=snapshot.Id.data();
  const typecode* current=Code_c->cptr();
  std::vector<typecode> restored(current,current+Npb);
  std::vector<unsigned char> used(nf,0);
  unsigned nclamp=0,nfree=0;
  for(unsigned p=0;p<Npb;p++){
    const typecode c=current[p];
    const auto it=saved.find(currentids[p]);
    if(!CODE_IsNormal(c)){
      if(CODE_IsPeriodic(c))continue; // regenerated after FlexStrucInit
      Run_ExceptioonFile("Invalid boundary state during clamp restoration.",file);
    }
    if(it==saved.end()){
      if(CODE_IsFlexStrucAny(c))
        Run_ExceptioonFile("Current elastic particle is absent from PartFlex.",file);
      continue; // not a saved clamp: keep the original fixed/moving code
    }
    const unsigned i=it->second;
    if(used[i])
      Run_ExceptioonFile("Duplicated current boundary ID.",file);
    used[i]=1;
    if(CODE_IsFlexStrucFlex(c)){
      if(hasbody && (roles[i]!=0 || bodies[i]!=CODE_GetIbodyFlexStruc(c)))
        Run_ExceptioonFile("Elastic body or role differs from PartFlex.",file);
      nfree++;
      continue;
    }
    if(CODE_IsFlexStrucAny(c) || !(CODE_IsFixed(c)||CODE_IsMoving(c)))
      Run_ExceptioonFile("Saved clamp is not an unclassified boundary particle.",file);
    if(hasrole && roles[i]!=1)
      Run_ExceptioonFile("Saved free elastic particle became a clamp.",file);
    unsigned body=FlexStrucCount,matches=0;
    for(unsigned b=0;b<FlexStrucCount;b++){
      const StFlexStrucData& material=snapshot.Material[b];
      if(std::find(material.clampcode,material.clampcode+material.nc,c)
          !=material.clampcode+material.nc){
        if(!hasbody || bodies[i]==b){body=b;matches++;}
      }
    }
    if(matches!=1)
      Run_ExceptioonFile(hasbody
          ? "Saved clamp body does not match current XML clamp membership."
          : "Legacy PartFlex clamp body is missing or ambiguous in current XML; "
            "do not infer it from deformed distances.",file);
    restored[p]=CODE_ToFlexStrucClamp(c,body);
    nclamp++;
  }
  if(std::find(used.begin(),used.end(),0)!=used.end())
    Run_ExceptioonFile("Saved elastic/clamp ID missing from current boundaries.",file);
  // Commit only after the complete saved identity set has been validated.
  std::copy(restored.begin(),restored.end(),Code_c->ptr());
  Log->Printf("FlexRestart.ClampMembership: restored %u free + %u clamp "
      "particles by stable ID (%s).",nfree,nclamp,
      hasbody ? "saved body/role" : "legacy IDs and unique XML body");
}

void JSphGpuSingle::LoadFlexDirectContinuationState(){
  User->FlexRestartState.Clear();
  User->FlexReferenceId.clear();
  User->FlexRestartWriterRunCode.clear();

  if(!PartBegin || !FlexStruc)
    return;

  const unsigned nf=CaseNflexstruc;
  if(!nf)
    return;

  // Loading restart state does not overwrite the relative snapshot.F.

  UserGFlexContinuationSnapshot snapshot(Npb,Idp_g->cptr(),Code_g->cptr(),
      FlexStrucCount,FlexStrucDatag,CaseNflexstruc,FlexStrucRidpg,nullptr);


  const std::string file=
      fun::GetDirWithSlash(PartBeginDir)
      +fun::PrintStr("PartFlex_%04u.bi4",unsigned(PartBegin));

  JBinaryData data;
  data.LoadFile(file);

  const unsigned format=data.GetvUint("FormatVer");

  if(format!=1 && format!=2 && format!=3)
    Run_ExceptioonFile("Unsupported PartFlex format.",file);

  const std::string referenceId=
      format==2 ? data.GetvText("ReferenceId") : "";
  const std::string writerRunCode=
      format==3 ? data.GetvText("WriterRunCode") : "";
  if(format==3 && writerRunCode.empty())
    Run_ExceptioonFile("PartFlex writer identifier is missing.",file);

  if(data.GetvUint("Cpart")!=unsigned(PartBegin))
    Run_ExceptioonFile("PartFlex frame number does not match.",file);

  if(data.GetvUint("Nflex")!=nf)
    Run_ExceptioonFile("Flexible particle count does not match.",file);

  const double savedtime=data.GetvDouble("TimeStep");
  const double timetol=1e-9*(1.0+std::fabs(TimeStep));
  if(!std::isfinite(savedtime) ||
     std::fabs(savedtime-TimeStep)>timetol)
    Run_ExceptioonFile("PartFlex time does not match restart time.",file);

  // Check array names, types and lengths before accessing their data.
  const unsigned* ids=static_cast<const unsigned*>(
      data.GetArrayTpSize(
          "Idp",JBinaryDataDef::DatUint,nf,file)->GetDataPointer());

  const tfloat3* row1=static_cast<const tfloat3*>(
      data.GetArrayTpSize(
          "Frow1",JBinaryDataDef::DatFloat3,nf,file)->GetDataPointer());

  const tfloat3* row2=static_cast<const tfloat3*>(
      data.GetArrayTpSize(
          "Frow2",JBinaryDataDef::DatFloat3,nf,file)->GetDataPointer());

  const tfloat3* row3=static_cast<const tfloat3*>(
      data.GetArrayTpSize(
          "Frow3",JBinaryDataDef::DatFloat3,nf,file)->GetDataPointer());

  // Optional additive v3 extension. Old CPU readers ignore these arrays.
  // Any partial/unknown extension is an error, never silent float fallback.
  const unsigned doubleVersion=data.GetvUint("ElasticDoubleStateVer",true,0);
  const bool hasH1=data.GetArray("Hrow1_64")!=nullptr;
  const bool hasH2=data.GetArray("Hrow2_64")!=nullptr;
  const bool hasH3=data.GetArray("Hrow3_64")!=nullptr;
  const bool hasDouble=data.ExistsValue("ElasticDoubleStateVer") ||
      data.ExistsValue("ElasticDoubleStateKind") || hasH1 || hasH2 || hasH3;
  const tdouble3 *hrow1=nullptr,*hrow2=nullptr,*hrow3=nullptr;
  if(hasDouble){
    if(doubleVersion!=1 || !hasH1 || !hasH2 || !hasH3 ||
       data.GetvText("ElasticDoubleStateKind",true,"")!="Htotal_FminusI_natural")
      Run_ExceptioonFile("Incomplete or unsupported double PartFlex extension.",file);
    hrow1=static_cast<const tdouble3*>(data.GetArrayTpSize("Hrow1_64",JBinaryDataDef::DatDouble3,nf,file)->GetDataPointer());
    hrow2=static_cast<const tdouble3*>(data.GetArrayTpSize("Hrow2_64",JBinaryDataDef::DatDouble3,nf,file)->GetDataPointer());
    hrow3=static_cast<const tdouble3*>(data.GetArrayTpSize("Hrow3_64",JBinaryDataDef::DatDouble3,nf,file)->GetDataPointer());
  }
  const bool mode2=unsigned(UserGParameters::ReadFlex(User.get()).PrecisionMode)==2;
  std::vector<tmatrix3d> restoredH64(mode2?nf:0);

  // Stable particle ID -> row in the saved file.
  std::unordered_map<unsigned,unsigned> savedindex;
  savedindex.reserve(nf);

  for(unsigned i=0;i<nf;i++){
    if(!savedindex.emplace(ids[i],i).second)
      Run_ExceptioonFile("Duplicated particle ID in PartFlex.",file);
  }

  std::vector<unsigned char> used(nf,0);
  std::vector<unsigned> restoredid(nf);
  std::vector<tmatrix3f> restored(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned p=snapshot.Ridp[pfs];

    if(p>=Np)
      Run_Exceptioon("Invalid flexible particle index.");

    const typecode code=snapshot.Code.data()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFlexStrucAny(code))
      Run_Exceptioon("Invalid flexible particle during restart.");

    const unsigned id=snapshot.Id.data()[p];
    const auto it=savedindex.find(id);

    if(it==savedindex.end())
      Run_ExceptioonFile("Flexible particle ID missing from PartFlex.",file);

    const unsigned i=it->second;
    if(used[i])
      Run_ExceptioonFile("Duplicated current flexible particle ID.",file);
    used[i]=1;

    const tmatrix3f f={
      row1[i].x,row1[i].y,row1[i].z,
      row2[i].x,row2[i].y,row2[i].z,
      row3[i].x,row3[i].y,row3[i].z
    };

    tmatrix3d h;
    if(hasDouble){
      h=TMatrix3d(hrow1[i].x,hrow1[i].y,hrow1[i].z,
          hrow2[i].x,hrow2[i].y,hrow2[i].z,hrow3[i].x,hrow3[i].y,hrow3[i].z);
      tmatrix3d fd=h;fd.a11+=1;fd.a22+=1;fd.a33+=1;
      const double determinant=fmath::Determinant3x3(fd);
      if(!std::isfinite(determinant) || determinant<=0)
        Run_ExceptioonFile("Invalid double PartFlex determinant.",file);
      for(int k=0;k<9;k++){
        const double hv=(&h.a11)[k],fv=(&fd.a11)[k],mirror=(&f.a11)[k];
        const double tolerance=4*std::numeric_limits<float>::epsilon()*std::max(1.,std::fabs(fv));
        if(!std::isfinite(hv) || !std::isfinite(mirror) || std::fabs(fv-mirror)>tolerance)
          Run_ExceptioonFile("Non-finite or inconsistent double/float PartFlex state.",file);
      }
    }
    else for(int k=0;k<9;k++)(&h.a11)[k]=double((&f.a11)[k])-(k%4==0?1.:0.);
    if(mode2)restoredH64[pfs]=h;
    restoredid[pfs]=id;
    restored[pfs]=f;
  }

  for(unsigned i=0;i<nf;i++){
    if(!used[i])
      Run_ExceptioonFile("Unmatched particle ID in PartFlex.",file);
  }

  UserParameters::StFlexRestartState state;

  state.Idp.swap(restoredid);
  state.Fstar.swap(restored);

  if(mode2)FlexPrecision->SetRestartH64(nf,restoredH64.data());
  Log->Print(mode2?(hasDouble?"FlexRestart.Precision: inherited double H.":
      "FlexRestart.Precision: promoted legacy float F; lost precision cannot be recovered."):
      "FlexRestart.Precision: using compatible float F for mode 0/1.");
  User->FlexRestartState=state;
  User->FlexReferenceId=referenceId;
  User->FlexRestartWriterRunCode=writerRunCode;

  const std::string storage=format==3
      ? data.GetvText("MaterialStorage",true,"legacy") : "legacy";
  if(storage=="material" || storage=="pair")LoadFlexMaterialData(storage);
  else if(storage=="legacy")
    Log->Print("WARNING: legacy PartFlex has no material-source marker; "
        "using XML natural-state defaults. Nondefault User.restart material "
        "conventions cannot be reconstructed from total F alone.");
  else if(storage!="xml")
    Run_ExceptioonFile("Invalid PartFlex material source.",file);

  Log->Printf(
      "FlexRestart.DirectContinuation: deformation state loaded for "
      "%u particles.",nf);
}

void JSphGpuSingle::SaveFlexMaterialDataOnce(){
  if(User->FlexMaterialDataWritten)return;
  const auto& state=User->FlexRestartState;
  const unsigned nf=CaseNflexstruc;
  UserGFlexContinuationSnapshot snapshot(Npb,Idp_g->cptr(),Code_g->cptr(),
      FlexStrucCount,FlexStrucDatag,CaseNflexstruc,FlexStrucRidpg,nullptr);

  const bool restart=User->IsFlexRestart();
  if(restart && (!state.HasRestartStateLayout() || state.Idp.size()!=nf))
    Run_Exceptioon("Material output requires a complete elastic restart state.");
  const std::string file=fun::GetDirWithSlash(DirDataOut)+"FlexMaterialData.ibi4";
  JBinaryData data("JFlexMaterialDataBi4");
  data.SetvUint("FormatVer",1);
  data.SetvText("WriterRunCode",RunCode);
  data.SetvUint("Nflex",nf);
  unsigned* ids=data.CreateArrayUint("Idp",nf,false);
  float* vn=data.CreateArrayFloat("NaturalVol",nf,false);
  float* rn=data.CreateArrayFloat("NaturalRho",nf,false);
  float* mass=data.CreateArrayFloat("ParticleMass",nf,false);
  for(unsigned i=0;i<nf;i++){
    const unsigned p=snapshot.Ridp[i];
    if(p>=Np)Run_Exceptioon("Invalid material-output particle index.");
    ids[i]=snapshot.Id.data()[p];
    if(restart){
      if(ids[i]!=state.Idp[i])
        Run_Exceptioon("Material-output particle identity mismatch.");
      vn[i]=state.NaturalVolume[i];
      rn[i]=state.NaturalDensity[i];
      mass[i]=state.ParticleMass[i];
    }
    else{
      const auto& mat=snapshot.Material[CODE_GetIbodyFlexStruc(snapshot.Code.data()[p])];
      vn[i]=mat.vol0;
      rn[i]=mat.rho0;
      mass[i]=vn[i]*rn[i];
    }
  }
  data.SaveFile(file,false,true);
  User->FlexMaterialDataWritten=true;
}

void JSphGpuSingle::LoadFlexMaterialData(const std::string& storage){
  const bool pair=storage=="pair";
  const std::string file=fun::GetDirWithSlash(PartBeginDir)
      +(pair ? "FlexPairData.ibi4" : "FlexMaterialData.ibi4");
  if(!fun::FileExists(file))
    Run_ExceptioonFile("Required elastic material file is missing.",file);
  JBinaryData data;
  data.LoadFile(file);
  if(data.GetvUint("FormatVer")!=1)
    Run_ExceptioonFile("Unsupported elastic material file format.",file);
  if(User->FlexRestartWriterRunCode.empty() ||
     data.GetvText("WriterRunCode")!=User->FlexRestartWriterRunCode)
    Run_ExceptioonFile("PartFlex and material data belong to different runs.",file);
  const unsigned nf=CaseNflexstruc;
  if(data.GetvUint("Nflex")!=nf)
    Run_ExceptioonFile("Elastic material particle count differs.",file);
  const unsigned* ids=static_cast<const unsigned*>(data.GetArrayTpSize(
      "Idp",JBinaryDataDef::DatUint,nf,file)->GetDataPointer());
  const float* vn=static_cast<const float*>(data.GetArrayTpSize(
      "NaturalVol",JBinaryDataDef::DatFloat,nf,file)->GetDataPointer());
  const float* rn=static_cast<const float*>(data.GetArrayTpSize(
      "NaturalRho",JBinaryDataDef::DatFloat,nf,file)->GetDataPointer());
  const float* mass=static_cast<const float*>(data.GetArrayTpSize(
      "ParticleMass",JBinaryDataDef::DatFloat,nf,file)->GetDataPointer());
  std::unordered_map<unsigned,unsigned> index;
  for(unsigned i=0;i<nf;i++)
    if(!index.emplace(ids[i],i).second)
      Run_ExceptioonFile("Duplicated material particle ID.",file);
  auto state=User->FlexRestartState;
  state.NaturalVolume.resize(nf);
  state.NaturalDensity.resize(nf);
  state.ParticleMass.resize(nf);
  for(unsigned i=0;i<nf;i++){
    const auto it=index.find(state.Idp[i]);
    if(it==index.end())
      Run_ExceptioonFile("Missing material particle ID.",file);
    const unsigned k=it->second;
    if(!std::isfinite(vn[k]) || vn[k]<=0 || !std::isfinite(rn[k]) || rn[k]<=0 ||
       !std::isfinite(mass[k]) || mass[k]<=0)
      Run_ExceptioonFile("Invalid elastic material invariants.",file);
    const double expected=double(vn[k])*rn[k];
    const double tolerance=32.0*std::numeric_limits<float>::epsilon()
        *std::max(expected,double(mass[k]));
    if(std::fabs(double(mass[k])-expected)>tolerance)
      Run_ExceptioonFile("Inconsistent elastic particle mass.",file);
    state.NaturalVolume[i]=vn[k];
    state.NaturalDensity[i]=rn[k];
    state.ParticleMass[i]=mass[k];
  }
  state.HasMaterialState=true;
  User->FlexRestartState=state;
  Log->Printf("FlexRestart: material invariants loaded from %s.",file.c_str());
}

void JSphGpuSingle::LoadBoundaryMotionRestart(){
  if(!PartBegin || !FlexStruc || TMdbc2<MDBC2_Std)return;
  if(!JDsExtraDataLoad::ExistsPartData(PartBeginDir,int(PartBegin)))
    Run_Exceptioon("Missing extra data for GPU boundary motion restart.");
  std::vector<unsigned> ids;
  std::vector<typecode> code;
  UserGFlexContinuationSnapshot::Copy(ids,Np,Idp_g->cptr());
  UserGFlexContinuationSnapshot::Copy(code,Np,Code_g->cptr());
  std::vector<tfloat3> vel(Np),ace(Np);
  JDsExtraDataLoad data(CaseNbound,CaseNfloat,Log);
  data.LoadPartData(PartBeginDir,int(PartBegin));
  data.LoadBoundaryMotion(Np,ids.data(),code.data(),vel.data(),ace.data());
  if(cudaMemcpy(MotionVel_g->ptr(),vel.data(),Np*sizeof(tfloat3),cudaMemcpyHostToDevice)!=cudaSuccess ||
     cudaMemcpy(MotionAce_g->ptr(),ace.data(),Np*sizeof(tfloat3),cudaMemcpyHostToDevice)!=cudaSuccess)
    Run_Exceptioon("GPU boundary motion upload failed.");
}

void JSphGpuSingle::SaveBoundaryMotionRestart(){
  if(!FlexStruc || TMdbc2<MDBC2_Std)return;
  std::vector<unsigned> ids;
  std::vector<typecode> code;
  std::vector<tfloat3> vel,ace;
  UserGFlexContinuationSnapshot::Copy(ids,Np,Idp_g->cptr());
  UserGFlexContinuationSnapshot::Copy(code,Np,Code_g->cptr());
  // float3/tfloat3 share the same packed representation.
  vel.resize(Np);ace.resize(Np);
  if(cudaMemcpy(vel.data(),MotionVel_g->cptr(),Np*sizeof(tfloat3),cudaMemcpyDeviceToHost)!=cudaSuccess ||
     cudaMemcpy(ace.data(),MotionAce_g->cptr(),Np*sizeof(tfloat3),cudaMemcpyDeviceToHost)!=cudaSuccess)
    Run_Exceptioon("GPU boundary motion download failed.");
  SvExtraDataBi4->AddBoundaryMotion(Np,ids.data(),code.data(),vel.data(),ace.data());
}

void JSphGpuSingle::ConfigFlexRestartMode(){
  if(User->FlexRestartRestorePairs!=User->FlexRestartGradientCorrection ||
     User->FlexRestartRestorePairs!=User->FlexRestartHourglassCorrection)
    Run_Exceptioon("GPU directcontinuation requires restorepairs, gradientcorrection and hourglasscorrection all true or all false.");
  User->ConfigFlexRestartMode(PartBegin!=0,FlexStruc && FlexStruc->GetCount()>0,Log);
}

void JSphGpuSingle::LoadFlexDirectContinuationPairData()
{
  if(!PartBegin || !FlexStruc)
    return;

  const bool required=
      User->UseFlexRestartRestorePairs ||
      User->UseFlexRestartGradientCorrection ||
      User->UseFlexRestartHourglassCorrection;
  const bool exportPairData=User->SaveFlexPairData && (SvData & SDAT_Binx);
  if(!required && !exportPairData)return;

  // v3 frames use the independent pair file from the same writing run.
  // Legacy v2 frames retain their explicit ReferenceId association.
  const bool modern=!User->FlexRestartWriterRunCode.empty();
  if(!modern && User->FlexReferenceId.empty())
    Run_Exceptioon(
        "Direct continuation: the legacy PartFlex frame has no reference "
        "association required for kernel inheritance or SaveFlexPairData.");

  const std::string file=fun::GetDirWithSlash(PartBeginDir)
      +(modern ? "FlexPairData.ibi4" : "FlexReference.ibi4");
  if(!fun::FileExists(file))
    Run_ExceptioonFile(
        "Direct continuation: reference kernel data are required for "
        "the requested inheritance or SaveFlexPairData export.",file);

  JBinaryData data;
  data.LoadFile(file);
  const unsigned pairFormat=data.GetvUint("FormatVer");
  if(pairFormat!=1 && pairFormat!=2)
    Run_ExceptioonFile("Unsupported flexible pair-data format.",file);
  if(pairFormat==2 && !modern)
    Run_ExceptioonFile("Split pair data require a v3 frame association.",file);

  const std::string referenceId=data.GetvText("ReferenceId");
  if(referenceId.empty())
    Run_ExceptioonFile("Missing natural-reference identifier.",file);
  if(modern){
    if(data.GetvText("WriterRunCode")!=User->FlexRestartWriterRunCode)
      Run_ExceptioonFile(
          "PartFlex and FlexPairData were written by different runs.",file);
  }
  else if(referenceId!=User->FlexReferenceId)
    Run_ExceptioonFile(
        "PartFlex and legacy natural-reference data do not belong together.",file);

  const unsigned nf=CaseNflexstruc;
  const unsigned count=data.GetvUint("Npairs");

  if(data.GetvUint("Nflex")!=nf)
    Run_ExceptioonFile(
        "Flexible reference particle count differs.",file);

  const auto get=
      [&](const char* name,JBinaryDataDef::TpData type,size_t n)
  {
    return data.GetArrayTpSize(name,type,n,file)
        ->GetDataPointer();
  };

  const auto ids=static_cast<const unsigned*>(
      get("Idp",JBinaryDataDef::DatUint,nf));

  const auto xn=static_cast<const tdouble3*>(
      get("NaturalPos",JBinaryDataDef::DatDouble3,nf));

  // v1 stores material arrays inline; v2 uses the independent material file.
  JBinaryData material;
  std::unordered_map<unsigned,unsigned> materialIndex;
  const float* vn=nullptr;
  const float* rn=nullptr;
  const float* mass=nullptr;
  if(pairFormat==1){
    vn=static_cast<const float*>(get("NaturalVol",JBinaryDataDef::DatFloat,nf));
    rn=static_cast<const float*>(get("NaturalRho",JBinaryDataDef::DatFloat,nf));
  }
  else{
    const std::string mf=fun::GetDirWithSlash(PartBeginDir)+"FlexMaterialData.ibi4";
    if(!fun::FileExists(mf))
      Run_ExceptioonFile("Split pair data require FlexMaterialData.ibi4.",mf);
    material.LoadFile(mf);
    if(material.GetvUint("FormatVer")!=1 ||
       material.GetvUint("Nflex")!=nf ||
       material.GetvText("WriterRunCode")!=User->FlexRestartWriterRunCode)
      Run_ExceptioonFile("Material file does not match split pair data.",mf);
    const auto getMaterial=[&](const char* name,JBinaryDataDef::TpData type){
      return material.GetArrayTpSize(name,type,nf,mf)->GetDataPointer();
    };
    const auto mid=static_cast<const unsigned*>(getMaterial("Idp",JBinaryDataDef::DatUint));
    vn=static_cast<const float*>(getMaterial("NaturalVol",JBinaryDataDef::DatFloat));
    rn=static_cast<const float*>(getMaterial("NaturalRho",JBinaryDataDef::DatFloat));
    mass=static_cast<const float*>(getMaterial("ParticleMass",JBinaryDataDef::DatFloat));
    for(unsigned i=0;i<nf;i++)
      if(!materialIndex.emplace(mid[i],i).second)
        Run_ExceptioonFile("Duplicated material particle ID.",mf);
  }

  const auto b1=static_cast<const tfloat3*>(
      get("Brow1",JBinaryDataDef::DatFloat3,nf));

  const auto b2=static_cast<const tfloat3*>(
      get("Brow2",JBinaryDataDef::DatFloat3,nf));

  const auto b3=static_cast<const tfloat3*>(
      get("Brow3",JBinaryDataDef::DatFloat3,nf));

  const auto offsets=static_cast<const unsigned*>(
      get("PairOffset",JBinaryDataDef::DatUint,size_t(nf)+1));

  const auto pairs=static_cast<const unsigned*>(
      get("PairIdp",JBinaryDataDef::DatUint,count));

  std::unordered_map<unsigned,unsigned> savedIndex;
  savedIndex.reserve(nf);

  for(unsigned k=0;k<nf;k++){
    if(!savedIndex.emplace(ids[k],k).second)
      Run_ExceptioonFile(
          "FlexRestart.DirectContinuation: duplicated natural-reference particle ID.",file);
  }

  if(offsets[0]!=0 || offsets[nf]!=count)
    Run_ExceptioonFile("Invalid reference pair offsets.",file);

  for(unsigned k=0;k<nf;k++){
    if(offsets[k]>offsets[k+1] || offsets[k+1]>count)
      Run_ExceptioonFile("Invalid reference pair offsets.",file);

    std::unordered_set<unsigned> seen;

    for(unsigned q=offsets[k];q<offsets[k+1];q++){
      if(pairs[q]==ids[k] ||
         savedIndex.find(pairs[q])==savedIndex.end() ||
         !seen.insert(pairs[q]).second)
      {
        Run_ExceptioonFile(
            "Invalid reference neighbour ID.",file);
      }
    }
  }

  // Transactional update: commit after validation.
  auto state=User->FlexRestartState;

  state.ReferenceKernel=data.GetvUint("Kernel");
  state.ReferenceH=data.GetvFloat("H");
  state.Reference2D=data.GetvBool("Simulate2D");

  if(!std::isfinite(state.ReferenceH) ||
     state.ReferenceH<=0)
  {
    Run_ExceptioonFile("Invalid reference smoothing length.",file);
  }

  const bool needsKernel=
      User->UseFlexRestartGradientCorrection ||
      User->UseFlexRestartHourglassCorrection;

  if(needsKernel &&
     (state.ReferenceKernel!=unsigned(TKernel) ||
      state.ReferenceH!=KernelH ||
      state.Reference2D!=Simulate2D))
  {
    Run_ExceptioonFile(
        "FlexRestart: kernel settings differ from natural-reference data.",
        file);
  }

  state.NaturalPos.resize(nf);
  state.NaturalKerCorr.resize(nf);
  state.PairOffset.resize(size_t(nf)+1);
  state.PairIdp.clear();
  state.PairIdp.reserve(count);
  state.PairOffset[0]=0;

  for(unsigned pfs=0;pfs<nf;pfs++){
    const auto it=savedIndex.find(state.Idp[pfs]);

    if(it==savedIndex.end())
      Run_ExceptioonFile(
          "FlexRestart.DirectContinuation: particle ID is absent from natural-reference data.",
          file);

    const unsigned k=it->second;

    if(!std::isfinite(xn[k].x) ||
       !std::isfinite(xn[k].y) ||
       !std::isfinite(xn[k].z))
    {
      Run_ExceptioonFile("Invalid natural position.",file);
    }

    unsigned mk=k;
    if(pairFormat==2){
      const auto mi=materialIndex.find(state.Idp[pfs]);
      if(mi==materialIndex.end())
        Run_ExceptioonFile("Particle ID missing from split material data.",file);
      mk=mi->second;
    }
    if(!std::isfinite(vn[mk]) || vn[mk]<=0 ||
       !std::isfinite(rn[mk]) || rn[mk]<=0 ||
       vn[mk]!=state.NaturalVolume[pfs] ||
       rn[mk]!=state.NaturalDensity[pfs] ||
       (mass && mass[mk]!=state.ParticleMass[pfs]))
    {
      Run_ExceptioonFile(
          "FlexRestart: natural volume/density differs from natural-reference data.",
          file);
    }

    const tmatrix3f b={
      b1[k].x,b1[k].y,b1[k].z,
      b2[k].x,b2[k].y,b2[k].z,
      b3[k].x,b3[k].y,b3[k].z
    };

    const float values[]={
      b.a11,b.a12,b.a13,
      b.a21,b.a22,b.a23,
      b.a31,b.a32,b.a33
    };

    for(float value:values){
      if(!std::isfinite(value))
        Run_ExceptioonFile(
            "Invalid natural kernel correction matrix.",file);
    }

    state.NaturalPos[pfs]=xn[k];
    state.NaturalKerCorr[pfs]=b;

    for(unsigned q=offsets[k];q<offsets[k+1];q++)
      state.PairIdp.push_back(pairs[q]);

    state.PairOffset[pfs+1]=
        unsigned(state.PairIdp.size());
  }

  // Optional mode-2 natural B. Old CPU readers retain their float B rows.
  const bool has64=data.ExistsValue("NaturalCorrection64Version") ||
      data.GetArray("Brow1_64") || data.GetArray("Brow2_64") || data.GetArray("Brow3_64");
  std::vector<tmatrix3d> b64;
  if(has64){
    if(data.GetvUint("NaturalCorrection64Version",true,0)!=1)
      Run_ExceptioonFile("Unsupported double natural correction format.",file);
    const auto d1=static_cast<const tdouble3*>(get("Brow1_64",JBinaryDataDef::DatDouble3,nf));
    const auto d2=static_cast<const tdouble3*>(get("Brow2_64",JBinaryDataDef::DatDouble3,nf));
    const auto d3=static_cast<const tdouble3*>(get("Brow3_64",JBinaryDataDef::DatDouble3,nf));
    b64.resize(nf);
    for(unsigned i=0;i<nf;i++){
      const unsigned k=savedIndex.at(state.Idp[i]);
      b64[i]=TMatrix3d(d1[k].x,d1[k].y,d1[k].z,d2[k].x,d2[k].y,d2[k].z,d3[k].x,d3[k].y,d3[k].z);
      for(unsigned q=0;q<9;q++){
        const double v=(&b64[i].a11)[q],f=(&state.NaturalKerCorr[i].a11)[q];
        if(!std::isfinite(v) || std::fabs(v-f)>4*std::numeric_limits<float>::epsilon()*std::max(1.,std::fabs(v)))
          Run_ExceptioonFile("Invalid double natural correction or inconsistent float mirror.",file);
      }
    }
  }
  FlexPrecision->SetNaturalCorrection64(b64);
  state.HasNaturalReferenceData=true;
  User->FlexRestartState=state;
  User->FlexReferenceId=referenceId;
  Log->Printf("FlexRestart: reference kernel data loaded from %s.",file.c_str());
}

void JSphGpuSingle::SaveFlexPairDataOnce()
{
  if(!User->SaveFlexPairData || !FlexStruc || !CaseNflexstruc)
    return;
  if(!(SvData & SDAT_Binx))
    return;
  if(User->FlexPairDataWritten)
    return;

  UserGFlexContinuationSnapshot snapshot(Npb,Idp_g->cptr(),Code_g->cptr(),
      FlexStrucCount,FlexStrucDatag,CaseNflexstruc,FlexStrucRidpg,nullptr);
  std::vector<tmatrix3f> correction;
  std::vector<unsigned> counts,pairs;
  std::vector<double2> xy;std::vector<double> z;
  if(!User->IsFlexRestart() || User->IsFlexUndeformedRestart()){
    if(Part!=PartIni)Run_Exceptioon("Natural reference export must start at the initial frame.");
    UserGFlexContinuationSnapshot::Copy(correction,CaseNflexstruc,KerCorrg);
    UserGFlexContinuationSnapshot::Copy(counts,CaseNflexstruc,NumPairsg);
    size_t count=0;for(auto n:counts)count+=n;
    if(count>std::numeric_limits<unsigned>::max())Run_Exceptioon("Too many natural pairs.");
    UserGFlexContinuationSnapshot::Copy(pairs,unsigned(count),PairIdxBufferg);
    UserGFlexContinuationSnapshot::Copy(xy,Npb,Posxy_g->cptr());
    UserGFlexContinuationSnapshot::Copy(z,Npb,Posz_g->cptr());
  }
  const bool undeformed=User->IsFlexUndeformedRestart();
  if((User->IsFlexPrescribedRestart() || User->IsFlexModelTransferRestart())
      && !undeformed)
    Run_Exceptioon("SaveFlexPairData is forbidden for deformed User.restart.");

  UserParameters::StFlexRestartState reference;

  if(User->IsFlexRestart() && !undeformed){
    if(!User->FlexRestartState.HasNaturalReferenceLayout()){
      Run_Exceptioon(
          "SaveFlexPairData requires original natural-reference geometry "
          "and kernel data; current restart geometry cannot replace them.");
    }

    reference=User->FlexRestartState;
  }
  else{
    const unsigned nf=CaseNflexstruc;

    reference.Idp.resize(nf);
    reference.NaturalPos.resize(nf);
    reference.NaturalKerCorr.resize(nf);
    reference.NaturalVolume.resize(nf);
    reference.NaturalDensity.resize(nf);
    reference.PairOffset.resize(size_t(nf)+1);

    reference.ReferenceKernel=unsigned(TKernel);
    reference.ReferenceH=KernelH;
    reference.Reference2D=Simulate2D;

    reference.PairOffset[0]=0;

    for(unsigned pfs=0;pfs<nf;pfs++){
      const unsigned p=snapshot.Ridp[pfs];

      if(p>=Np)
        Run_Exceptioon("Invalid flexible particle index.");

      const unsigned body=
          CODE_GetIbodyFlexStruc(snapshot.Code[p]);

      reference.Idp[pfs]=snapshot.Id[p];
      reference.NaturalPos[pfs]=TDouble3(xy[p].x,xy[p].y,z[p]);
      reference.NaturalKerCorr[pfs]=correction[pfs];
      reference.NaturalVolume[pfs]=snapshot.Material[body].vol0;
      reference.NaturalDensity[pfs]=snapshot.Material[body].rho0;

      const unsigned previous=reference.PairOffset[pfs];
      const unsigned count=counts[pfs];

      if(count>std::numeric_limits<unsigned>::max()-previous)
        Run_Exceptioon("Too many flexible reference pairs.");

      reference.PairOffset[pfs+1]=previous+count;
    }

    reference.PairIdp.resize(reference.PairOffset[nf]);

    for(unsigned pfs=0;pfs<nf;pfs++){
      for(unsigned k=0;k<counts[pfs];k++){
        const unsigned neighbour=pairs[reference.PairOffset[pfs]+k];

        if(neighbour>=nf)
          Run_Exceptioon("Invalid flexible neighbour index.");

        reference.PairIdp[reference.PairOffset[pfs]+k]=
            snapshot.Id[snapshot.Ridp[neighbour]];
      }
    }

    reference.HasNaturalReferenceData=true;

    // Existing solver run identifier.
    User->FlexReferenceId=RunCode;
  }

  if(User->FlexReferenceId.empty())
    Run_Exceptioon("Flexible reference identifier is unavailable.");

  const unsigned nf=unsigned(reference.Idp.size());

  const std::string file=
      fun::GetDirWithSlash(DirDataOut)+"FlexPairData.ibi4";

  // Pair output also requires the shared material file, even without frames.
  SaveFlexMaterialDataOnce();
  JBinaryData data("JFlexPairDataBi4");

  data.SetvUint("FormatVer",2);
  data.SetvText("WriterRunCode",RunCode);
  data.SetvText("ReferenceId",User->FlexReferenceId);

  data.SetvUint("Nflex",nf);
  data.SetvUint(
      "Npairs",unsigned(reference.PairIdp.size()));

  data.SetvUint("Kernel",reference.ReferenceKernel);
  data.SetvFloat("H",reference.ReferenceH);
  data.SetvBool("Simulate2D",reference.Reference2D);

  unsigned* ids=data.CreateArrayUint("Idp",nf,false);

  tdouble3* xn=
      data.CreateArrayDouble3("NaturalPos",nf,false);


  tfloat3* b1=data.CreateArrayFloat3("Brow1",nf,false);
  tfloat3* b2=data.CreateArrayFloat3("Brow2",nf,false);
  tfloat3* b3=data.CreateArrayFloat3("Brow3",nf,false);

  unsigned* offsets=data.CreateArrayUint(
      "PairOffset",size_t(nf)+1,false);

  unsigned* storedPairs=data.CreateArrayUint(
      "PairIdp",reference.PairIdp.size(),false);

  for(unsigned pfs=0;pfs<nf;pfs++){
    ids[pfs]=reference.Idp[pfs];
    xn[pfs]=reference.NaturalPos[pfs];

    const tmatrix3f& b=reference.NaturalKerCorr[pfs];

    b1[pfs]=TFloat3(b.a11,b.a12,b.a13);
    b2[pfs]=TFloat3(b.a21,b.a22,b.a23);
    b3[pfs]=TFloat3(b.a31,b.a32,b.a33);
  }

  std::copy(
      reference.PairOffset.begin(),
      reference.PairOffset.end(),
      offsets);

  std::copy(
      reference.PairIdp.begin(),
      reference.PairIdp.end(),
      storedPairs);

  std::vector<tmatrix3d> b64;
  if(User->IsFlexRestart() && !User->IsFlexUndeformedRestart())b64=FlexPrecision->NaturalCorrection64();
  else if(unsigned(UserGParameters::ReadFlex(User.get()).PrecisionMode)==2)
    FlexPrecision->DownloadCorrection64(nf,b64);
  if(!b64.empty()){
    if(b64.size()!=nf)Run_Exceptioon("Natural double correction count mismatch.");
    data.SetvUint("NaturalCorrection64Version",1);
    auto a=data.CreateArrayDouble3("Brow1_64",nf,false);
    auto b=data.CreateArrayDouble3("Brow2_64",nf,false);
    auto c=data.CreateArrayDouble3("Brow3_64",nf,false);
    for(unsigned i=0;i<nf;i++){
      const auto& v=b64[i];a[i]=TDouble3(v.a11,v.a12,v.a13);
      b[i]=TDouble3(v.a21,v.a22,v.a23);c[i]=TDouble3(v.a31,v.a32,v.a33);
    }
  }
  data.SaveFile(file,false,true);

  // Mark only after successful writing.
  User->FlexPairDataWritten=true;
}

// Host-side preparation only. CPU counterpart: RestoreFlexRestartPairs and
// PrepareFlexRestart{Gradient,Hourglass}Correction in UserFlexContinuation.cpp.
class UserGFlexCorrectionMath{
public:
  static tmatrix3d Transpose(const tmatrix3d& a){
    tmatrix3d b;for(int r=0;r<3;r++)for(int c=0;c<3;c++)(&b.a11)[3*r+c]=(&a.a11)[3*c+r];return b;
  }
  static tmatrix3d Multiply(const tmatrix3d& a,const tmatrix3d& b){
    tmatrix3d d={};for(int r=0;r<3;r++)for(int c=0;c<3;c++)for(int k=0;k<3;k++)
      (&d.a11)[3*r+c]+=(&a.a11)[3*r+k]*(&b.a11)[3*k+c];return d;
  }
  static tdouble3 MV(const tmatrix3d& a,const tdouble3& b){
    return TDouble3(a.a11*b.x+a.a12*b.y+a.a13*b.z,
      a.a21*b.x+a.a22*b.y+a.a23*b.z,a.a31*b.x+a.a32*b.y+a.a33*b.z);
  }
  static tfloat3 MV(const tmatrix3f& a,const tfloat3& b){
    return TFloat3(a.a11*b.x+a.a12*b.y+a.a13*b.z,
      a.a21*b.x+a.a22*b.y+a.a23*b.z,a.a31*b.x+a.a32*b.y+a.a33*b.z);
  }
  static tdouble3 Scale(tdouble3 a,double v){return TDouble3(a.x*v,a.y*v,a.z*v);}
  static tdouble3 Promote(tfloat3 a){return TDouble3(a.x,a.y,a.z);}
  static tmatrix3d Promote(const tmatrix3f& a){
    tmatrix3d b;for(unsigned k=0;k<9;k++)(&b.a11)[k]=(&a.a11)[k];return b;
  }
  static double Fac64(TpKernel kernel,const StCteSph& c,double rr){
    const double r=sqrt(rr),q=r/double(c.kernelh);
    if(kernel==KERNEL_Wendland){const double a=1-.5*q;return double(c.kwend.bwenh)*a*a*a;}
    if(r>double(c.kernelh)){const double a=2-q;return double(c.kcubic.c2)*a*a/r;}
    return rr?(double(c.kcubic.c1)*q+double(c.kcubic.d1)*q*q)/r:double(c.kcubic.c1)/double(c.kernelh);
  }
};

unsigned JSphGpuSingle::InitFlexRestartReferenceAndCorrections(unsigned oldPairs){
  if(!User->IsFlexRestart())return oldPairs;
  const bool corrected=User->UseFlexRestartRestorePairs;
  if(!corrected && !(User->SaveFlexPairData && (SvData & SDAT_Binx)))return oldPairs;
  LoadFlexDirectContinuationPairData();
  if(!corrected)return oldPairs; // Preserve provenance for export, but do not inherit operators.
  const auto& h=User->FlexRestartState;
  const unsigned n=CaseNflexstruc,count=unsigned(h.PairIdp.size());
  if(!h.HasNaturalReferenceLayout())Run_Exceptioon("Missing natural reference for GPU corrections.");
  if(TKernel!=KERNEL_Wendland && TKernel!=KERNEL_Cubic)Run_Exceptioon("Unsupported correction kernel.");
  const unsigned mode=unsigned(UserGParameters::ReadFlex(User.get()).PrecisionMode);
  const bool wide=mode==2;
  const auto& natural64=FlexPrecision->NaturalCorrection64();
  UserGFlexContinuationSnapshot snapshot(Npb,Idp_g->cptr(),Code_g->cptr(),
      FlexStrucCount,FlexStrucDatag,n,FlexStrucRidpg,nullptr);
  std::unordered_map<unsigned,unsigned> index;
  std::vector<unsigned> counts(n),pairs(count);
  for(unsigned i=0;i<n;i++){
    if(snapshot.Id[snapshot.Ridp[i]]!=h.Idp[i] || !index.emplace(h.Idp[i],i).second)
      Run_Exceptioon("GPU natural reference ID mismatch.");
    counts[i]=h.PairOffset[i+1]-h.PairOffset[i];
    if(!counts[i])Run_Exceptioon("GPU gradient inheritance requires neighbours for every elastic particle.");
  }
  for(unsigned k=0;k<count;k++)pairs[k]=index.at(h.PairIdp[k]);
  std::vector<double2> xy;std::vector<double> z;
  UserGFlexContinuationSnapshot::Copy(xy,Npb,Posxy_g->cptr());
  UserGFlexContinuationSnapshot::Copy(z,Npb,Posz_g->cptr());
  std::vector<tmatrix3d> star(n),inv(n),transform(n),offset(n);
  std::vector<tmatrix3f> transformf(n);
  std::vector<tdouble3> u0(n),gi(count),gj(count),mi(count),mj(count);
  std::vector<double> weight(count);
  for(unsigned i=0;i<n;i++){
    const unsigned p=snapshot.Ridp[i];
    u0[i]=TDouble3(xy[p].x-h.NaturalPos[i].x,xy[p].y-h.NaturalPos[i].y,z[p]-h.NaturalPos[i].z);
    tmatrix3d hs={};
    if(wide)hs=FlexPrecision->TotalH64(i,hs);
    else{hs=UserGFlexCorrectionMath::Promote(h.Fstar[i]);hs.a11-=1;hs.a22-=1;hs.a33-=1;}
    star[i]=hs;star[i].a11+=1;star[i].a22+=1;star[i].a33+=1;
    inv[i]=fmath::InverseMatrix3x3(star[i]);
    const tmatrix3d b=natural64.empty()?UserGFlexCorrectionMath::Promote(h.NaturalKerCorr[i]):natural64[i];
    transform[i]=UserGFlexCorrectionMath::Multiply(UserGFlexCorrectionMath::Transpose(inv[i]),UserGFlexCorrectionMath::Transpose(b));
    transformf[i]=fmath::MulMatrix3x3(fmath::TrasMatrix3x3(fmath::InverseMatrix3x3(h.Fstar[i])),fmath::TrasMatrix3x3(h.NaturalKerCorr[i]));
    offset[i]=UserGFlexCorrectionMath::Multiply(hs,inv[i]);
    for(unsigned k=0;k<9;k++)(&offset[i].a11)[k]=-(&offset[i].a11)[k];
  }
  for(unsigned i=0;i<n;i++)for(unsigned k=h.PairOffset[i];k<h.PairOffset[i+1];k++){
    const unsigned j=pairs[k];
    const tdouble3 a=h.NaturalPos[i],b=h.NaturalPos[j];
    const tdouble3 rn=TDouble3(a.x-b.x,a.y-b.y,a.z-b.z);
    const tfloat3 rf=TFloat3(float(rn.x),float(rn.y),float(rn.z));
    const float rr=rf.x*rf.x+rf.y*rf.y+rf.z*rf.z;
    const double rr64=rn.x*rn.x+rn.y*rn.y+rn.z*rn.z;
    if(!std::isfinite(rr64) || rr64<=0 || !std::isfinite(rr) || rr<=0)Run_Exceptioon("Invalid natural pair distance.");
    const float fac=TKernel==KERNEL_Wendland?fsph::GetKernel_Fac<KERNEL_Wendland>(CSP,rr):fsph::GetKernel_Fac<KERNEL_Cubic>(CSP,rr);
    const tfloat3 gf=TFloat3(fac*rf.x,fac*rf.y,fac*rf.z);
    if(wide){
      // A CPU/float-origin pair file retains its original discrete kernel.
      const tdouble3 g=natural64.empty()?UserGFlexCorrectionMath::Promote(gf):
          UserGFlexCorrectionMath::Scale(rn,UserGFlexCorrectionMath::Fac64(TKernel,CSP,rr64));
      gi[k]=UserGFlexCorrectionMath::Scale(UserGFlexCorrectionMath::MV(transform[i],g),h.NaturalVolume[i]);
      gj[k]=UserGFlexCorrectionMath::Scale(UserGFlexCorrectionMath::MV(transform[j],g),h.NaturalVolume[j]);
      mi[k]=UserGFlexCorrectionMath::MV(star[i],rn);mj[k]=UserGFlexCorrectionMath::MV(star[j],rn);
      weight[k]=double(h.NaturalVolume[i])*double(h.NaturalVolume[i])/rr64;
    }
    else{
      auto ai=UserGFlexCorrectionMath::MV(transformf[i],gf),aj=UserGFlexCorrectionMath::MV(transformf[j],gf);
      ai=TFloat3(ai.x*h.NaturalVolume[i],ai.y*h.NaturalVolume[i],ai.z*h.NaturalVolume[i]);
      aj=TFloat3(aj.x*h.NaturalVolume[j],aj.y*h.NaturalVolume[j],aj.z*h.NaturalVolume[j]);
      gi[k]=UserGFlexCorrectionMath::Promote(ai);gj[k]=UserGFlexCorrectionMath::Promote(aj);
      mi[k]=UserGFlexCorrectionMath::Promote(UserGFlexCorrectionMath::MV(h.Fstar[i],rf));
      mj[k]=UserGFlexCorrectionMath::Promote(UserGFlexCorrectionMath::MV(h.Fstar[j],rf));
      weight[k]=double(h.NaturalVolume[i]*h.NaturalVolume[i]/rr);
    }
    const double du[3]={u0[i].x-u0[j].x,u0[i].y-u0[j].y,u0[i].z-u0[j].z};
    const double g[3]={gi[k].x,gi[k].y,gi[k].z};
    for(unsigned row=0;row<3;row++)for(unsigned col=0;col<3;col++)
      (&offset[i].a11)[3*row+col]-=du[row]*g[col];
  }
  for(unsigned i=0;i<n;i++){
    if(Simulate2D)offset[i].a12=offset[i].a21=offset[i].a22=offset[i].a23=offset[i].a32=0;
    for(unsigned k=0;k<9;k++)if(!std::isfinite((&offset[i].a11)[k]))Run_Exceptioon("Invalid inherited gradient offset.");
  }
  // Transactional topology replacement at initialisation; never use the old
  // geometry-based neighbour count to size the precision pair caches.
  unsigned* newbuffer=nullptr;
  if(cudaMalloc(reinterpret_cast<void**>(&newbuffer),size_t(count)*sizeof(unsigned))!=cudaSuccess)
    Run_Exceptioon("GPU reference pair allocation failed.");
  std::vector<unsigned*> addresses(n);
  for(unsigned i=0;i<n;i++)addresses[i]=newbuffer+h.PairOffset[i];
  if(cudaMemcpy(newbuffer,pairs.data(),size_t(count)*sizeof(unsigned),cudaMemcpyHostToDevice)!=cudaSuccess ||
     cudaMemcpy(NumPairsg,counts.data(),n*sizeof(unsigned),cudaMemcpyHostToDevice)!=cudaSuccess ||
     cudaMemcpy(PairIdxg,addresses.data(),n*sizeof(unsigned*),cudaMemcpyHostToDevice)!=cudaSuccess){
    cudaFree(newbuffer);Run_Exceptioon("GPU reference pair upload failed.");
  }
  cudaFree(PairIdxBufferg);PairIdxBufferg=newbuffer;
  MemGpuFixed-=size_t(oldPairs)*sizeof(unsigned);MemGpuFixed+=size_t(count)*sizeof(unsigned);
  FlexPrecision->UploadCorrections(n,count,gi.data(),gj.data(),mi.data(),mj.data(),weight.data(),offset.data());
  Log->Printf("GPU directcontinuation: all three corrections enabled; %u original directed pairs restored.",count);
  return count;
}
