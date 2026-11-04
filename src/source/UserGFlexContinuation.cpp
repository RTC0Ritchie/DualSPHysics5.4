#include "JSphGpuSingle.h"
#include "UserParameters.h"
#include "UserGFlexPrecision.h"
#include "UserGParameters.h"
#include "JBinaryData.h"
#include "JLog2.h"
#include "Functions.h"
#include "FunctionsMath.h"
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

void JSphGpuSingle::CompleteFlexRestartMaterialState(bool validateOnly)
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


  UserGFlexContinuationSnapshot snapshot(Npb,Idp_g->cptr(),Code_g->cptr(),
      FlexStrucCount,FlexStrucDatag,CaseNflexstruc,FlexStrucRidpg,nullptr);
  auto& state=User->FlexRestartState;
  const unsigned nf=CaseNflexstruc;

  if(state.Idp.size()!=nf ||
     state.Fstar.size()!=nf)
  {
    Run_Exceptioon(
        "FlexRestart: incomplete restart state.");
  }

  const bool suppliedMaterial=state.HasMaterialState;
  if(suppliedMaterial && (state.NaturalVolume.size()!=nf ||
      state.NaturalDensity.size()!=nf || state.ParticleMass.size()!=nf))
    Run_Exceptioon("FlexRestart: incomplete supplied material state.");

  state.Jstar.resize(nf);
  state.NaturalVolume.resize(nf);
  state.NaturalDensity.resize(nf);
  state.ReferenceVolume.resize(nf);
  state.ReferenceDensity.resize(nf);
  state.ParticleMass.resize(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned p=snapshot.Ridp[pfs];

    if(p>=Np ||
       state.Idp[pfs]!=snapshot.Id.data()[p])
    {
      Run_Exceptioon(
          "FlexRestart: particle ID mismatch.");
    }

    const typecode code=snapshot.Code.data()[p];

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
    const StFlexStrucData& material=snapshot.Material[body];

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

    if(suppliedMaterial){
      const float vn=state.NaturalVolume[pfs];
      const float rn=state.NaturalDensity[pfs];
      const float mass=state.ParticleMass[pfs];
      if(!std::isfinite(vn) || vn<=0 || !std::isfinite(rn) || rn<=0 ||
         !std::isfinite(mass) || mass<=0)
        Run_Exceptioon("FlexRestart: invalid supplied material invariants.");
      const double expected=double(vn)*rn;
      const double tolerance=32.0*std::numeric_limits<float>::epsilon()
          *std::max(expected,double(mass));
      if(std::fabs(double(mass)-expected)>tolerance)
        Run_Exceptioon("FlexRestart: inconsistent supplied particle mass.");
      state.ReferenceVolume[pfs]=vn*jstar;
      state.ReferenceDensity[pfs]=rn/jstar;
    }
    else if(conserveMass && densityIsNatural){
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
  state.HasMaterialState=true;
  if(User->IsFlexDirectContinuation())
    Log->Print(suppliedMaterial
        ? "FlexRestart.Material: saved material invariants restored."
        : "FlexRestart.Material: XML natural-state defaults (mass=true, natural=true).");

}

void JSphGpuSingle::InitFlexRestartState(){
  User->FlexRestartState.Clear();
  User->FlexReferenceId.clear();User->FlexRestartWriterRunCode.clear();
  User->FlexMaterialDataWritten=false;User->FlexPairDataWritten=false;
  if(User->UseFlexRestartRestorePairs || User->UseFlexRestartGradientCorrection ||
     User->UseFlexRestartHourglassCorrection || User->FlexRestartBoundaryCorrectionMode)
    Run_Exceptioon("GPU: optional restart pair/gradient/hourglass/boundary corrections are not migrated. Disable these options for basic continuation.");
  // Reserved extension point: natural-reference pair/correction upload belongs
  // here, before CalcFlexStrucKerCorr, independently of the precision policy.
  if(User->SaveFlexPairData)
    Log->Print("WARNING: GPU natural pair export is not migrated; PartFlex and FlexMaterialData are saved for basic direct continuation.");
  if(!User->IsFlexRestart())return;
  if(!User->IsFlexDirectContinuation())
    Run_Exceptioon("GPU currently supports direct elastic continuation only.");
  if(!CaseNflexstruc)Run_Exceptioon("GPU restart has no elastic particles.");
  LoadFlexDirectContinuationState();
  CompleteFlexRestartMaterialState(false);
  const auto& s=User->FlexRestartState;
  FlexPrecision->UploadRestart(CaseNflexstruc,s.Fstar.data(),s.Jstar.data(),
      s.ReferenceVolume.data(),s.ReferenceDensity.data(),s.ParticleMass.data(),
      s.NaturalVolume.data(),s.NaturalDensity.data());
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
  User->ConfigFlexRestartMode(PartBegin!=0,FlexStruc && FlexStruc->GetCount()>0,Log);
}
