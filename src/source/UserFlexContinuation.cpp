#include "JSphCpuSingle.h"
#include "UserParameters.h"
#include "JBinaryData.h"
#include "JLog2.h"
#include "Functions.h"
#include "FunctionsMath.h"
#include "FunSphKernel.h"

#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <limits>
#include <memory>

namespace {
  inline tfloat3 FlexMV(
      const tmatrix3f& a,
      const tfloat3& b)
  {
    return TFloat3(
        a.a11*b.x+a.a12*b.y+a.a13*b.z,
        a.a21*b.x+a.a22*b.y+a.a23*b.z,
        a.a31*b.x+a.a32*b.y+a.a33*b.z);
  }

  inline tfloat3 FlexScale(
      const tfloat3& a,
      float s)
  {
    return TFloat3(a.x*s,a.y*s,a.z*s);
  }
}

using namespace std;

//==============================================================================
/// Saves particle IDs and total deformation gradients relative to Xn.
//==============================================================================
void JSphCpuSingle::SaveFlexData(){
  if(!User->SaveFlexData || !FlexStruc)
    return;
  if(!(SvData & SDAT_Binx))
    return;

  const unsigned nf=CaseNflexstruc;
  if(!nf)
    return;

  if(!FlexStrucRidpc || !DefGradc)
    Run_Exceptioon("Flexible state arrays are unavailable.");

  const std::string file=
      fun::GetDirWithSlash(DirDataOut)
      +fun::PrintStr("PartFlex_%04u.bi4",unsigned(Part));

  // A v3 frame contains only a source marker, not per-frame material arrays.
  const std::string materialStorage=
      (User->IsFlexRestart() || User->SaveFlexPairData) ? "material" : "xml";
  if(materialStorage=="material")SaveFlexMaterialDataOnce();

  JBinaryData data("JPartFlexBi4");
  data.SetvUint("FormatVer",3);
  data.SetvText("MaterialStorage",materialStorage);
  data.SetvText("WriterRunCode",RunCode);
  data.SetvUint("Cpart",unsigned(Part));
  data.SetvDouble("TimeStep",TimeStep);
  data.SetvUint("Nflex",nf);

  unsigned* ids=data.CreateArrayUint("Idp",nf,false);
  tfloat3* row1=data.CreateArrayFloat3("Frow1",nf,false);
  tfloat3* row2=data.CreateArrayFloat3("Frow2",nf,false);
  tfloat3* row3=data.CreateArrayFloat3("Frow3",nf,false);

  std::unordered_set<unsigned> seen;
  seen.reserve(nf);

  for(unsigned pfs=0;pfs<nf;pfs++){
    // Flexible-array index -> main-array index -> stable particle ID.
    const unsigned p=FlexStrucRidpc[pfs];

    if(p>=Np)
      Run_Exceptioon("Invalid flexible particle index.");

    const typecode code=Code_c->cptr()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFlexStrucAny(code))
      Run_Exceptioon("Invalid flexible particle in PartFlex output.");

    const unsigned id=Idp_c->cptr()[p];
    if(!seen.insert(id).second)
      Run_Exceptioon("Duplicated flexible particle ID.");

    ids[pfs]=id;

    tmatrix3f f=DefGradc[pfs];

    if(User->IsFlexRestart()){
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

void JSphCpuSingle::LoadFlexDirectContinuationState(){
  User->FlexRestartState.Clear();
  User->FlexReferenceId.clear();
  User->FlexRestartWriterRunCode.clear();

  if(!PartBegin || !FlexStruc)
    return;

  const unsigned nf=CaseNflexstruc;
  if(!nf)
    return;

  // Loading restart state does not overwrite the relative DefGradc.
  if(!FlexStrucRidpc)
    Run_Exceptioon("Flexible particle indices are unavailable.");

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
    const unsigned p=FlexStrucRidpc[pfs];

    if(p>=Np)
      Run_Exceptioon("Invalid flexible particle index.");

    const typecode code=Code_c->cptr()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFlexStrucAny(code))
      Run_Exceptioon("Invalid flexible particle during restart.");

    const unsigned id=Idp_c->cptr()[p];
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

//==============================================================================
/// Saves available natural-reference geometry and discretization data.
//==============================================================================

void JSphCpuSingle::SaveFlexPairDataOnce()
{
  if(!User->SaveFlexPairData || !FlexStruc || !CaseNflexstruc)
    return;
  if(!(SvData & SDAT_Binx))
    return;
  if(User->FlexPairDataWritten)
    return;

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
      const unsigned p=FlexStrucRidpc[pfs];

      if(p>=Np)
        Run_Exceptioon("Invalid flexible particle index.");

      const unsigned body=
          CODE_GetIbodyFlexStruc(Code_c->cptr()[p]);

      reference.Idp[pfs]=Idp_c->cptr()[p];
      reference.NaturalPos[pfs]=Pos0c[pfs];
      reference.NaturalKerCorr[pfs]=KerCorrc[pfs];
      reference.NaturalVolume[pfs]=FlexStrucDatac[body].vol0;
      reference.NaturalDensity[pfs]=FlexStrucDatac[body].rho0;

      const unsigned previous=reference.PairOffset[pfs];
      const unsigned count=NumPairsc[pfs];

      if(count>std::numeric_limits<unsigned>::max()-previous)
        Run_Exceptioon("Too many flexible reference pairs.");

      reference.PairOffset[pfs+1]=previous+count;
    }

    reference.PairIdp.resize(reference.PairOffset[nf]);

    for(unsigned pfs=0;pfs<nf;pfs++){
      for(unsigned k=0;k<NumPairsc[pfs];k++){
        const unsigned neighbour=PairIdxc[pfs][k];

        if(neighbour>=nf)
          Run_Exceptioon("Invalid flexible neighbour index.");

        reference.PairIdp[reference.PairOffset[pfs]+k]=
            Idp_c->cptr()[FlexStrucRidpc[neighbour]];
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

  unsigned* pairs=data.CreateArrayUint(
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
      pairs);

  data.SaveFile(file,false,true);

  // Mark only after successful writing.
  User->FlexPairDataWritten=true;
}

void JSphCpuSingle::LoadFlexDirectContinuationPairData()
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

  state.HasNaturalReferenceData=true;
  User->FlexRestartState=state;
  User->FlexReferenceId=referenceId;
  Log->Printf("FlexRestart: reference kernel data loaded from %s.",file.c_str());
}

void JSphCpuSingle::InitFlexRestartReferenceAndCorrections()
{
  if(!User->IsFlexRestart() ||
     !FlexStruc || !CaseNflexstruc)
  {
    return;
  }

  const bool required=
      User->UseFlexRestartRestorePairs ||
      User->UseFlexRestartGradientCorrection ||
      User->UseFlexRestartHourglassCorrection;
  const bool exportPairData=User->SaveFlexPairData && (SvData & SDAT_Binx)
      && !User->IsFlexUndeformedRestart();

  if(User->IsFlexDirectContinuation() && (required || exportPairData)){
    LoadFlexDirectContinuationPairData();
  }

  const auto& state=User->FlexRestartState;

  if((required || exportPairData) && !state.HasNaturalReferenceLayout()){
    Run_Exceptioon(
        "FlexRestart: requested inheritance or SaveFlexPairData requires "
        "original natural-reference geometry and kernel data.");
  }

  const bool needsKernel=
      User->UseFlexRestartGradientCorrection ||
      User->UseFlexRestartHourglassCorrection;

  if(needsKernel &&
     (state.ReferenceKernel!=unsigned(TKernel) ||
      state.ReferenceH!=KernelH ||
      state.Reference2D!=Simulate2D))
  {
    Run_Exceptioon(
        "FlexRestart: kernel settings differ from "
        "natural-reference data.");
  }

  if(User->UseFlexRestartRestorePairs)
    RestoreFlexRestartPairs();

  if(User->UseFlexRestartGradientCorrection)
    PrepareFlexRestartGradientCorrection();

  if(User->UseFlexRestartHourglassCorrection)
    PrepareFlexRestartHourglassCorrection();
}

//==============================================================================
// Optional restart discretization corrections.
// Builds runtime corrections from available natural-reference data.
//==============================================================================
void JSphCpuSingle::RestoreFlexRestartPairs()
{
  const auto& state=User->FlexRestartState;

  if(!state.HasNaturalReferenceLayout())
    Run_Exceptioon("Flexible reference pairs are unavailable.");

  const unsigned nf=CaseNflexstruc;
  const size_t count=state.PairIdp.size();

  std::unordered_map<unsigned,unsigned> currentIndex;
  currentIndex.reserve(nf);

  size_t oldCount=0;

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned p=FlexStrucRidpc[pfs];
    const unsigned id=Idp_c->cptr()[p];

    if(id!=state.Idp[pfs] ||
       !currentIndex.emplace(id,pfs).second)
    {
      Run_Exceptioon("Flexible reference ID mapping is invalid.");
    }

    oldCount+=NumPairsc[pfs];
  }

  // Build new storage before replacing the active buffer.
  std::unique_ptr<unsigned[]> buffer(new unsigned[count]);

  for(size_t q=0;q<count;q++){
    const auto it=currentIndex.find(state.PairIdp[q]);

    if(it==currentIndex.end())
      Run_Exceptioon("Reference neighbour is absent from this run.");

    buffer[q]=it->second;
  }

  delete[] PairIdxBufferc;
  PairIdxBufferc=buffer.release();

  MemCpuFixed-=sizeof(unsigned)*oldCount;
  MemCpuFixed+=sizeof(unsigned)*count;

  for(unsigned pfs=0;pfs<nf;pfs++){
    const unsigned begin=state.PairOffset[pfs];
    const unsigned end=state.PairOffset[pfs+1];

    NumPairsc[pfs]=end-begin;
    PairIdxc[pfs]=PairIdxBufferc+begin;
  }

  Log->Printf(
      "User.FlexRestartRestorePairs: restored %u directed pairs.",
      unsigned(count));
}

void JSphCpuSingle::PrepareFlexRestartPairCache()
{
  auto& h=User->FlexRestartState;
  const unsigned n=CaseNflexstruc;

  if(!h.HasRestartStateLayout() ||
     !h.HasNaturalReferenceLayout() ||
     h.Idp.size()!=n)
  {
    Run_Exceptioon(
        "FlexRestart: natural-reference correction data are unavailable.");
  }

  // Both corrections share the same active pair layout.
  if(!h.CorrectionOffset.empty())
    return;

  h.CorrectionOffset.resize(size_t(n)+1);
  h.CorrectionOffset[0]=0;

  for(unsigned i=0;i<n;i++){
    if(h.Idp[i]!=Idp_c->cptr()[FlexStrucRidpc[i]])
      Run_Exceptioon(
          "Flexible correction particle ID mismatch.");

    if(User->UseFlexRestartGradientCorrection &&
       !NumPairsc[i])
    {
      Run_Exceptioon(
          "Gradient correction requires neighbours.");
    }

    h.CorrectionOffset[i+1]=
        h.CorrectionOffset[i]+NumPairsc[i];

    for(unsigned k=0;k<NumPairsc[i];k++){
      const unsigned j=PairIdxc[i][k];

      if(j>=n || j==i)
        Run_Exceptioon("Invalid correction neighbour.");

      if(User->UseFlexRestartGradientCorrection){
        const unsigned a=h.PairOffset[i];
        const unsigned b=h.PairOffset[i+1];

        if(b-a!=NumPairsc[i] ||
           h.PairIdp[a+k]!=h.Idp[j])
        {
          Run_Exceptioon(
              "Gradient correction requires "
              "the original ordered pairs.");
        }
      }

      h.CorrectionNeighbour.push_back(j);
    }
  }
}

void JSphCpuSingle::PrepareFlexRestartGradientCorrection()
{
  if(!User->UseFlexRestartGradientCorrection)
    return;

  PrepareFlexRestartPairCache();

  auto& h=User->FlexRestartState;
  const unsigned n=CaseNflexstruc;

  if(TKernel!=KERNEL_Wendland &&
     TKernel!=KERNEL_Cubic)
  {
    Run_Exceptioon(
        "Unsupported flexible correction kernel.");
  }

  std::vector<tmatrix3f> transform(n);

  for(unsigned i=0;i<n;i++){
    const tmatrix3f inv=
        fmath::InverseMatrix3x3(h.Fstar[i]);

    // q0 = F*^{-T} Bn^T Gn Vn.
    transform[i]=fmath::MulMatrix3x3(
        fmath::TrasMatrix3x3(inv),
        fmath::TrasMatrix3x3(h.NaturalKerCorr[i]));
  }

  const size_t count=h.CorrectionNeighbour.size();

  h.Grad0I.resize(count);
  h.Grad0J.resize(count);

  for(unsigned i=0;i<n;i++){
    for(size_t k=h.CorrectionOffset[i];
        k<h.CorrectionOffset[i+1];k++)
    {
      const unsigned j=h.CorrectionNeighbour[k];

      const tdouble3 a=h.NaturalPos[i];
      const tdouble3 b=h.NaturalPos[j];

      const tfloat3 r=TFloat3(
          float(a.x-b.x),
          float(a.y-b.y),
          float(a.z-b.z));

      const float rr=
          r.x*r.x+r.y*r.y+r.z*r.z;

      if(!std::isfinite(rr) || rr<=0)
        Run_Exceptioon(
            "Invalid natural pair distance.");

      const float fac=
          TKernel==KERNEL_Wendland
          ? fsph::GetKernel_Fac<KERNEL_Wendland>(CSP,rr)
          : fsph::GetKernel_Fac<KERNEL_Cubic>(CSP,rr);

      const tfloat3 g=FlexScale(r,fac);

      // Both coefficients use the SAME i-minus-j direction.
      h.Grad0I[k]=FlexScale(
          FlexMV(transform[i],g),h.NaturalVolume[i]);

      h.Grad0J[k]=FlexScale(
          FlexMV(transform[j],g),h.NaturalVolume[j]);
    }
  }

  h.GradientReady=true;

  Log->Print(
      "User.FlexRestartGradientCorrection: "
      "transformed gradients prepared.");
}

void JSphCpuSingle::PrepareFlexRestartHourglassCorrection()
{
  if(!User->UseFlexRestartHourglassCorrection)
    return;

  PrepareFlexRestartPairCache();

  auto& h=User->FlexRestartState;
  const size_t count=h.CorrectionNeighbour.size();

  h.HgMappedI.resize(count);
  h.HgMappedJ.resize(count);
  h.HgWeight.resize(count);

  for(unsigned i=0;i<CaseNflexstruc;i++){
    for(size_t k=h.CorrectionOffset[i];
        k<h.CorrectionOffset[i+1];k++)
    {
      const unsigned j=h.CorrectionNeighbour[k];

      const tdouble3 a=h.NaturalPos[i];
      const tdouble3 b=h.NaturalPos[j];

      const tfloat3 r=TFloat3(
          float(a.x-b.x),
          float(a.y-b.y),
          float(a.z-b.z));

      const float rr=
          r.x*r.x+r.y*r.y+r.z*r.z;

      if(!std::isfinite(rr) || rr<=0)
        Run_Exceptioon(
            "Invalid natural HG pair distance.");

      // Fixed throughout this restart.
      h.HgMappedI[k]=FlexMV(h.Fstar[i],r);
      h.HgMappedJ[k]=FlexMV(h.Fstar[j],r);

      // Match the original ordinary HG branch.
      h.HgWeight[k]=h.NaturalVolume[i]*h.NaturalVolume[i]/rr;
    }
  }

  h.HourglassReady=true;

  Log->Print(
      "User.FlexRestartHourglassCorrection: "
      "natural-reference mappings and penalty weights prepared.");
}



//==============================================================================
// Material invariants are constant within a run. No geometry or kernels here.
//==============================================================================
void JSphCpuSingle::SaveFlexMaterialDataOnce(){
  if(User->FlexMaterialDataWritten)return;
  const auto& state=User->FlexRestartState;
  const unsigned nf=CaseNflexstruc;
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
    const unsigned p=FlexStrucRidpc[i];
    if(p>=Np)Run_Exceptioon("Invalid material-output particle index.");
    ids[i]=Idp_c->cptr()[p];
    if(restart){
      if(ids[i]!=state.Idp[i])
        Run_Exceptioon("Material-output particle identity mismatch.");
      vn[i]=state.NaturalVolume[i];
      rn[i]=state.NaturalDensity[i];
      mass[i]=state.ParticleMass[i];
    }
    else{
      const auto& mat=FlexStrucDatac[CODE_GetIbodyFlexStruc(Code_c->cptr()[p])];
      vn[i]=mat.vol0;
      rn[i]=mat.rho0;
      mass[i]=vn[i]*rn[i];
    }
  }
  data.SaveFile(file,false,true);
  User->FlexMaterialDataWritten=true;
}

// Reads material arrays only: this does not enable pair/kernel inheritance.
void JSphCpuSingle::LoadFlexMaterialData(const std::string& storage){
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
