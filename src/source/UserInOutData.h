#ifndef USERINOUTDATA_H
#define USERINOUTDATA_H

#include "JSphInOut.h"
#include "JSphInOutZone.h"
#include "JSphInOutVel.h"
#include "JSphInOutZsurf.h"
#include "JSphMk.h"
#include "JLinearValue.h"
#include "JBinaryData.h"
#include "JPartDataBi4.h"
#include "Functions.h"

#include <fstream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class UserInOutData {
  static void Require(bool ok,const char* text){
    if(!ok)
      throw std::runtime_error(std::string("SaveInOutData: ")+text);
  }

  struct Hash {
    ullong value=14695981039346656037ULL;

    void Bytes(const void* ptr,size_t n){
      const byte* p=static_cast<const byte*>(ptr);
      for(size_t i=0;i<n;i++){
        value^=p[i];
        value*=1099511628211ULL;
      }
    }

    template<class T> void Add(const T& v){
      Bytes(&v,sizeof(T));
    }

    template<class T> void Array(const T* p,size_t n){
      Bytes(p,n*sizeof(T));
    }
  };

  static ullong FileHash(const std::string& path){
    Hash h;
    std::ifstream file(path.c_str(),std::ios::binary);
    Require(bool(file),"cannot read checkpoint/configuration file.");

    char block[65536];
    while(file){
      file.read(block,sizeof(block));
      h.Bytes(block,size_t(file.gcount()));
    }
    Require(file.eof(),"file reading failed.");
    return h.value;
  }

  static ullong ConfigHash(JSphInOut& s,const JSphMk& mk){
    Hash h;
    h.Add(FileHash(s.XmlFile));

    const unsigned endian=0x01020304;
    const unsigned codebytes=sizeof(typecode);
    h.Add(endian);
    h.Add(codebytes);

#define IO_H(x) h.Add(s.x)
    IO_H(Stable);
    IO_H(MapRealPosMin); IO_H(MapRealPosMax); IO_H(CodeNewPart);
    IO_H(DetermLimit); IO_H(ExtrapolateMode); IO_H(UseBoxLimit);
    IO_H(FreeLimitMin); IO_H(FreeLimitMax); IO_H(RefillingRate);

#define IO_C(x) h.Add(s.CSP.x)
    IO_C(simulate2d); IO_C(simulate2dposy); IO_C(tkernel);
    IO_C(kernelh); IO_C(cteb); IO_C(gamma); IO_C(rhopzero); IO_C(dp);
    IO_C(massfluid); IO_C(massbound); IO_C(gravity);
    IO_C(kernelsize); IO_C(kernelsize2); IO_C(cs0); IO_C(eta2);
#undef IO_C

    IO_H(ListSize);
    IO_H(PtCount);
#undef IO_H

    h.Array(s.Planes,s.ListSize);
    h.Array(s.CfgZone,s.ListSize);
    h.Array(s.CfgUpdate,s.ListSize);
    h.Array(s.Width,s.ListSize);
    h.Array(s.DirData,s.ListSize);
    h.Array(s.DirVel,s.ListSize);
    h.Array(s.PtZone,s.PtCount);
    h.Array(s.PtPos,s.PtCount);

    for(auto* z:s.List){
      h.Array(z->PtDom,10);
      h.Add(z->BoxLimitMin);
      h.Add(z->BoxLimitMax);

      for(auto* p:{z->InOutVel->InputTimeVel,z->InOutZsurf->InputTime}){
        const bool present=(p!=NULL);
        h.Add(present);
        if(p){
          h.Add(p->Nvalues);
          h.Add(p->Count);
          h.Array(p->Times,p->Count);
          h.Array(p->Values,size_t(p->Count)*p->Nvalues);
        }
      }
    }

    for(unsigned i=0;i<mk.Size();i++){
      const auto* b=mk.Mkblock(i);
      h.Add(b->Begin);
      h.Add(b->Count);
      h.Add(b->Code);
      h.Add(b->Mk);
    }
    return h.value;
  }

  static std::vector<unsigned> Intervals(JSphInOut& s){
    std::vector<unsigned> out;
    for(auto* z:s.List){
      for(auto* p:{z->InOutVel->InputTimeVel,z->InOutZsurf->InputTime}){
        if(p){
          out.push_back(p->Position);
          out.push_back(p->PositionNext);
        }
      }
    }
    return out;
  }

  static void RebuildTime(JSphInOut& s,double time,
      const std::vector<unsigned>& iv){
    size_t k=0;

    for(auto* z:s.List){
      for(auto* p:{z->InOutVel->InputTimeVel,z->InOutZsurf->InputTime}){
        if(p){
          Require(k+2<=iv.size(),"missing interpolation interval.");
          const unsigned a=iv[k++];
          const unsigned b=iv[k++];

          Require(a<p->Count && b<p->Count && (b==a || b==a+1),
              "invalid interpolation interval.");

          p->Position=a;
          p->PositionNext=b;
          p->TimeStep=time;
          p->TimePre=p->Times[a];
          p->TimeNext=p->Times[b];

          const double dt=p->TimeNext-p->TimePre;
          p->TimeFactor=(dt? (time-p->TimePre)/dt: 0);
          if(p->TimeFactor<0)p->TimeFactor=0;
          if(p->TimeFactor>1)p->TimeFactor=1;

          // This event flag is not used by the supported InOut calculations.
          p->NewInterval=false;
        }
      }

      // Force reconstruction of velocity coefficients from saved endpoints.
      z->InOutVel->TimeVelIdx0=UINT_MAX;
      z->InOutVel->TimeVelIdx1=UINT_MAX;
    }

    Require(k==iv.size(),"unexpected interpolation interval.");
    s.UpdateVelData(time);
    s.UpdateZsurfData(time,true);
  }

  template<class T> static std::vector<T> ReadArray(JBinaryData& data,
      const char* name,JBinaryDataDef::TpData type){
    auto* a=data.GetArray(name);
    Require(a && a->GetType()==type,"missing array or wrong type.");

    std::vector<T> v(a->GetCount());
    if(!v.empty())a->GetDataCopy(v.size(),v.data());
    return v;
  }

public:
  static void Check(JSphInOut& s){
    Require(s.Cpu && !s.PeriActive && !s.ReuseIds && s.ListSize,
        "requires non-periodic CPU InOut without ID reuse.");

    for(auto* z:s.List){
      const auto& v=*z->InOutVel;
      const auto& h=*z->InOutZsurf;

      Require(!z->ExternalVarInput && !v.AwasVel && !v.InputVelGrid
          && !v.InputMeshVel && !v.FlowActive && h.MeshFile.empty()
          && !h.InputMesh && !h.GaugeSwl && !h.GaugeMesh && !h.TimeCount
          && h.UniformZsurf && h.ZsurfMode!=InZsurf_Calculated
          && (v.VelMode==InVelM_Fixed || v.VelMode==InVelM_Variable)
          && v.VelProfile!=InVelP_JetCircle,
          "unsupported external/grid/AWAS/flow/jet/extrapolated-velocity mode.");

      for(auto* p:{v.InputTimeVel,h.InputTime}){
        if(p){
          Require(p->Count && p->LoopTsub==0,
              "empty or looping table is unsupported.");
        }
      }
    }
  }

  static void Save(JSphInOut& s,const JSphMk& mk,const std::string& dir,
      const std::string& run,unsigned part,unsigned step,double time,
      unsigned np,const unsigned* idp,const typecode* code,
      ullong initial,ullong total,unsigned idmax,bool& written){
    Check(s);

    Require(total>=initial && total && total<=UINT_MAX && idmax==total-1,
        "particle ID counters cannot be reconstructed.");

    Require(s.NewNpTotal==total-initial && s.NewNpPart==0
        && s.Nstep==(step? step-1: 0),
        "unexpected output phase or inlet counters.");

    std::vector<unsigned> inlet,converted;
    std::vector<byte> zone;

    for(unsigned p=0;p<np;p++){
      Require(CODE_IsNormal(code[p]),
          "excluded/periodic particle in main PART.");

      // Flex and other boundaries belong to MK/Flex initialization.
      if(!CODE_IsFluid(code[p]))continue;

      Require(idp[p]<total,"particle ID exceeds total-created count.");

      const typecode base=(idp[p]<initial?
          mk.GetCodeById(idp[p]): s.CodeNewPart);

      Require(CODE_IsFluid(base),
          "fluid ID belongs to an initial boundary.");

      if(CODE_IsFluidInout(code[p])){
        const unsigned iz=CODE_GetIzoneFluidInout(code[p]);

        Require(iz<s.ListSize
            && code[p]==(CODE_ToFluidInout(s.CodeNewPart,iz)),
            "temporary or unsupported inlet identity at output.");

        inlet.push_back(idp[p]);
        zone.push_back(byte(iz));
      }
      else if(code[p]!=base){
        Require(idp[p]<initial && code[p]==s.CodeNewPart,
            "fluid identity cannot be represented losslessly.");
        converted.push_back(idp[p]);
      }
    }

    Require(inlet.size()==s.CurrentNp,"inlet particle count mismatch.");

    for(auto* z:s.List){
      Require(z->InOutVel->CurrentTime==time,
          "velocity time differs from PART.");

      for(auto* p:{z->InOutVel->InputTimeVel,z->InOutZsurf->InputTime}){
        if(p){
          Require(p->TimeStep==time,"table time differs from PART.");
        }
      }
    }

    const ullong signature=ConfigHash(s,mk);
    const std::string root=fun::GetDirWithSlash(dir);
    const std::string hf=root+"PartHeadInOut.ibi4";
    JBinaryData head("PartHeadInOut");

    if(!written){
      head.SetvUint("Version",2);
      head.SetvText("RunCode",run);
      head.SetvUllong("ConfigHash",signature);
      head.SaveFile(hf);
      written=true;
    }
    else{
      head.LoadFile(hf,"PartHeadInOut");
      Require(head.GetvUint("Version")==2
          && head.GetvText("RunCode")==run
          && head.GetvUllong("ConfigHash")==signature,
          "header/configuration changed.");
    }

    JBinaryData frame("PartInOut");
    frame.SetvUint("Version",2);
    frame.SetvText("RunCode",run);
    frame.SetvUint("Part",part);
    frame.SetvDouble("Time",time);
    frame.SetvUllong("ConfigHash",signature);

    // Read-only binding to the unchanged main PART.
    frame.SetvUllong("MainHash",
        FileHash(root+JPartDataBi4::GetFileNamePart(part)));

    frame.CreateArray("InletId",JBinaryDataDef::DatUint,
        inlet.size(),inlet.data(),false);
    frame.CreateArray("InletZone",JBinaryDataDef::DatUchar,
        zone.size(),zone.data(),false);
    frame.CreateArray("ConvertedId",JBinaryDataDef::DatUint,
        converted.size(),converted.data(),false);

    const auto iv=Intervals(s);
    frame.CreateArray("Intervals",JBinaryDataDef::DatUint,
        iv.size(),iv.data(),false);

    frame.SaveFile(root+fun::FileNameSec("PartInOut.bi4",part));
  }

  // Only an InOut restore helper; this does not enable the solver's restart path.
  // Load the unchanged main PART first. Initialize MK/Flex codes for initial IDs;
  // assign GetCodeNewFluid() to newly created IDs, then call this helper.
  static void Restore(JSphInOut& s,const JSphMk& mk,JBinaryData& head,
      JBinaryData& frame,const JPartDataBi4& main,unsigned part,
      unsigned np,const unsigned* idp,typecode* code){
    Check(s);
    const ullong signature=ConfigHash(s,mk);

    Require(head.GetvUint("Version")==2
        && frame.GetvUint("Version")==2
        && head.GetvText("RunCode")==frame.GetvText("RunCode")
        && (main.Get_RunCode()=="00000000"
            || frame.GetvText("RunCode")==main.Get_RunCode())
        && frame.GetvUllong("MainHash")==FileHash(main.GetFileLoaded())
        && frame.GetvUint("Part")==part
        && frame.GetvDouble("Time")==main.Get_TimeStep()
        && head.GetvUllong("ConfigHash")==signature
        && frame.GetvUllong("ConfigHash")==signature,
        "incompatible checkpoint/configuration.");

    Require(np==main.Get_Npok(),"main PART particle count mismatch.");

    const ullong initial=main.Get_CaseNp();
    const ullong total=main.Get_NpTotal();
    Require(total>=initial && total && total<=UINT_MAX,
        "invalid creation count.");

    const auto inlet=ReadArray<unsigned>(
        frame,"InletId",JBinaryDataDef::DatUint);
    const auto zone=ReadArray<byte>(
        frame,"InletZone",JBinaryDataDef::DatUchar);
    const auto converted=ReadArray<unsigned>(
        frame,"ConvertedId",JBinaryDataDef::DatUint);
    const auto iv=ReadArray<unsigned>(
        frame,"Intervals",JBinaryDataDef::DatUint);

    Require(inlet.size()==zone.size(),"inlet ID/zone counts differ.");

    std::unordered_map<unsigned,unsigned> index;
    for(unsigned p=0;p<np;p++){
      Require(idp[p]<total && index.emplace(idp[p],p).second,
          "invalid/duplicate particle ID.");
    }

    std::vector<typecode> next(code,code+np);
    std::vector<byte> used(np,0);

    auto lookup=[&](unsigned id){
      auto it=index.find(id);
      Require(it!=index.end(),
          "checkpoint particle is missing from main PART.");

      const unsigned p=it->second;
      Require(!used[p] && CODE_IsFluid(next[p]),
          "duplicate or non-fluid checkpoint ID.");

      used[p]=1;
      return p;
    };

    for(unsigned id:converted){
      Require(id<initial,"converted original ID is not original.");
      next[lookup(id)]=s.CodeNewPart;
    }

    for(size_t i=0;i<inlet.size();i++){
      Require(zone[i]<s.ListSize,"invalid inlet zone.");
      next[lookup(inlet[i])]=CODE_ToFluidInout(s.CodeNewPart,zone[i]);
    }

    RebuildTime(s,main.Get_TimeStep(),iv);

    s.Nstep=(main.Get_Step()? main.Get_Step()-1: 0);
    s.NewNpTotal=total-initial;
    s.NewNpPart=0;
    s.CurrentNp=unsigned(inlet.size());

    std::copy(next.begin(),next.end(),code);
  }
};

#endif