#ifndef USER_INOUT_BOUND_MODE_H
#define USER_INOUT_BOUND_MODE_H

#include "DualSphDef.h"
#include "JBinaryData.h"
#include "JXml.h"
#include "JLog2.h"

#include <algorithm>
#include <map>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

class UserParameters;
class JSphInOut;
class JSphInOutZone;
class JSphCpuSingle;

// Selection storage lives outside UserParameters to preserve its layout.
class UserNoPenetrationMk final {
public:
  struct Range {
    unsigned Mk,Begin,End;
  };
  struct Data {
    std::vector<unsigned> Mks;
    std::vector<Range> Ranges;
  };
  static Data& Get(const UserParameters* user){return All()[user];}
  static void Reset(const UserParameters* user){All().erase(user);}
  static void Parse(const UserParameters* user,const std::string& text){
    auto& data=Get(user);
    data.Mks.clear();
    data.Ranges.clear();
    size_t begin=0;
    while(true){
      const size_t end=text.find(',',begin);
      const std::string token=text.substr(begin,
          end==std::string::npos?end:end-begin);
      const size_t first=token.find_first_not_of(" \t\r\n");
      const size_t last=token.find_last_not_of(" \t\r\n");
      if(first==std::string::npos)
        throw std::runtime_error("NoPenetrationMk: empty MK in list.");
      unsigned value=0;
      for(size_t i=first;i<=last;i++){
        if(token[i]<'0' || token[i]>'9')
          throw std::runtime_error(
              "NoPenetrationMk: use comma-separated integer MKs.");
        value=value*10+unsigned(token[i]-'0');
        if(value>65535)
          throw std::runtime_error("NoPenetrationMk: MK exceeds 65535.");
      }
      data.Mks.push_back(value);
      if(end==std::string::npos)break;
      begin=end+1;
    }
    std::sort(data.Mks.begin(),data.Mks.end());
    data.Mks.erase(std::unique(data.Mks.begin(),data.Mks.end()),data.Mks.end());
  }
  static bool Contains(const std::vector<Range>& ranges,unsigned id){
    for(const auto& range:ranges)
      if(id>=range.Begin && id<range.End)return true;
    return false;
  }
private:
  static std::map<const UserParameters*,Data>& All(){
    static std::map<const UserParameters*,Data> data;
    return data;
  }
};

class UserInOutBoundMode final {
public:
  // XML request, not the applied solver state.
  class StConfig {
  public:
    bool Enabled=false;
    unsigned NoBackflow=0; // 0=off, 1=legacy plane projection, 2=integration constraint

    // -1: unspecified; 0: wall; 1: inlet.
    int ValveStatue=-1;

    // Relative mkbound, not the absolute output MK.
    int ValveMkBound=-1;
  };

  static void Require(bool ok,const char* text){
    if(!ok)
      throw std::runtime_error(
          std::string("InOutBoundMode: ")+text);
  }

  // Applied configuration and restart history.
  class StRuntime {
  public:
    bool Enabled=false;
    bool Open=false;
    bool HistoryOpen=false;

    ullong GeometryHash=0;
    std::vector<typecode> Codes;

    bool IsGate(typecode code)const{
      return std::binary_search(
          Codes.begin(),Codes.end(),
          CODE_GetTypeAndValue(code));
    }

    void Header(JBinaryData& h)const{
      h.SetvUint("ValveVersion",1);
      h.SetvBool("ValveEnabled",Enabled);

      if(Enabled){
        h.SetvUllong("ValveGeometryHash",GeometryHash);

        std::vector<unsigned> codes(
            Codes.begin(),Codes.end());

        h.CreateArray(
            "ValveCodes",JBinaryDataDef::DatUint,
            codes.size(),codes.data(),false);
      }
    }

    void CheckHeader(JBinaryData& h)const{
      const bool saved=
          h.GetvBool("ValveEnabled",true,false);

      Require(
          saved==Enabled,
          "checkpoint valve registration differs; "
          "do not remove InOutBoundMode from a valve restart.");

      if(!Enabled)return;

      Require(
          h.GetvUint("ValveVersion")==1
          && h.GetvUllong("ValveGeometryHash")==GeometryHash,
          "valve geometry changed or checkpoint has no valve history.");

      auto* a=h.GetArray("ValveCodes");

      Require(
          a && a->GetType()==JBinaryDataDef::DatUint
          && a->GetCount()==Codes.size(),
          "invalid valve MK mapping.");

      std::vector<unsigned> codes(Codes.size());
      a->GetDataCopy(codes.size(),codes.data());

      Require(
          std::equal(
              codes.begin(),codes.end(),Codes.begin()),
          "valve MK mapping changed.");
    }

    void Frame(JBinaryData& f)const{
      f.SetvBool("ValveEnabled",Enabled);

      if(Enabled)
        f.SetvUint("ValveStatue",Open?1:0);
    }

    void Restore(JBinaryData& h,JBinaryData& f){
      CheckHeader(h);

      Require(
          f.GetvBool("ValveEnabled",true,false)==Enabled,
          "frame/header valve mismatch.");

      if(Enabled){
        const unsigned value=f.GetvUint("ValveStatue");
        Require(value<=1,"invalid saved ValveStatue.");
        HistoryOpen=(value==1);
      }
    }
  };

  static StConfig Read(const UserParameters* user){
    const auto& configs=Configs();
    const auto it=configs.find(user);

    return it==configs.end()?StConfig{}:it->second;
  }

  static void PrintConfig(
      const UserParameters* user,JLog2* log)
  {
    if(!log)return;

    const auto valve=Read(user);
    log->Printf("  InOutNoBackflow: %u (0=off, 1=plane, 2=integration)",valve.NoBackflow);

    log->Printf(
        "  InOutBoundMode: %s (requested)",
        valve.Enabled?"true":"false");

    if(valve.ValveStatue>=0){
      log->Printf(
          "    ValveStatue: %d (%s%s)",
          valve.ValveStatue,
          valve.ValveStatue==1?"open inlet":"closed boundary",
          valve.Enabled?"; requested":"; inactive");
    }

    if(valve.ValveMkBound>=0){
      log->Printf(
          "    ValveMkBound: %d%s",
          valve.ValveMkBound,
          valve.Enabled?"; requested":"; inactive");
    }

    if(valve.Enabled){
      log->Print(
          "    CPU/GPU valve requested; geometry and restart checks "
          "will run during InOut initialization.");
    }
  }

  static bool IsImplemented(){
    return true;
  }

  static StRuntime& Runtime(const JSphInOut* io){
    return States()[io];
  }

  static void ResetRuntime(const JSphInOut* io){
    States().erase(io);
  }

  static unsigned FileVersion(const JSphInOut* io){
    return Runtime(io).Enabled?4u:3u;
  }

  static void InitCpu(JSphCpuSingle& s);
  static void InitNoBackflowCpu(JSphCpuSingle& s);
  static void PrepareConstraintCpu(JSphCpuSingle& s,bool force);
  static float ConstraintSpeed(const JSphInOutZone* zone,double time);
  static void ApplyNoBackflowCpu(JSphCpuSingle& s);
  static void CheckNoBackflowProximityCpu(JSphCpuSingle& s,
      unsigned newnp,std::vector<typecode>& checked);

private:
  friend class UserParameters;

  UserInOutBoundMode()=delete;

  static std::map<const UserParameters*,StConfig>& Configs(){
    static std::map<const UserParameters*,StConfig> data;
    return data;
  }

  static std::map<const JSphInOut*,StRuntime>& States(){
    static std::map<const JSphInOut*,StRuntime> data;
    return data;
  }

  static void Reset(const UserParameters* user){
    Configs().erase(user);
  }

  static void Store(
      const UserParameters* user,const StConfig& config)
  {
    Configs()[user]=config;
  }

  static void LoadXml(
      const UserParameters* user,
      const JXml* xml,const TiXmlElement* ele)
  {
    xml->CheckAttributeNames(
        ele,"key value active comment units_comment");

    xml->CheckElementNames(ele,true,"ValveStatue ValveMkBound");

    StConfig config;
    config.Enabled=xml->GetAttributeBool(ele,"value");

    const TiXmlElement* state=
        ele->FirstChildElement("ValveStatue");

    if(state){
      Require(
          !state->NextSiblingElement("ValveStatue"),
          "repeated ValveStatue nodes.");

      xml->CheckAttributeNames(
          state,"value comment units_comment");

      Require(
          !state->FirstChildElement(),
          "ValveStatue does not accept child elements.");

      const double value=
          xml->GetAttributeDouble(state,"value");

      Require(
          value==0.0 || value==1.0,
          "ValveStatue must be 0 or 1.");

      config.ValveStatue=int(value);
    }

    Require(
        !config.Enabled || config.ValveStatue>=0,
        "ValveStatue is required when value is true.");

    const TiXmlElement* mk=
        ele->FirstChildElement("ValveMkBound");

    if(mk){
      Require(
          !mk->NextSiblingElement("ValveMkBound"),
          "repeated ValveMkBound nodes.");

      xml->CheckAttributeNames(
          mk,"value comment units_comment");

      Require(
          !mk->FirstChildElement(),
          "ValveMkBound does not accept child elements.");

      const double value=
          xml->GetAttributeDouble(mk,"value");

      Require(
          std::isfinite(value) && value>=0.0 && value<=65535.0
          && std::floor(value)==value,
          "ValveMkBound must be one integer in 0..65535.");

      config.ValveMkBound=int(value);
    }

    Require(
        !config.Enabled || config.ValveMkBound>=0,
        "ValveMkBound is required when value is true.");
    Store(user,config);
  }

  static void LoadNoBackflowXml(
      const UserParameters* user,
      const JXml* xml,const TiXmlElement* ele)
  {
    xml->CheckAttributeNames(
        ele,"key value active comment units_comment");
    Require(!ele->FirstChildElement(),
        "InOutNoBackflow does not accept child elements.");
    const double value=xml->GetAttributeDouble(ele,"value");
    Require(value==0.0 || value==1.0 || value==2.0,
        "InOutNoBackflow must be 0, 1 or 2.");
    auto config=Read(user);
    config.NoBackflow=unsigned(value);
    Store(user,config);
  }

  static void RegisterCpu(JSphCpuSingle& s);
  static void CheckClosedCpu(JSphCpuSingle& s);
};

#endif


// Compile this section only from JSphCpuSingle_InOut.cpp.
// Keep this section outside the public include guard.
#if defined(USER_INOUT_BOUND_MODE_CPU_IMPLEMENTATION) \
    && !defined(USER_INOUT_BOUND_MODE_CPU_DEFINED)
#define USER_INOUT_BOUND_MODE_CPU_DEFINED

#include "JSphCpuSingle.h"
#include "JCellDivCpuSingle.h"
#include "UserParameters.h"
#include "UserInOutData.h"
#include "JDsPartsInit.h"
#include "JDsTimersCpu.h"
#include "JSimpleNeigs.h"
#include "JSphFlexStruc.h"

#include <set>

// Uses the same coefficient/time interpolation as prescribed uniform inlet velocity.
float UserInOutBoundMode::ConstraintSpeed(const JSphInOutZone* zone,double time){
  const auto* velocity=zone->GetInOutVel();
  const auto a=velocity->GetCurrentCoefs0(),b=velocity->GetCurrentCoefs1();
  if(zone->GetVelMode()==InVelM_Fixed)return a.x;
  const float t=float(time);
  if(t<=a.w || a.w==b.w)return a.x;
  if(t>=b.w)return b.x;
  return (t-a.w)/(b.w-a.w)*(b.x-a.x)+a.x;
}

void UserInOutBoundMode::PrepareConstraintCpu(JSphCpuSingle& s,bool force){
  if(Read(s.User.get()).NoBackflow!=2)return;
  const auto& state=Runtime(s.InOut);if(state.Enabled && !state.Open)return;
  const auto* zone=s.InOut->GetZone(0);const auto plane=zone->GetPlane();
  const double face=-double(plane.d)/plane.c,width=2*s.Dp;
  const double xmin=zone->GetBoxLimitMin().x,xmax=zone->GetBoxLimitMax().x;
  const bool predictor=s.InterStep==INTERSTEP_SymPredictor;
  const double dt=s.SymplecticDtPre*(predictor?.5:1.);
  const float u=ConstraintSpeed(zone,s.TimeStep);
  const auto* pos=s.Pos_c->cptr();const auto* code=s.Code_c->cptr();
  auto* vel=s.Velrho_c->ptr();
  const auto* pre=(!predictor && force?s.VelrhoPre_c->cptr():s.Velrho_c->cptr());
  auto* ace=(force?s.Ace_c->ptr():nullptr);
  const int begin=int(s.Npb),end=int(s.Np);
  #ifdef OMP_USE
    #pragma omp parallel for schedule(static) if(end-begin>OMP_LIMIT_COMPUTELIGHT)
  #endif
  for(int p=begin;p<end;p++){
    if(!CODE_IsNormal(code[p]) || !CODE_IsFluidNotInout(code[p]))continue;
    const double d=pos[p].z-face;
    if(d<0 || d>=width || pos[p].x<xmin || pos[p].x>xmax)continue;
    const float lower=u*float(1.-d/width);
    if(!force)vel[p].z=std::max(vel[p].z,lower);
    else ace[p].z=std::max(ace[p].z,float((double(lower)-pre[p].z)/dt)-s.Gravity.z);
  }
}

void UserInOutBoundMode::InitNoBackflowCpu(JSphCpuSingle& s){
  if(!Read(s.User.get()).NoBackflow)return;
  Require(s.InOut && s.Simulate2D && !s.PeriActive
      && s.TStep==STEP_Symplectic && !s.FtCount,
      "InOutNoBackflow requires CPU 2-D Symplectic, "
      "without periodic or floating bodies.");
  Require(s.InOut->GetCount()==1,
      "InOutNoBackflow currently supports one inlet zone.");

  UserInOutData::Check(*s.InOut);
  const auto* z=s.InOut->GetZone(0);
  const auto dir=z->GetDirection();
  const auto plane=z->GetPlane();
  Require(std::abs(dir.x)<1e-12 && std::abs(dir.y)<1e-12
      && std::abs(dir.z-1.0)<1e-12
      && plane.a==0 && plane.b==0 && plane.c>0,
      "InOutNoBackflow requires a horizontal upward inlet.");
  Require((z->GetConfigUpdate()&JSphInOutZone::RefillSpFull_MASK)
      && !(z->GetConfigUpdate()&JSphInOutZone::RemoveZsurf_MASK),
      "InOutNoBackflow requires refilling=0 and removezsurf=false.");
  Require(z->GetInOutVel()->UseCoefficients()
      && std::isfinite(z->GetInOutVel()->GetVelMin())
      && z->GetInOutVel()->GetVelMin()>=0,
      "InOutNoBackflow requires prescribed nonnegative inlet velocity.");
  if(Read(s.User.get()).NoBackflow==2){
    Require(z->GetInOutVel()->GetVelProfile()==InVelP_Uniform && z->GetRhopMode()==InRhop_Extrapolated,
        "InOutNoBackflow=2 requires uniform prescribed velocity and imposerhop mode=2.");
    Require(s.InOut->GetDetermLimit()>=1000.f,
        "InOutNoBackflow=2 currently requires zeroth-order density extrapolation (determlimit>=1000).");
    Require(!s.Shifting && !s.ShiftingAdv,
        "InOutNoBackflow=2 currently requires Shifting=0.");
    s.Log->Print("InOutNoBackflow=2: two-Dp inflow transition; velocity enters continuity and force evaluation; no post-step position clipping.");
    return;
  }
  s.Log->Print("InOutNoBackflow: ordinary fluid is constrained at the "
      "inlet plane; inlet buffers retain their prescribed state.");
}

void UserInOutBoundMode::ApplyNoBackflowCpu(JSphCpuSingle& s){
  if(Read(s.User.get()).NoBackflow!=1)return;
  auto& state=Runtime(s.InOut);
  if(state.Enabled && !state.Open)return;
  Require(s.PosPre_c->Active(),
      "InOutNoBackflow: stage origin positions are unavailable.");
  const auto* oldpos=s.PosPre_c->cptr();
  const auto* codes=s.Code_c->cptr();
  auto* pos=s.Pos_c->ptr();
  auto* velrho=s.Velrho_c->ptr();
  auto* dcell=s.Dcell_c->ptr();
  auto* mutablecode=s.Code_c->ptr();

  const auto* zone=s.InOut->GetZone(0);
  const auto lo=zone->GetBoxLimitMin();
  const auto hi=zone->GetBoxLimitMax();
  const auto plane=zone->GetPlane();
  const double face=-double(plane.d)/double(plane.c);
  const double eps=std::max(s.Dp*1e-4,
      8.0*std::numeric_limits<float>::epsilon()
      *std::max(1.0,std::abs(face)));
  Require(eps<s.Dp*0.05,
      "InOutNoBackflow: coordinates are too large relative to Dp.");

  for(unsigned p=s.Npb;p<s.Np;p++){
    const auto code=codes[p];
    if(!CODE_IsNormal(code) || !CODE_IsFluidNotInout(code))continue;

    const auto now=pos[p];
    const double d1=now.z-face;
    if(d1>=0)continue;
    // Read the stage origin only for particles below the inlet face.
    const auto old=oldpos[p];
    const double d0=old.z-face;
    if(d0<0)continue;
    const double alpha=d0/(d0-d1);
    const double crossx=old.x+alpha*(now.x-old.x);
    if(crossx<double(lo.x) || crossx>double(hi.x))continue;

    s.UpdatePos(now,0,0,face+eps-now.z,false,p,
        pos,dcell,mutablecode);
    auto& vr=velrho[p];
    if(vr.z<0)vr.z=0;
  }
}


void UserInOutBoundMode::CheckNoBackflowProximityCpu(
    JSphCpuSingle& s,unsigned newnp,std::vector<typecode>& checked)
{
  InitNoBackflowCpu(s);
  Require(newnp<=s.Np && checked.size()==s.Np,
      "InOutNoBackflow: invalid initial particle list.");
  const auto* z=s.InOut->GetZone(0);
  const auto lo=z->GetBoxLimitMin();
  const auto hi=z->GetBoxLimitMax();
  const auto plane=z->GetPlane();
  const double face=-double(plane.d)/double(plane.c);
  const double tol=s.Dp*1e-6;
  std::vector<unsigned> keep;

  // Exempt only ordinary fluid on the valid side of this inlet.
  for(unsigned p=0;p<s.Np-newnp;p++){
    const auto code=checked[p];
    if(!CODE_IsNormal(code) || !CODE_IsFluidNotInout(code))continue;
    const auto pos=s.Pos_c->cptr()[p];
    const float x=float(pos.x);
    if(x<lo.x || x>hi.x || pos.z<face-tol)continue;
    keep.push_back(p);
    checked[p]=CODE_SetOutIgnore(code);
  }

  // Keep an independent minimum-distance check against the new buffers.
  JSimpleNeigs neigs(s.Np,s.Pos_c->cptr(),s.Scell);
  for(unsigned p=s.Np-newnp;p<s.Np;p++){
    const unsigned count=neigs.NearbyPositions(
        s.Pos_c->cptr()[p],p,s.Dp*0.5);
    const unsigned* list=neigs.GetSelectPos();
    for(unsigned i=0;i<count;i++){
      const auto code=s.Code_c->cptr()[list[i]];
      Require(!(CODE_IsNormal(code) && CODE_IsFluidNotInout(code)),
          "InOutNoBackflow: ordinary fluid is within 0.5*Dp "
          "of a new buffer; adjust the initial state or inlet geometry.");
    }
  }

  s.InOut->InitCheckProximity(s.Np,newnp,s.Scell,
      s.Pos_c->cptr(),s.Idp_c->cptr(),checked.data());
  for(unsigned p:keep)checked[p]=s.Code_c->cptr()[p];
}

void UserInOutBoundMode::RegisterCpu(JSphCpuSingle& s){
  Require(
      s.InOut && s.Simulate2D && !s.PeriActive && !s.FtCount,
      "CPU valve requires 2-D, no VRes, periodic or floating bodies.");

  Require(
      s.User->SaveInOutData && (s.SvData&SDAT_Binx)
      && s.SvPosDouble && !s.OutputParts,
      "requires SaveInOutData=1, full binary output "
      "and SavePosDouble=1.");

  Require(
      s.TBoundary==BC_MDBC && s.UseNormals,
      "this version requires mDBC and boundary normals.");

  Require(
      s.InOut->GetCount()==1,
      "this version supports one inlet zone.");

  if(s.User->FlexBoundContactMode==2){
    s.Log->PrintWarning(
        "InOut valve switches fixed particles only; "
        "the independent analytic Flex contact surface remains active.");
  }

  Require(
      !(s.User->FlexRestartBoundaryCorrectionMode==5
        && s.User->FlexRestartBoundaryCorrectionBoundaryForm==3),
      "boundarylayercorrection mode=5 with boundaryform=3 "
      "is not supported by this valve version.");

  UserInOutData::Check(*s.InOut);

  const auto* z=s.InOut->GetZone(0);

  Require(
      (z->GetConfigUpdate()&JSphInOutZone::RefillSpFull_MASK)
      && !(z->GetConfigUpdate()&JSphInOutZone::RemoveZsurf_MASK),
      "this version requires refilling=0 and removezsurf=false.");

  const auto dir=z->GetDirection();

  Require(
      std::abs(dir.x)<1e-12
      && std::abs(dir.y)<1e-12
      && std::abs(dir.z-1)<1e-12,
      "this version requires a horizontal bottom inlet directed upward.");

  auto& v=Runtime(s.InOut);
  v.Enabled=true;
  v.Open=(Read(s.User.get()).ValveStatue==1);

  const auto* pc=s.PartsInit->GetCode();
  const auto* pp=s.PartsInit->GetPos();

  const int requested=Read(s.User.get()).ValveMkBound;
  Require(requested>=0 && requested<=65535,
      "ValveMkBound is missing or invalid.");

  const unsigned block=
      s.MkInfo->GetMkBlockByMkBound(word(requested));
  Require(block<s.MkInfo->Size(),
      "ValveMkBound does not exist in the case.");

  const auto* gate=s.MkInfo->Mkblock(block);
  Require(
      gate->Bound && gate->Count>0
      && CODE_IsFixed(gate->Code)
      && !CODE_IsFlexStrucAny(gate->Code),
      "ValveMkBound must select a nonempty fixed boundary MK.");

  const typecode gatecode=CODE_GetTypeAndValue(gate->Code);
  v.Codes.assign(1,gatecode);

  bool intersects=false;
  for(unsigned p=0;p<s.PartsInit->GetNp();p++){
    if(CODE_IsFluid(pc[p])
        || !z->InZone(true,ToTFloat3(pp[p])))
      continue;

    Require(
        CODE_GetTypeAndValue(pc[p])==gatecode,
        "inlet intersects a boundary outside ValveMkBound; "
        "adjust the geometry or the specified MK.");
    intersects=true;
  }

  Require(intersects,
      "ValveMkBound has no particles inside the inlet zone.");
  v.GeometryHash=14695981039346656037ULL;

  for(unsigned p=0;p<s.PartsInit->GetNp();p++){
    if(!v.IsGate(pc[p]))continue;

    const auto lo=z->GetBoxLimitMin();
    const auto hi=z->GetBoxLimitMax();

    Require(
        pp[p].x>=double(lo.x)-s.Dp*0.01
        && pp[p].x<=double(hi.x)+s.Dp*0.01
        && pp[p].z<double(hi.z)+s.Dp*0.01,
        "selected mkbound extends outside the gate footprint; "
        "give the gate its own MK.");

    const byte* b=reinterpret_cast<const byte*>(&pp[p]);

    for(size_t k=0;k<sizeof(tdouble3);k++){
      v.GeometryHash^=b[k];
      v.GeometryHash*=1099511628211ULL;
    }
  }

  for(typecode c:v.Codes){
    const unsigned block=s.MkInfo->GetMkBlockByCode(c);

    Require(
        block<s.MkInfo->Size(),
        "unsupported boundary code in valve region.");

    const auto* b=s.MkInfo->Mkblock(block);

    if(s.FlexStruc){
      for(unsigned i=0;i<s.FlexStruc->GetCount();i++){
        const auto* body=s.FlexStruc->GetBody(i);

        Require(
            b->MkType!=body->MkBound
            && std::find(
                body->MkClamp.begin(),
                body->MkClamp.end(),
                b->MkType)==body->MkClamp.end(),
            "an elastic body or clamp cannot be used as a valve wall.");
      }
    }

    unsigned count=0;
    unsigned normals=0;

    for(unsigned p=0;p<s.Np;p++){
      if(!v.IsGate(s.Code_c->cptr()[p])
          || CODE_GetTypeAndValue(s.Code_c->cptr()[p])!=c)
        continue;

      Require(
          s.Idp_c->cptr()[p]<s.CaseNfixed
          && CODE_IsNormal(s.Code_c->cptr()[p]),
          "gate must consist of retained fixed particles.");

      const auto n=s.BoundNor_c->cptr()[p];

      if(n.x!=0 || n.y!=0 || n.z!=0){
        normals++;

        Require(
            n.z>0
            && std::abs(n.x)<=s.Dp*1e-6
            && std::abs(n.y)<=s.Dp*1e-6,
            "gate requires explicitly assigned upward mDBC normals.");

        const auto plane=z->GetPlane();

        // BoundNor is the full displacement to the ghost node.
        const double wallz=
            s.Pos_c->cptr()[p].z+0.5*double(n.z);

        Require(
            std::abs(double(plane.c)*wallz+double(plane.d))
                <s.Dp*1e-3,
            "inlet plane and mDBC wall plane do not coincide.");
      }

      count++;
    }

    Require(normals>0,"gate MK has no mDBC normals.");

    Require(
        count==b->Count,
        "gate particles are missing from the input PART.");

    s.Log->Printf(
        "InOut valve: mkbound=%u, count=%u, target=%s.",
        b->MkType,count,v.Open?"inlet":"wall");
  }
}

void UserInOutBoundMode::CheckClosedCpu(JSphCpuSingle& s){
  const auto& v=Runtime(s.InOut);
  const auto* zone=s.InOut->GetZone(0);
  const auto lo=zone->GetBoxLimitMin();
  const auto hi=zone->GetBoxLimitMax();
  const double tol=s.Dp*1e-6;

  // Only an open-to-closed particle restart may remove ordinary fluid.
  const bool closing=s.PartBegin && v.HistoryOpen && !v.Open;

  bool found=false;
  double topz=0.0;
  double wallz=0.0;
  double bottomz=double(lo.z);

  for(unsigned p=0;p<s.Np;p++){
    if(!v.IsGate(s.Code_c->cptr()[p]))continue;

    const auto pos=s.Pos_c->cptr()[p];
    const auto normal=s.BoundNor_c->cptr()[p];
    bottomz=std::min(bottomz,pos.z);

    // BoundNor stores the full displacement to the mDBC ghost node.
    // Use the uppermost valid gate layer to avoid the float inlet plane.
    if(normal.z>0 && (!found || pos.z>topz)){
      found=true;
      topz=pos.z;
      wallz=pos.z+0.5*double(normal.z);
    }
  }

  Require(found,"cannot determine the physical valve wall plane.");

  unsigned removed=0;
  for(unsigned p=0;p<s.Np;p++){
    const auto code=s.Code_c->cptr()[p];
    if(!CODE_IsNormal(code) || !CODE_IsFluidNotInout(code))continue;

    const auto pos=s.Pos_c->cptr()[p];
    const bool inside=
        pos.x>=double(lo.x)-tol && pos.x<=double(hi.x)+tol
        && pos.z>=bottomz-tol && pos.z<wallz-tol;
    if(!inside)continue;

    Require(closing,
        "ordinary fluid is inside the closed valve region; "
        "cleanup is allowed only on an open-to-closed restart.");

    s.Code_c->ptr()[p]=CODE_SetOutIgnore(code);
    removed++;
  }

  if(closing){
    s.Log->Printf(
        "InOut valve close cleanup: ordinary fluid removed=%u, "
        "mass=%.12g kg (solver particle mass); wall z=%.12g m.",
        removed,double(removed)*double(s.MassFluid),wallz);
  }
}

void UserInOutBoundMode::InitCpu(JSphCpuSingle& s){
  auto* io=s.InOut;
  io->Nstep=s.Nstep;

  const unsigned initial=io->Config(
      s.PartBegin?0:s.TimeStepIni,
      s.Stable,s.PeriActive,
      s.MapRealPosMin,s.MapRealPosMax,
      s.MkInfo->GetCodeNewFluid(),s.PartsInit,s.GaugeSystem);

  RegisterCpu(s);
  auto& v=Runtime(io);

  if(s.PartBegin){
    JPartDataBi4 main;
    main.LoadFilePart(s.PartBeginDir,s.PartBegin);

    Require(
        main.Get_TimeStep()==s.TimeStep && !main.Get_PosSimple(),
        "restart requires absolute time and double positions.");

    const std::string dir=fun::GetDirWithSlash(s.PartBeginDir);

    JBinaryData head("PartHeadInOut");
    JBinaryData frame("PartInOut");

    head.LoadFile(
        dir+"PartHeadInOut.ibi4","PartHeadInOut");

    frame.LoadFile(
        dir+fun::FileNameSec("PartInOut.bi4",s.PartBegin),
        "PartInOut");

    UserInOutData::Restore(
        *io,*s.MkInfo,head,frame,main,s.PartBegin,
        s.Np,s.Idp_c->cptr(),s.Code_c->ptr());

    s.Nstep=main.Get_Step();
    s.TotalNp=main.Get_NpTotal();
    s.IdMax=unsigned(s.TotalNp-1);

    Require(
        v.HistoryOpen || io->GetCurrentNp()==0,
        "closed checkpoint contains inlet buffers.");
  }
  else{
    s.InOutIgnoreFluidDef(io->MkFluidList,s.Code_c->ptr());
  }

  if(!v.Open){
    CheckClosedCpu(s);

    for(unsigned p=0;p<s.Np;p++){
      if(CODE_IsFluidInout(s.Code_c->cptr()[p])){
        s.Code_c->ptr()[p]=
            CODE_SetOutIgnore(s.Code_c->cptr()[p]);
      }
    }

    io->SetCurrentNp(0);
  }

  s.CellDivSingle->SetValveCodes(
      v.Open?v.Codes:std::vector<typecode>());

  s.CellDivSingle->SetIncreaseNp(io->GetNpResizePlus1());

  s.BoundChanged=true;
  s.RunCellDivide(true);

  const bool create=
      v.Open && (!s.PartBegin || !v.HistoryOpen);

  if(create){
    Require(initial>0,"inlet geometry creates no buffers.");

    Require(
        s.TotalNp+initial<=UINT_MAX,
        "particle IDs would overflow.");

    const unsigned old=s.Np;

    s.ResizeParticlesSizeData(
        old,old+initial+io->GetNpResizePlus0(),
        old+initial,0,false);

    io->LoadInitPartsData(
        s.IdMax+1,initial,
        s.Idp_c->ptr()+old,s.Code_c->ptr()+old,
        s.Pos_c->ptr()+old,s.Velrho_c->ptr()+old);

    for(unsigned p=old;p<old+initial;p++){
      s.UpdatePos(
          s.Pos_c->cptr()[p],0,0,0,false,p,
          s.Pos_c->ptr(),s.Dcell_c->ptr(),s.Code_c->ptr());
    }

    if(s.SpsTauRho2_c)
      s.SpsTauRho2_c->MemsetOffset(old,0,initial);

    if(s.BoundNor_c)
      s.BoundNor_c->MemsetOffset(old,0,initial);

    if(s.FSType_c)
      s.FSType_c->MemsetOffset(old,3,initial);

    #ifdef AVAILABLE_DIVCLEAN
    if(s.PsiClean_c)
      s.PsiClean_c->MemsetOffset(old,0,initial);
    #endif

    s.Np+=initial;
    s.TotalNp+=initial;
    s.IdMax=unsigned(s.TotalNp-1);
    io->AddNewNp(initial);

    // Check with a temporary code array.
    // Only the dormant valve wall is exempted.
    std::vector<typecode> checked(
        s.Code_c->cptr(),s.Code_c->cptr()+s.Np);

    for(unsigned p=0;p<s.Np;p++){
      if(v.IsGate(checked[p]))
        checked[p]=CODE_SetOutIgnore(checked[p]);
    }

    if(Read(s.User.get()).NoBackflow)
      CheckNoBackflowProximityCpu(s,initial,checked);
    else{
      io->InitCheckProximity(
          s.Np,initial,s.Scell,
          s.Pos_c->cptr(),s.Idp_c->cptr(),checked.data());
    }

    for(unsigned p=0;p<s.Np;p++){
      if(!v.IsGate(s.Code_c->cptr()[p])){
        Require(
            checked[p]==s.Code_c->cptr()[p],
            "new inlet buffers overlap ordinary fluid; "
            "restart refused.");
      }
    }

    s.BoundChanged=true;
    s.RunCellDivide(true);
  }

  if(!s.PartBegin || create){
    io->UpdateVelData(s.TimeStepIni);
    io->UpdateZsurfData(s.TimeStepIni,true);

    if(v.Open)
      s.InOutUpdatePartsData(s.TimeStepIni);
  }

  s.NpfMinimum=unsigned(
      s.MinFluidStop
      *(s.Np-s.NpbPer-s.NpfPer-s.CaseNbound));

  io->VisuConfig("\nInOut valve configuration:"," ");
}

#endif