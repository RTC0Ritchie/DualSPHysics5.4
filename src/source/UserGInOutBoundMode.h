#ifndef USER_G_INOUT_BOUND_MODE_H
#define USER_G_INOUT_BOUND_MODE_H
#include "UserInOutBoundMode.h"
#include "UserGIceProtection.h"
class JSphGpuSingle;
class UserGInOutBoundMode final {
  static void Download(JSphGpuSingle& s,bool normals=false);
  static void RegisterGpu(JSphGpuSingle& s);
  static void CheckClosedGpu(JSphGpuSingle& s);
  static void Require(bool ok,const char* msg){UserInOutBoundMode::Require(ok,msg);}
  static UserInOutBoundMode::StConfig Read(const UserParameters* u){return UserInOutBoundMode::Read(u);}
  static UserInOutBoundMode::StRuntime& Runtime(const JSphInOut* io){return UserInOutBoundMode::Runtime(io);}
public:
  static void InitGpu(JSphGpuSingle& s);
  static void InitNoBackflowGpu(JSphGpuSingle& s);
  static void ApplyNoBackflowGpu(JSphGpuSingle& s);
  static void CheckNoBackflowProximityGpu(JSphGpuSingle& s,unsigned newnp,std::vector<typecode>& checked);
  static void Launch(unsigned np,unsigned nb,double face,double eps,double xmin,double xmax,
      const double2* oldxy,const double* oldz,double2* xy,double* z,float4* vel,
      typecode* code,unsigned* cells,UserGIceProtection::Domain domain);
};
#endif
#ifdef USER_G_INOUT_BOUND_MODE_IMPLEMENTATION
#include "JSphGpuSingle.h"
#include "JCellDivGpuSingle.h"
#include "JDsPartsInit.h"
#include "JPartsLoad4.h"
#include "JSphFlexStruc.h"
#include "JSphInOut.h"
#include "JSphInOutZone.h"
#include "JSphInOutVel.h"
#include "UserInOutData.h"
#include "UserParameters.h"
#include "JSimpleNeigs.h"
#include "JSphGpu_InOut_iker.h"
#include <climits>
void UserGInOutBoundMode::Download(JSphGpuSingle& s,bool normals){
  unsigned filtered=0;
  Require(s.Np==s.ParticlesDataDown(s.Np,0,true,false,nullptr,filtered),"GPU initialization particle download mismatch.");
  if(normals){Require(s.BoundNor_g!=nullptr,"valve requires boundary normals.");s.BoundNor_g->CuCopyToHost(s.AuxVel_c,s.Np);}
}
void UserGInOutBoundMode::InitNoBackflowGpu(JSphGpuSingle& s){
  if(!Read(s.User.get()).NoBackflow)return;
  Require(s.InOut && s.Simulate2D && !s.PeriActive
      && s.TStep==STEP_Symplectic && !s.FtCount,
      "InOutNoBackflow requires GPU 2-D Symplectic, "
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
  s.Log->Print("InOutNoBackflow: ordinary fluid is constrained at the "
      "inlet plane; inlet buffers retain their prescribed state.");
}

void UserGInOutBoundMode::CheckNoBackflowProximityGpu(
    JSphGpuSingle& s,unsigned newnp,std::vector<typecode>& checked)
{
  InitNoBackflowGpu(s);
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
    const auto pos=s.AuxPos_c->cptr()[p];
    const float x=float(pos.x);
    if(x<lo.x || x>hi.x || pos.z<face-tol)continue;
    keep.push_back(p);
    checked[p]=CODE_SetOutIgnore(code);
  }

  // Keep an independent minimum-distance check against the new buffers.
  JSimpleNeigs neigs(s.Np,s.AuxPos_c->cptr(),s.Scell);
  for(unsigned p=s.Np-newnp;p<s.Np;p++){
    const unsigned count=neigs.NearbyPositions(
        s.AuxPos_c->cptr()[p],p,s.Dp*0.5);
    const unsigned* list=neigs.GetSelectPos();
    for(unsigned i=0;i<count;i++){
      const auto code=s.Code_c->cptr()[list[i]];
      Require(!(CODE_IsNormal(code) && CODE_IsFluidNotInout(code)),
          "InOutNoBackflow: ordinary fluid is within 0.5*Dp "
          "of a new buffer; adjust the initial state or inlet geometry.");
    }
  }

  s.InOut->InitCheckProximity(s.Np,newnp,s.Scell,
      s.AuxPos_c->cptr(),s.Idp_c->cptr(),checked.data());
  for(unsigned p:keep)checked[p]=s.Code_c->cptr()[p];
}

void UserGInOutBoundMode::RegisterGpu(JSphGpuSingle& s){
  Require(
      s.InOut && s.Simulate2D && !s.PeriActive && !s.FtCount,
      "GPU valve requires 2-D, no VRes, periodic or floating bodies.");

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

      const auto n=s.AuxVel_c->cptr()[p];

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
            s.AuxPos_c->cptr()[p].z+0.5*double(n.z);

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

void UserGInOutBoundMode::CheckClosedGpu(JSphGpuSingle& s){
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

    const auto pos=s.AuxPos_c->cptr()[p];
    const auto normal=s.AuxVel_c->cptr()[p];
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

    const auto pos=s.AuxPos_c->cptr()[p];
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

void UserGInOutBoundMode::ApplyNoBackflowGpu(JSphGpuSingle& s){
  if(!Read(s.User.get()).NoBackflow)return;
  const auto& state=Runtime(s.InOut);
  if(state.Enabled && !state.Open)return;
  Require(s.PosxyPre_g->Active() && s.PoszPre_g->Active(),"GPU stage origin positions unavailable.");
  const auto* zone=s.InOut->GetZone(0);const auto plane=zone->GetPlane();
  const double face=-double(plane.d)/double(plane.c);
  const double eps=std::max(s.Dp*1e-4,8.0*std::numeric_limits<float>::epsilon()*std::max(1.0,std::abs(face)));
  Require(eps<s.Dp*0.05,"InOutNoBackflow: coordinates too large relative to Dp.");
  Launch(s.Np,s.Npb,face,eps,zone->GetBoxLimitMin().x,zone->GetBoxLimitMax().x,
      s.PosxyPre_g->cptr(),s.PoszPre_g->cptr(),s.Posxy_g->ptr(),s.Posz_g->ptr(),
      s.Velrho_g->ptr(),s.Code_g->ptr(),s.Dcell_g->ptr(),
      {s.MapRealPosMin,s.MapRealSize,s.Scell,s.MovLimit,s.DomCellCode});
}
void UserGInOutBoundMode::InitGpu(JSphGpuSingle& s){
  auto* io=s.InOut;io->Nstep=s.Nstep;
  const unsigned initial=io->Config(s.PartBegin?0:s.TimeStepIni,s.Stable,s.PeriActive,
      s.MapRealPosMin,s.MapRealPosMax,s.MkInfo->GetCodeNewFluid(),s.PartsInit,s.GaugeSystem);
  Download(s,true);RegisterGpu(s);auto& v=Runtime(io);
  if(s.PartBegin){
    JPartDataBi4 main;main.LoadFilePart(s.PartBeginDir,s.PartBegin);
    Require(main.Get_TimeStep()==s.TimeStep && !main.Get_PosSimple(),"restart requires absolute time and double positions.");
    const std::string dir=fun::GetDirWithSlash(s.PartBeginDir);
    JBinaryData head("PartHeadInOut"),frame("PartInOut");
    head.LoadFile(dir+"PartHeadInOut.ibi4","PartHeadInOut");
    frame.LoadFile(dir+fun::FileNameSec("PartInOut.bi4",s.PartBegin),"PartInOut");
    UserInOutData::Restore(*io,*s.MkInfo,head,frame,main,s.PartBegin,s.Np,s.Idp_c->cptr(),s.Code_c->ptr());
    s.Nstep=main.Get_Step();s.TotalNp=main.Get_NpTotal();s.IdMax=unsigned(s.TotalNp-1);
    Require(v.HistoryOpen || io->GetCurrentNp()==0,"closed checkpoint contains inlet buffers.");
  }else{
    s.InOutIgnoreFluidDef(io->MkFluidList,s.Code_g->ptr());s.Code_g->CuCopyToHost(s.Code_c,s.Np);
  }
  if(!v.Open){
    CheckClosedGpu(s);
    for(unsigned p=0;p<s.Np;p++)if(CODE_IsFluidInout(s.Code_c->cptr()[p]))s.Code_c->ptr()[p]=CODE_SetOutIgnore(s.Code_c->cptr()[p]);
    io->SetCurrentNp(0);
  }
  s.Code_g->CuCopyFromHost(s.Code_c,s.Np);
  s.CellDivSingle->SetValveCode(v.Open,v.Codes.front());
  s.CellDivSingle->SetIncreaseNp(io->GetNpResizePlus1());
  s.BoundChanged=true;s.RunCellDivide(true);
  const bool create=v.Open && (!s.PartBegin || !v.HistoryOpen);
  if(create){
    Require(initial>0,"inlet geometry creates no buffers.");
    Require(s.TotalNp+initial<=UINT_MAX,"particle IDs would overflow.");
    const unsigned old=s.Np;
    s.ResizeParticlesSizeData(0,old,old+initial+io->GetNpResizePlus0(),old+initial,0,false);
    io->LoadInitPartsData(s.IdMax+1,initial,s.Idp_c->ptr(),s.Code_c->ptr(),s.AuxPos_c->ptr(),s.Velrho_c->ptr());
    s.Pos3ToPos21(initial,s.AuxPos_c->cptr(),s.Posxy_c->ptr(),s.Posz_c->ptr());
    s.Idp_g->CuCopyFromHostOffset(old,s.Idp_c,0,initial);
    s.Code_g->CuCopyFromHostOffset(old,s.Code_c,0,initial);
    s.Posxy_g->CuCopyFromHostOffset(old,s.Posxy_c,0,initial);
    s.Posz_g->CuCopyFromHostOffset(old,s.Posz_c,0,initial);
    s.Velrho_g->CuCopyFromHostOffset(old,s.Velrho_c,0,initial);
    cusphinout::UpdatePosFluid(s.PeriActive,initial,old,s.Posxy_g->ptr(),s.Posz_g->ptr(),s.Dcell_g->ptr(),s.Code_g->ptr());
    if(s.SpsTauRho2_g)s.SpsTauRho2_g->CuMemsetOffset(old,0,initial);
    if(s.BoundNor_g)s.BoundNor_g->CuMemsetOffset(old,0,initial);
    if(s.FSType_g)s.FSType_g->CuMemsetOffset(old,3,initial);
    #ifdef AVAILABLE_DIVCLEAN
    if(s.PsiClean_g)s.PsiClean_g->CuMemsetOffset(old,0,initial);
    #endif
    s.Np+=initial;s.TotalNp+=initial;s.IdMax=unsigned(s.TotalNp-1);io->AddNewNp(initial);
    Download(s);
    std::vector<typecode> checked(s.Code_c->cptr(),s.Code_c->cptr()+s.Np);
    for(unsigned p=0;p<s.Np;p++)if(v.IsGate(checked[p]))checked[p]=CODE_SetOutIgnore(checked[p]);
    if(Read(s.User.get()).NoBackflow)CheckNoBackflowProximityGpu(s,initial,checked);
    else io->InitCheckProximity(s.Np,initial,s.Scell,s.AuxPos_c->cptr(),s.Idp_c->cptr(),checked.data());
    for(unsigned p=0;p<s.Np;p++)if(!v.IsGate(s.Code_c->cptr()[p]))Require(checked[p]==s.Code_c->cptr()[p],"new buffers overlap ordinary fluid; restart refused.");
    s.BoundChanged=true;s.RunCellDivide(true);
  }
  if(!s.PartBegin || create){io->UpdateVelData(s.TimeStepIni);io->UpdateZsurfData(s.TimeStepIni,true);if(v.Open)s.InOutUpdatePartsData(s.TimeStepIni);}
  s.NpfMinimum=unsigned(s.MinFluidStop*(s.Np-s.NpbPer-s.NpfPer-s.CaseNbound));
  io->VisuConfig("\nGPU InOut valve configuration:"," ");
}
#endif
