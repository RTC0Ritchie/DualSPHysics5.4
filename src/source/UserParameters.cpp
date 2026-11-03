#include "UserParameters.h"
#include "JXml.h"
#include "JLog2.h"
#include "JCaseEParms.h"
#include "Functions.h"
#include "UserFlexBoundaryCorrection.h"
#include "UserIceBottomProtection.h"
#include "UserIceTopProtection.h"

#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <cmath>
#include <limits>

UserParameters::UserParameters(){
  ClassName="UserParameters";
  Reset();
}


void UserParameters::Reset(){
  FlexBoundContactMode=1;

  SurfaceType=StSurface2d{};

  KnContactMode=1;
  KnContact=0.0;
  KnContactLinearity=1;

  unContactMode=1;
  unContact=0.0;

  ElasticHighFrequencyDissipation=false;
  ElasticViscousDissipation=false;
  ElasticViscousFactor=0.5;

  NoPenetrationMkEnabled=false;
  NoPenetrationMkBound=-1;

  IceTopNoPenetrationEnabled=false;
  IceRightNoPen::Reset(this);
  IceBottomNoPen::Reset(this);

  Mdbc2AdaptiveEnabled=false;
  Mdbc2AdaptiveMode=0;
  Mdbc2AdaptiveHead=0;
  Mdbc2AdaptiveSupport=0;
  Mdbc2AdaptiveHistory.clear();
  Mdbc2AdaptiveAce.clear();

  IceTopFluidCleanupEnabled=false;

  SaveFlexData=false;
  SaveFlexPairData=false;
  SaveInOutData=false;
  InOutHeadWritten=false;

  FlexRestartRequested=false;
  FlexRestartSourceRequested=0;

  FlexRestartElasticInput=0;
  FlexRestartElasticFile.clear();
  FlexRestartFileVariableType=0;
  FlexRestartSpecialField=0;
  FlexRestartSeaLevel=0.0;
  FlexRestartColumnGapCoef=0.1;
  FlexRestartConserveMass=true;
  FlexRestartDensityIsNatural=true;

  FlexRestartBoundaryCorrectionMode=0;
  FlexRestartBoundaryCorrectionLayers=0;
  FlexRestartBoundaryCorrectionLimit=1.5;
  FlexRestartBoundaryCorrectionInwardReach=0.0;
  FlexRestartBoundaryCorrectionMaxIterations=2000;
  FlexRestartBoundaryCorrectionQuadraticError=0.0;
  FlexRestartBoundaryCorrectionBoundaryForm=1;

  NoPenetrationMkIdBegin=0;
  NoPenetrationMkIdEnd=0;

  FlexRestartRestorePairs=false;
  FlexRestartGradientCorrection=false;
  FlexRestartHourglassCorrection=false;

  OutputObjects.clear();

  ElasticDissipation.reset();


  FlexRestartState.Clear();
  
  UseFlexRestartRestorePairs=false;
  UseFlexRestartGradientCorrection=false;
  UseFlexRestartHourglassCorrection=false;

  FlexRestartBoundaryData=StFlexRestartBoundaryData{};

  FlexPairDataWritten=false;
  FlexMaterialDataWritten=false;
  FlexReferenceId.clear();
  FlexRestartWriterRunCode.clear();

  FlexRestartSource=TpFlexRestartSource::None;
}

void UserParameters::LoadXml(const JXml* xml,JLog2* log){
  // Recover to alldefault value at the beginning of each load, in case of multiple loads.
  Reset();

  // Load the user parameters from the XML file. If the node is missing, use default configuration.
  const TiXmlNode* node=xml->GetNodeSimple("case.User");
  if(!node){
    if(log){
      log->Printf("User configuration not found; using defaults.");
    }
    return;
  }
  const TiXmlElement* root=node->ToElement();
  xml->CheckElementNames(
      root,true,
      "*parameter output restart directcontinuation boundarylayercorrection elasticdissipation");

  if(const TiXmlElement* damping=root->FirstChildElement("elasticdissipation")){
    if(damping->NextSiblingElement("elasticdissipation"))
      Run_Exceptioon("Repeated User.elasticdissipation node.");

    xml->CheckAttributeNames(damping,"comment");
    xml->CheckElementNames(damping,true,"highfrequency viscous");

    const auto child=[&](const char* name)->const TiXmlElement* {
      const TiXmlElement* p=damping->FirstChildElement(name);

      if(p && (p->NextSiblingElement(name) || p->FirstChildElement()))
        Run_Exceptioon("Elastic dissipation child must be a unique leaf.");

      return p;
    };

    if(const TiXmlElement* p=child("highfrequency")){
      xml->CheckAttributeNames(p,"enabled comment");
      ElasticHighFrequencyDissipation=
          xml->GetAttributeBool(p,"enabled",true,false);
    }

    if(const TiXmlElement* p=child("viscous")){
      xml->CheckAttributeNames(p,"enabled factor comment");

      ElasticViscousDissipation=
          xml->GetAttributeBool(p,"enabled",true,false);
      ElasticViscousFactor=
          xml->GetAttributeDouble(p,"factor",true,0.5);

      if(!std::isfinite(ElasticViscousFactor)
          || ElasticViscousFactor<0)
        Run_Exceptioon(
            "KV factor must be finite and nonnegative.");
    }
  }

  // Record active parameter nodes and reject duplicate keys.
  std::map<std::string,const TiXmlElement*> nodes;
  for(const TiXmlElement* ele=root->FirstChildElement("parameter");
      ele;
      ele=ele->NextSiblingElement("parameter"))
  {
    if(!xml->GetAttributeBool(ele,"active",true,true))
      continue;

    const std::string key=xml->GetAttributeStr(ele,"key");
    if(!nodes.emplace(key,ele).second){
      Run_ExceptioonFile(
          std::string("Repeated User parameter: ")+key,
          xml->ErrGetFileRow(ele));
    }
  }

  // Read the ordinary key/value parameters.
  JCaseEParms parameters;
  // Only pass the "parameter" nodes
  for(const auto& item:nodes){
    const TiXmlElement* ele=item.second;

    std::string value=xml->GetAttributeStr(ele,"value");

    if(!value.empty() && value[0]=='#'
        && value.find(':')==std::string::npos){
      value=fun::DoubleStr(xml->GetAttributeDouble(ele,"value"));
    }

    parameters.Add(
        item.first,
        value,
        "true",
        xml->GetAttributeStr(ele,"comment",true),
        xml->GetAttributeStr(ele,"units_comment",true));
  }

  // ------------------------------------------------------------
  // Selective no-penetration.
  // ------------------------------------------------------------
  const auto nopenit=nodes.find("NoPenetrationMk");
  if(nopenit!=nodes.end()){
    const TiXmlElement* ele=nopenit->second;

    xml->CheckAttributeNames(
        ele,"key value mkbound active comment units_comment");

    if(ele->FirstChildElement())
      Run_Exceptioon(
          "User.NoPenetrationMk does not accept child elements.");

    const int enabled=xml->GetAttributeInt(ele,"value");
    if(enabled!=0 && enabled!=1)
      Run_Exceptioon(
          "User.NoPenetrationMk value must be 0 or 1.");

    NoPenetrationMkEnabled=(enabled==1);

    if(NoPenetrationMkEnabled){
      NoPenetrationMkBound=xml->GetAttributeInt(ele,"mkbound");

      if(NoPenetrationMkBound<0 || NoPenetrationMkBound>65535)
        Run_Exceptioon(
            "User.NoPenetrationMk mkbound must be in [0,65535].");
    }
  }

  // Modes: 0=off, 1=temporal, 2=support, 3=both.
  const auto mdbcit=nodes.find("Mdbc2Adaptive");
  if(mdbcit!=nodes.end()){
    const TiXmlElement* ele=mdbcit->second;
    xml->CheckAttributeNames(
        ele,"key value head support active comment units_comment");
    if(ele->FirstChildElement())
      Run_Exceptioon("User.Mdbc2Adaptive does not accept child elements.");
    Mdbc2AdaptiveMode=xml->GetAttributeInt(ele,"value");
    if(Mdbc2AdaptiveMode<0 || Mdbc2AdaptiveMode>3)
      Run_Exceptioon("User.Mdbc2Adaptive value must be 0, 1, 2 or 3.");
    Mdbc2AdaptiveEnabled=(Mdbc2AdaptiveMode!=0);
    if(Mdbc2AdaptiveMode&1){
      Mdbc2AdaptiveHead=xml->GetAttributeDouble(ele,"head");
      if(!std::isfinite(Mdbc2AdaptiveHead) || Mdbc2AdaptiveHead<=0)
        Run_Exceptioon("User.Mdbc2Adaptive head must be finite and positive.");
    }
    if(Mdbc2AdaptiveMode&2){
      Mdbc2AdaptiveSupport=xml->GetAttributeDouble(ele,"support");
      if(!std::isfinite(Mdbc2AdaptiveSupport)
          || Mdbc2AdaptiveSupport<0 || Mdbc2AdaptiveSupport>1)
        Run_Exceptioon("User.Mdbc2Adaptive support must be in [0,1].");
    }
  }

  const auto topnopen=nodes.find("IceTopNoPenetration");
  if(topnopen!=nodes.end()){
    const TiXmlElement* ele=topnopen->second;

    xml->CheckAttributeNames(
        ele,"key value rightside active comment units_comment");

    if(ele->FirstChildElement())
      Run_Exceptioon(
          "User.IceTopNoPenetration does not accept child elements.");

    const int enabled=xml->GetAttributeInt(ele,"value");

    if(enabled!=0 && enabled!=1)
      Run_Exceptioon(
          "User.IceTopNoPenetration value must be 0 or 1.");

    const int rightside=xml->GetAttributeInt(ele,"rightside",true,0);
    if(rightside!=0 && rightside!=1)
      Run_Exceptioon("User.IceTopNoPenetration rightside must be 0 or 1.");
    IceTopNoPenetrationEnabled=(enabled==1);
    IceRightNoPen::Get(this).Enabled=(enabled==1 && rightside==1);
  }

  const auto bottomnopen=nodes.find("IceBottomNoPenetration");
  if(bottomnopen!=nodes.end()){
    const TiXmlElement* ele=bottomnopen->second;

    xml->CheckAttributeNames(
        ele,"key value wettime hcoef active comment units_comment");

    if(ele->FirstChildElement())
      Run_Exceptioon(
          "User.IceBottomNoPenetration does not accept child elements.");

    const int enabled=xml->GetAttributeInt(ele,"value");

    if(enabled!=0 && enabled!=1)
      Run_Exceptioon(
          "User.IceBottomNoPenetration value must be 0 or 1.");

    auto& bottom=IceBottomNoPen::Get(this);
    bottom.Enabled=(enabled==1);

    if(bottom.Enabled){
      bottom.WetTime=
          xml->GetAttributeDouble(ele,"wettime",true,0.25);
      bottom.HCoef=
          xml->GetAttributeDouble(ele,"hcoef",true,4.0);

      if(!std::isfinite(bottom.WetTime) || bottom.WetTime<=0)
        Run_Exceptioon(
            "User.IceBottomNoPenetration wettime must be finite and positive.");

      if(!std::isfinite(bottom.HCoef) || bottom.HCoef<0)
        Run_Exceptioon(
            "User.IceBottomNoPenetration hcoef must be finite and non-negative.");
    }
  }

  // One-time removal of fluid above the current ice top.
  const auto cleanit=nodes.find("IceTopFluidCleanup");
  if(cleanit!=nodes.end()){
    const TiXmlElement* ele=cleanit->second;

    xml->CheckAttributeNames(
        ele,"key value active comment units_comment");

    if(ele->FirstChildElement())
      Run_Exceptioon(
          "User.IceTopFluidCleanup does not accept child elements.");

    const int enabled=xml->GetAttributeInt(ele,"value");

    if(enabled!=0 && enabled!=1)
      Run_Exceptioon(
          "User.IceTopFluidCleanup value must be 0 or 1.");

    IceTopFluidCleanupEnabled=(enabled==1);
  }
  // ------------------------------------------------------------
  // Contact algorithm.
  // ------------------------------------------------------------
  FlexBoundContactMode=
      parameters.GetValueInt("FlexBoundContactMode",true,1);

  if(FlexBoundContactMode<1 || FlexBoundContactMode>2){
    Run_Exceptioon(
        "User.FlexBoundContactMode must be 1 or 2.");
  }

  if(FlexBoundContactMode==2){ // Surface contact.
    const auto it=nodes.find("FlexBoundContactMode");
    if(it==nodes.end()){
      Run_Exceptioon(
          "Surface mode of User.FlexBoundContactMode node is missing.");
    }
    const TiXmlElement* contactNode=it->second;
    xml->CheckElementNames(contactNode,true,"SurfaceType");
    const TiXmlElement* surfaceNode=
        xml->GetFirstElement(contactNode,"SurfaceType");
    SurfaceType.value=
        xml->GetAttributeInt(surfaceNode,"value");
    if(SurfaceType.value<1 || SurfaceType.value>2){
      Run_Exceptioon(
          "User.SurfaceType must be 1 or 2.");
    }

    if(SurfaceType.value==1){ // Horizontal wall.
      xml->CheckElementNames(surfaceNode,true,"WallZ");
      // Missing WallZ or value causes an XML error.
      SurfaceType.WallZ=
          xml->ReadElementDouble(surfaceNode,"WallZ","value");
    }
    else if(SurfaceType.value==2){
      xml->CheckElementNames(
          surfaceNode,true,"point1 point2 point slope");

      SurfaceType.NormalDirectionMode=
          xml->GetAttributeInt(
              surfaceNode,"NormalDirectionMode",true,1);

      if(SurfaceType.NormalDirectionMode<1 ||
        SurfaceType.NormalDirectionMode>2){
        Run_Exceptioon(
            "User.SurfaceType.NormalDirectionMode must be 1 or 2.");
      }

      const bool hasP1=xml->ExistsElement(surfaceNode,"point1");
      const bool hasP2=xml->ExistsElement(surfaceNode,"point2");
      const bool hasPoint=xml->ExistsElement(surfaceNode,"point");
      const bool hasSlope=xml->ExistsElement(surfaceNode,"slope");

      // Exactly one complete definition is permitted.
      const bool useTwoPoints=
          hasP1 && hasP2 && !hasPoint && !hasSlope;

      const bool usePointSlope=
          hasPoint && hasSlope && !hasP1 && !hasP2;

      if(!useTwoPoints && !usePointSlope){
        Run_Exceptioon(
            "Specify either point1 + point2 or point + slope, "
            "without mixing the two definitions.");
      }

      if(useTwoPoints){
        const TiXmlElement* p1=
            xml->GetFirstElement(surfaceNode,"point1");
        const TiXmlElement* p2=
            xml->GetFirstElement(surfaceNode,"point2");

        xml->CheckAttributeNames(p1,"x z comment");
        xml->CheckAttributeNames(p2,"x z comment");

        SurfaceType.point1.x=xml->GetAttributeDouble(p1,"x");
        SurfaceType.point1.z=xml->GetAttributeDouble(p1,"z");
        SurfaceType.point2.x=xml->GetAttributeDouble(p2,"x");
        SurfaceType.point2.z=xml->GetAttributeDouble(p2,"z");
      }
      else{
        // New definition is enabled for normal-force mode.
        if(SurfaceType.NormalDirectionMode!=2){
          Run_Exceptioon(
              "point + slope requires NormalDirectionMode=2.");
        }

        const TiXmlElement* point=
            xml->GetFirstElement(surfaceNode,"point");
        const TiXmlElement* slope=
            xml->GetFirstElement(surfaceNode,"slope");

        xml->CheckAttributeNames(point,"x z comment");
        xml->CheckAttributeNames(slope,"value comment");

        const double x0=xml->GetAttributeDouble(point,"x");
        const double z0=xml->GetAttributeDouble(point,"z");
        const double k=xml->GetAttributeDouble(slope,"value");

        if(!std::isfinite(x0) ||
          !std::isfinite(z0) ||
          !std::isfinite(k)){
          Run_Exceptioon(
              "Surface point coordinates and slope must be finite.");
        }

        // Convert to the existing two-point representation.
        SurfaceType.point1.x=x0;
        SurfaceType.point1.z=z0;
        SurfaceType.point2.x=x0+1.0;
        SurfaceType.point2.z=z0+k;
      }

      // Common validation for both definitions.
      const double x1=SurfaceType.point1.x;
      const double z1=SurfaceType.point1.z;
      const double x2=SurfaceType.point2.x;
      const double z2=SurfaceType.point2.z;

      if(!std::isfinite(x1) || !std::isfinite(z1) ||
        !std::isfinite(x2) || !std::isfinite(z2)){
        Run_Exceptioon("Surface coordinates must be finite.");
      }

      const double dx=x2-x1;
      if(std::fabs(dx)<1e-12){
        Run_Exceptioon(
            "Surface points must have different x coordinates.");
      }

      const double k=(z2-z1)/dx;
      if(!std::isfinite(k)){
        Run_Exceptioon("Surface slope must be finite.");
      }
    }
  }

  // ------------------------------------------------------------
  // Normal stiffness.
  // ------------------------------------------------------------
  KnContactMode=
      parameters.GetValueInt("KnContactMode",true,1);

  if(KnContactMode<1 || KnContactMode>2){
    Run_Exceptioon("User.KnContactMode must be 1 or 2.");
  }

  if(KnContactMode==1){
    // Required in manual mode.
    KnContact=parameters.GetValueDouble("KnContact");

    if(KnContact<0.0){
      Run_Exceptioon(
          "User.KnContact must be non-negative.");
    }
  }

  KnContactLinearity=
      parameters.GetValueInt("KnContactLinearity",true,1);
  if(KnContactLinearity<1 || KnContactLinearity>2){
    Run_Exceptioon("User.KnContactLinearity must be 1 or 2.");
  }

  // ------------------------------------------------------------
  // Normal damping.
  // ------------------------------------------------------------
  unContactMode=
      parameters.GetValueInt("unContactMode",true,1);

  if(unContactMode<1 || unContactMode>2){
    Run_Exceptioon("User.unContactMode must be 1 or 2.");
  }

  if(unContactMode==1){
    // Required in manual mode. Zero damping is allowed.
    unContact=parameters.GetValueDouble("unContact");

    if(unContact<0.0){
      Run_Exceptioon(
          "User.unContact must be non-negative.");
    }
  }

  // ------------------------------------------------------------
  // Input/output data.
  // ------------------------------------------------------------
  const double saveinout=parameters.GetValueDouble("SaveInOutData",true,0);
  if(saveinout!=0 && saveinout!=1)
    Run_Exceptioon("User.SaveInOutData must be 0 or 1.");
  SaveInOutData=(saveinout==1);
    
  // ------------------------------------------------------------
  // Elastic-state output.
  // ------------------------------------------------------------
  const int saveflexdata=
      parameters.GetValueInt("SaveFlexData",true,0);

  if(saveflexdata!=0 && saveflexdata!=1){
    Run_Exceptioon("User.SaveFlexData must be 0 or 1.");
  }

  SaveFlexData=(saveflexdata==1);

  const int saveflexpairdata=
      parameters.GetValueInt("SaveFlexPairData",true,0);

  if(saveflexpairdata!=0 && saveflexpairdata!=1)
    Run_Exceptioon("User.SaveFlexPairData must be 0 or 1.");

  SaveFlexPairData=(saveflexpairdata==1);


  // ------------------------------------------------------------
  // Explicit elastic restart request.
  // ------------------------------------------------------------

  const TiXmlElement* restart=
      root->FirstChildElement("restart");

  if(restart){
    if(restart->NextSiblingElement("restart")){
      Run_ExceptioonFile(
          "Repeated User.restart node.",
          xml->ErrGetFileRow(restart));
    }

    xml->CheckAttributeNames(restart,"enabled comment");
    xml->CheckElementNames(
        restart,true,
        "restartsource conservation densityreference elasticinput "
        "boundarylayercorrection");

    FlexRestartRequested=
        xml->GetAttributeBool(
            restart,"enabled",true,false);

    const TiXmlElement* source=
        restart->FirstChildElement("restartsource");

    if(source){
      if(source->NextSiblingElement("restartsource")){
        Run_ExceptioonFile(
            "Repeated User.restart.restartsource node.",
            xml->ErrGetFileRow(source));
      }

      xml->CheckAttributeNames(source,"value comment");

      if(source->FirstChildElement()){
        Run_ExceptioonFile(
            "User.restart.restartsource does not accept child elements.",
            xml->ErrGetFileRow(source));
      }

      FlexRestartSourceRequested=
          xml->GetAttributeInt(source,"value");

      if(FlexRestartSourceRequested!=1 &&
         FlexRestartSourceRequested!=2)
      {
        Run_ExceptioonFile(
            "User.restart.restartsource must be "
            "1 (prescribed) or 2 (model_transfer).",
            xml->ErrGetFileRow(source));
      }
    }
    else if(FlexRestartRequested){
      Run_ExceptioonFile(
          "User.restart enabled=true requires an explicit "
          "restartsource: 1 (prescribed) or 2 (model_transfer).",
          xml->ErrGetFileRow(restart));
    }

    const auto readRestartBool=
        [&](const char* name,const char* attribute,bool defaultValue)->bool
    {
      const TiXmlElement* item=restart->FirstChildElement(name);

      if(!item)return defaultValue;

      if(item->NextSiblingElement(name)){
        Run_ExceptioonFile(
            std::string("Repeated User.restart.")+name+" node.",
            xml->ErrGetFileRow(item));
      }

      xml->CheckAttributeNames(
          item,std::string(attribute)+" comment");

      if(item->FirstChildElement()){
        Run_ExceptioonFile(
            std::string("User.restart.")+name+
            " does not accept child elements.",
            xml->ErrGetFileRow(item));
      }

      return xml->GetAttributeBool(item,attribute);
    };

    FlexRestartConserveMass=
        readRestartBool("conservation","mass",true);

    FlexRestartDensityIsNatural=
        readRestartBool("densityreference","natural",true);
    LoadRestartElasticInputXml(xml,restart);
  }

  const TiXmlElement* boundary=
      root->FirstChildElement("boundarylayercorrection");

  const TiXmlElement* legacyBoundary=restart
      ? restart->FirstChildElement("boundarylayercorrection")
      : nullptr;

  if(boundary && legacyBoundary)
    Run_ExceptioonFile(
        "Specify boundarylayercorrection only once, preferably under User.",
        xml->ErrGetFileRow(boundary));

  if(boundary)
    LoadRestartBoundaryCorrectionXml(xml,root);
  else if(legacyBoundary)
    LoadRestartBoundaryCorrectionXml(xml,restart);

  // ------------------------------------------------------------
  // Direct-continuation corrections.
  // Accept the new node or the legacy parameter keys, not both.
  // ------------------------------------------------------------

  const TiXmlElement* continuation=
      root->FirstChildElement("directcontinuation");

  const bool hasLegacyCorrections=
      nodes.find("FlexRestartRestorePairs")!=nodes.end() ||
      nodes.find("FlexRestartGradientCorrection")!=nodes.end() ||
      nodes.find("FlexRestartHourglassCorrection")!=nodes.end();

  if(continuation){
    if(continuation->NextSiblingElement("directcontinuation")){
      Run_ExceptioonFile(
          "Repeated User.directcontinuation node.",
          xml->ErrGetFileRow(continuation));
    }

    if(hasLegacyCorrections){
      Run_ExceptioonFile(
          "User.directcontinuation cannot be combined with "
          "legacy FlexRestart correction parameters.",
          xml->ErrGetFileRow(continuation));
    }

    xml->CheckAttributeNames(
        continuation,
        "restorepairs gradientcorrection hourglasscorrection comment");

    if(continuation->FirstChildElement()){
      Run_ExceptioonFile(
          "User.directcontinuation does not accept child elements.",
          xml->ErrGetFileRow(continuation));
    }

    FlexRestartRestorePairs=
        xml->GetAttributeBool(
            continuation,"restorepairs",true,false);

    FlexRestartGradientCorrection=
        xml->GetAttributeBool(
            continuation,"gradientcorrection",true,false);

    FlexRestartHourglassCorrection=
        xml->GetAttributeBool(
            continuation,"hourglasscorrection",true,false);
  }
  else{
    // Compatibility with existing XML files.
    const auto readbool=[&](const std::string& key) -> bool {
      const std::string value=
          parameters.GetValueStr(key,true,"false");

      if(value=="true" || value=="1")
        return true;

      if(value=="false" || value=="0")
        return false;

      Run_Exceptioon(
          std::string("User.")+key+
          " must be true/false or 1/0.");

      return false;
    };

    FlexRestartRestorePairs=
        readbool("FlexRestartRestorePairs");

    FlexRestartGradientCorrection=
        readbool("FlexRestartGradientCorrection");

    FlexRestartHourglassCorrection=
        readbool("FlexRestartHourglassCorrection");
  }

  // ------------------------------------------------------------
  // Output configuration.
  // ------------------------------------------------------------
  LoadOutputXml(xml,root);
  // Print only after all checks succeed.
  VisuConfig(log);

}

void UserParameters::LoadRestartElasticInputXml(
    const JXml* xml,const TiXmlElement* restart)
{
  const TiXmlElement* input=
      restart->FirstChildElement("elasticinput");

  if(!input){
    if(FlexRestartRequested){
      Run_ExceptioonFile(
          "User.restart enabled=true requires elasticinput.",
          xml->ErrGetFileRow(restart));
    }
    return;
  }

  if(input->NextSiblingElement("elasticinput")){
    Run_ExceptioonFile(
        "Repeated User.restart.elasticinput node.",
        xml->ErrGetFileRow(input));
  }

  xml->CheckAttributeNames(input,"source comment");

  FlexRestartElasticInput=
      xml->GetAttributeInt(input,"source");

  // Optional in every input mode; consumed only by hydrostatic initialisation.
  FlexRestartColumnGapCoef=0.1;

  const TiXmlElement* columnGap=
      input->FirstChildElement("columngapcoef");

  if(columnGap){
    if(columnGap->NextSiblingElement("columngapcoef")){
      Run_ExceptioonFile(
          "Repeated elasticinput.columngapcoef node.",
          xml->ErrGetFileRow(columnGap));
    }

    xml->CheckAttributeNames(columnGap,"value comment");

    if(columnGap->FirstChildElement()){
      Run_ExceptioonFile(
          "Columngapcoef does not accept child elements.",
          xml->ErrGetFileRow(columnGap));
    }

    FlexRestartColumnGapCoef=
        xml->GetAttributeDouble(columnGap,"value");

    if(!std::isfinite(FlexRestartColumnGapCoef) ||
       FlexRestartColumnGapCoef<0.0)
    {
      Run_ExceptioonFile(
          "Columngapcoef must be finite and non-negative.",
          xml->ErrGetFileRow(columnGap));
    }
  }

  if(FlexRestartElasticInput==1){
    xml->CheckElementNames(input,true,"file columngapcoef");

    const TiXmlElement* file=
        input->FirstChildElement("file");

    if(!file){
      Run_ExceptioonFile(
          "Elastic file input requires a file node.",
          xml->ErrGetFileRow(input));
      return;
    }

    if(file->NextSiblingElement("file")){
      Run_ExceptioonFile(
          "Repeated elastic input file node.",
          xml->ErrGetFileRow(file));
    }

    xml->CheckAttributeNames(
        file,"filename variabletype comment");

    if(file->FirstChildElement()){
      Run_ExceptioonFile(
          "Elastic input file does not accept child elements.",
          xml->ErrGetFileRow(file));
    }

    FlexRestartElasticFile=
        xml->GetAttributeStr(file,"filename");

    if(FlexRestartElasticFile.find_first_not_of(
        " \t\r\n")==std::string::npos)
    {
      Run_ExceptioonFile(
          "Elastic input filename must not be empty.",
          xml->ErrGetFileRow(file));
    }

    FlexRestartFileVariableType=
        xml->GetAttributeInt(file,"variabletype");

    if(FlexRestartFileVariableType!=1 &&
       FlexRestartFileVariableType!=2)
    {
      Run_ExceptioonFile(
          "Elastic file variabletype must be "
          "1 (deformation_gradient) or 2 (cauchy_stress).",
          xml->ErrGetFileRow(file));
    }
  }
  else if(FlexRestartElasticInput==2){
    xml->CheckElementNames(input,true,"specialfield columngapcoef");

    const TiXmlElement* field=
        input->FirstChildElement("specialfield");

    if(!field){
      Run_ExceptioonFile(
          "Special-field input requires a specialfield node.",
          xml->ErrGetFileRow(input));
      return;
    }

    if(field->NextSiblingElement("specialfield")){
      Run_ExceptioonFile(
          "Repeated specialfield node.",
          xml->ErrGetFileRow(field));
    }

    xml->CheckAttributeNames(field,"value sealevel comment");

    if(field->FirstChildElement()){
      Run_ExceptioonFile(
          "Specialfield does not accept child elements.",
          xml->ErrGetFileRow(field));
    }

    FlexRestartSpecialField=
        xml->GetAttributeInt(field,"value");

    if(FlexRestartSpecialField!=1 &&
       FlexRestartSpecialField!=2 &&
       FlexRestartSpecialField!=3)
    {
      Run_ExceptioonFile(
          "Specialfield must be "
          "1 (undeformed), 2 (hydrostatic_pressure), "
          "or 3 (sea_level_pressure).",
          xml->ErrGetFileRow(field));
    }

    if(FlexRestartSpecialField==3){
      FlexRestartSeaLevel=
          xml->GetAttributeDouble(field,"sealevel");

      if(!std::isfinite(FlexRestartSeaLevel))
        Run_ExceptioonFile(
            "Specialfield=3 requires a finite sealevel.",
            xml->ErrGetFileRow(field));

      if(FlexRestartRequested){
        const TiXmlNode* ctes=
            xml->GetNodeSimple("case.casedef.constantsdef");

        if(!ctes ||
           xml->ReadElementInt(ctes,"rhopgradient","value",true,2)!=3)
          Run_ExceptioonFile(
              "Specialfield=3 requires rhopgradient=3.",
              xml->ErrGetFileRow(field));
      }
    }
  }
  else{
    Run_ExceptioonFile(
        "Elasticinput source must be "
        "1 (file) or 2 (special_field).",
        xml->ErrGetFileRow(input));
  }
}

void UserParameters::LoadRestartBoundaryCorrectionXml(
    const JXml* xml,const TiXmlElement* restart)
{
  const TiXmlElement* node=
      restart->FirstChildElement("boundarylayercorrection");
  if(!node)return;

  if(node->NextSiblingElement("boundarylayercorrection"))
    Run_ExceptioonFile(
        "Repeated boundarylayercorrection node.",
        xml->ErrGetFileRow(node));

  xml->CheckAttributeNames(node,"mode comment");
  xml->CheckElementNames(
      node,true,
      "layers limit inwardreach quadraticerror boundaryform maxiterations");

  const auto readInteger=
      [&](const TiXmlElement* item,const char* attribute,
          int maximum)->int
  {
    const double value=xml->GetAttributeDouble(item,attribute);
    if(!std::isfinite(value) || value<0.0 ||
       value>double(maximum) || std::floor(value)!=value)
      Run_ExceptioonFile(
          std::string("Boundary-layer correction: ")+attribute+
          " must be an integer in [0, "+
          std::to_string(maximum)+"].",
          xml->ErrGetFileRow(item));
    return int(value);
  };

  const auto readChild=[&](const char* name)->const TiXmlElement*
  {
    const TiXmlElement* item=node->FirstChildElement(name);
    if(!item)return nullptr;

    if(item->NextSiblingElement(name))
      Run_ExceptioonFile(
          std::string("Repeated boundary-layer correction child: ")+name,
          xml->ErrGetFileRow(item));

    xml->CheckAttributeNames(item,"value comment");

    if(item->FirstChildElement())
      Run_ExceptioonFile(
          std::string("Boundary-layer correction child must be a leaf: ")+name,
          xml->ErrGetFileRow(item));

    return item;
  };

  FlexRestartBoundaryCorrectionMode=readInteger(node,"mode",5);

  const TiXmlElement* layers=readChild("layers");
  const TiXmlElement* limit=readChild("limit");
  const TiXmlElement* inward=readChild("inwardreach");
  const TiXmlElement* quadratic=readChild("quadraticerror");
  const TiXmlElement* maximum=readChild("maxiterations");
  const TiXmlElement* form=readChild("boundaryform");

  if(layers)
    FlexRestartBoundaryCorrectionLayers=
        readInteger(layers,"value",std::numeric_limits<int>::max());

  if(limit){
    const double value=xml->GetAttributeDouble(limit,"value");

    if(!std::isfinite(value) || value<=1.0 || value>10.0)
      Run_ExceptioonFile(
          "Boundary-layer correction limit must satisfy 1 < value <= 10.",
          xml->ErrGetFileRow(limit));

    FlexRestartBoundaryCorrectionLimit=value;
  }

  if(inward)
    FlexRestartBoundaryCorrectionInwardReach=
        xml->GetAttributeDouble(inward,"value");

  if(quadratic)
    FlexRestartBoundaryCorrectionQuadraticError=xml->GetAttributeDouble(quadratic,"value");
  if(maximum){
    const int value=readInteger(maximum,"value",std::numeric_limits<int>::max());
    if(value==0)Run_ExceptioonFile("Mode 5 maxiterations must be positive.",xml->ErrGetFileRow(maximum));
    FlexRestartBoundaryCorrectionMaxIterations=unsigned(value);
  }

  if(form)
    FlexRestartBoundaryCorrectionBoundaryForm=
        readInteger(form,"value",3);

  FlexBoundaryCorrection::Options options;
  options.Layers=FlexRestartBoundaryCorrectionLayers;
  options.InwardReach=FlexRestartBoundaryCorrectionInwardReach;
  options.MaxIterations=FlexRestartBoundaryCorrectionMaxIterations;
  options.QuadraticError=FlexRestartBoundaryCorrectionQuadraticError;
  options.BoundaryForm=FlexRestartBoundaryCorrectionBoundaryForm;

  const std::string error=
      FlexBoundaryCorrection::ValidateOptions(options);

  if(!error.empty())
    Run_ExceptioonFile(error,xml->ErrGetFileRow(node));

  if((FlexRestartBoundaryCorrectionMode==1 ||
      FlexRestartBoundaryCorrectionMode==3 ||
      FlexRestartBoundaryCorrectionMode==4 ||
      FlexRestartBoundaryCorrectionMode==5) && !layers)
    Run_ExceptioonFile(
        "Boundary-layer correction modes 1/3/4/5 require layers; "
        "use value=0 explicitly for all particles.",
        xml->ErrGetFileRow(node));

  if(FlexRestartBoundaryCorrectionMode==2 && !limit)
    Run_ExceptioonFile(
        "Boundary-layer correction mode 2 requires limit.",
        xml->ErrGetFileRow(node));
}

void UserParameters::LoadOutputXml(
    const JXml* xml,const TiXmlElement* userNode){
  OutputObjects.clear();

  const TiXmlElement* output=
      userNode->FirstChildElement("output");

  if(!output)return;

  // Allow multiple output nodes
  xml->CheckAttributeNames(output,"comment");
  xml->CheckElementNames(output,true,"*object");

  // Output failure information altogether with the XML file and line number.
  const auto Fail=
      [&](const TiXmlElement* ele,const std::string& message){
    Run_ExceptioonFile(message,xml->ErrGetFileRow(ele));
  };
  const auto Trim=[](const std::string& text)->std::string
  {
    const std::string spaces=" \t\r\n";

    const size_t first=text.find_first_not_of(spaces);
    if(first==std::string::npos)return "";

    const size_t last=text.find_last_not_of(spaces);
    return text.substr(first,last-first+1);
  };

  const auto ReadList=
      [&](const TiXmlElement* ele,
          const std::string& attribute,
          const std::vector<std::string>& allowed,
          bool optional)->std::vector<std::string>
  {
    std::vector<std::string> values;

    if(optional && !ele->Attribute(attribute.c_str())){
      return values;
    }

    const std::string text=
        xml->GetAttributeStr(ele,attribute);

    size_t begin=0;

    while(true){
      const size_t end=text.find(',',begin);

      const std::string value=Trim(
          text.substr(begin,
              end==std::string::npos
                  ? std::string::npos : end-begin));

      if(value.empty()){
        Fail(ele,"Empty item in output attribute: "+attribute);
      }

      if(std::find(values.begin(),values.end(),value)
          !=values.end()){
        Fail(ele,"Repeated item in "+attribute+": "+value);
      }

      if(!allowed.empty()
          && std::find(allowed.begin(),allowed.end(),value)
              ==allowed.end()){
        Fail(ele,"Unsupported item in "+attribute+": "+value);
      }

      values.push_back(value);

      if(end==std::string::npos)break;
      begin=end+1;
    }

    return values;
  };

  // Check that the element has no child elements and only the expected attributes.
  const auto CheckLeaf=
      [&](const TiXmlElement* ele,
          const std::string& attributes)
  {
    xml->CheckAttributeNames(ele,attributes);

    if(ele->FirstChildElement()){
      Fail(ele,"Output variable must not contain child elements.");
    }
  };

  // Only accept a specific value for an attribute (source and measure), otherwise fail.
  const auto RequireValue=
      [&](const TiXmlElement* ele,
          const std::string& attribute,
          const std::string& expected)
  {
    const std::string value=
        Trim(xml->GetAttributeStr(ele,attribute));

    if(value!=expected){
      Fail(ele,"Unsupported "+attribute+"='"+value+
          "'. Currently supported: "+expected);
    }
  };

  const std::vector<std::string> tensorNames={
      "xx","yy","zz","xy","xz","yz"};

  const std::vector<std::string> invariantNames={
      "mean","mech_pressure","von_mises"};

  std::set<std::string> usedNames;
  std::vector<StOutputObject> objects;

  //---------------------
  // Start to input xml
  //---------------------
  for(const TiXmlElement* obj=output->FirstChildElement("object");
      obj;
      obj=obj->NextSiblingElement("object"))
  {
    xml->CheckAttributeNames(obj,"name mk formats outputfolder comment");
    xml->CheckElementNames(obj,true,"dynamic static");

    StOutputObject config;

    // --------------------------------------------------------
    // Output file name
    // --------------------------------------------------------
    config.Name=Trim(xml->GetAttributeStr(obj,"name"));

    const std::string validNameChars=
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_";

    if(config.Name.empty()
        || config.Name.find_first_not_of(validNameChars)
            !=std::string::npos){
      Fail(obj,
          "Output object name must contain only letters, "
          "digits and underscores.");
    }

    if(!usedNames.insert(config.Name).second){
      Fail(obj,"Repeated output object name: "+config.Name);
    }

    // --------------------------------------------------------
    // MK list
    // --------------------------------------------------------
    const std::vector<std::string> mkText=
        ReadList(obj,"mk",{},false);

    std::set<unsigned> usedMk;

    for(const std::string& text:mkText){
      // Output-only selector; XML MK values remain in 0..65535.
      if(text=="inlet"){
        config.Mk.push_back(65536u);
        continue;
      }
      unsigned value=0;

      for(char c:text){
        if(c<'0' || c>'9'){
          Fail(obj,"MK must be a non-negative integer: "+text);
        }
        value=value*10u+unsigned(c-'0');

        if(value>65535u){
          Fail(obj,"MK is outside the range 0..65535: "+text);
        }
      }
      if(!usedMk.insert(value).second){
        Fail(obj,"Repeated MK value: "+text);
      }
      config.Mk.push_back(value);
    }

    // --------------------------------------------------------
    // output file formats
    // --------------------------------------------------------
    const std::vector<std::string> formats=
        ReadList(obj,"formats",{"csv","vtk"},false);

    for(const std::string& format:formats){
      if(format=="csv")config.Csv=true;
      else if(format=="vtk")config.Vtk=true;
    }

    // --------------------------------------------------------
    // output folder
    // --------------------------------------------------------
    config.OutputFolder=
    xml->GetAttributeStrSimple(obj,"outputfolder",true,"");
    // Use Linux path separators.
    if(config.OutputFolder.find('\\')!=std::string::npos){
      Fail(obj,"outputfolder must use '/' as the path separator.");
    }
    // A $ path must stay relative to DirOut.
    if(!config.OutputFolder.empty()
        && config.OutputFolder[0]=='$'){
      const std::string subdir=config.OutputFolder.substr(1);

      if(subdir.empty() || subdir[0]=='/'){
        Fail(obj,
            "A $ outputfolder requires a relative subdirectory.");
      }

      size_t begin=0;

      while(true){
        const size_t end=subdir.find('/',begin);

        const std::string part=subdir.substr(
            begin,
            end==std::string::npos
                ? std::string::npos : end-begin);

        if(part==".."){
          Fail(obj,
              "A $ outputfolder must not contain '..' components.");
        }

        if(end==std::string::npos)break;
        begin=end+1;
      }
    }

    // --------------------------------------------------------
    // Time-varying output variables.
    // --------------------------------------------------------
    const TiXmlElement* dynamic=
        obj->FirstChildElement("dynamic");

    if(dynamic){
      xml->CheckAttributeNames(dynamic,"comment");
      xml->CheckElementNames(
          dynamic,true,
          "density pressure volume velocity strain stress");

      const TiXmlElement* ele=nullptr;

      ele=dynamic->FirstChildElement("density");
      if(ele){
        CheckLeaf(ele,"source comment");

        const auto sources=
          ReadList(ele,"source",{"material","sph"},false);

        for(const std::string& source:sources){
          if(source=="material"){
            config.Dynamic.DensityMaterial=true;
          }
          else if(source=="sph"){
            config.Dynamic.DensitySPH=true;
          }
        }
      }

      ele=dynamic->FirstChildElement("pressure");
      if(ele){
        CheckLeaf(ele,"source comment");

        const auto sources=
          ReadList(ele,"source",{"sph"},false);

        for(const std::string& source:sources){
          if(source=="sph"){
            config.Dynamic.PressureSPH=true;
          }
        }
      }

      ele=dynamic->FirstChildElement("volume");
      if(ele){
        CheckLeaf(ele,"source comment");

        const auto sources=
          ReadList(ele,"source",{"material","sph"},false);

        for(const std::string& source:sources){
          if(source=="material"){
            config.Dynamic.VolumeMaterial=true;
          }
          else if(source=="sph"){
            config.Dynamic.VolumeSPH=true;
          }
        }
      }

      ele=dynamic->FirstChildElement("velocity");
      if(ele){
        CheckLeaf(ele,"comment");
        config.Dynamic.Velocity=true;
      }

      ele=dynamic->FirstChildElement("strain");
      if(ele){
        CheckLeaf(ele,"measure components comment");
        RequireValue(ele,"measure","green_lagrange");

        config.Dynamic.StrainComponents=
            ReadList(ele,"components",tensorNames,false);
      }

      ele=dynamic->FirstChildElement("stress");
      if(ele){
        CheckLeaf(ele,"measure components invariants comment");
        RequireValue(ele,"measure","cauchy");

        config.Dynamic.StressComponents=
            ReadList(ele,"components",tensorNames,true);

        config.Dynamic.StressInvariants=
            ReadList(ele,"invariants",invariantNames,true);

        if(config.Dynamic.StressComponents.empty()
            && config.Dynamic.StressInvariants.empty()){
          Fail(ele,
              "Stress requires components and/or invariants.");
        }
      }
    }

    // --------------------------------------------------------
    // Time-invariant output variables.
    // --------------------------------------------------------
    const TiXmlElement* statics=
        obj->FirstChildElement("static");

    if(statics){
      xml->CheckAttributeNames(statics,"comment");
      xml->CheckElementNames(statics,true,"mass boundary");

      const TiXmlElement* mass=
          statics->FirstChildElement("mass");

      if(mass){
        CheckLeaf(mass,"comment");
        config.Static.Mass=true;
      }

      const TiXmlElement* boundary=
          statics->FirstChildElement("boundary");

      if(boundary){
        CheckLeaf(boundary,"comment");
        config.Static.Boundary=true;
      }
    }

    // Every written output object must have at least one variable selected.
    const StOutputDynamic& d=config.Dynamic;

    const bool hasVariable=
      d.DensityMaterial
      || d.DensitySPH
      || d.PressureSPH
      || d.VolumeMaterial
      || d.VolumeSPH
      || d.Velocity
      || !d.StrainComponents.empty()
      || !d.StressComponents.empty()
      || !d.StressInvariants.empty()
      || config.Static.Mass
      || config.Static.Boundary;

    if(!hasVariable){
      Fail(obj,"Output object has no selected variables: "
          +config.Name);
    }

    objects.push_back(config);
  }

  // Submit after check every objects
  OutputObjects.swap(objects);
}

void UserParameters::VisuConfig(JLog2* log)const{
  if(!log)return;

  log->Printf("User configuration:");
  log->Printf("  FlexBoundContactMode: %d",
              FlexBoundContactMode);

  if(FlexBoundContactMode==2){
    log->Printf("  SurfaceType: %d",SurfaceType.value);

    if(SurfaceType.value==1){
      log->Printf("  WallZ: %.17g",SurfaceType.WallZ);
    }
    else if(SurfaceType.value==2){
      log->Printf("  point1: x=%.17g, z=%.17g",
                  SurfaceType.point1.x,
                  SurfaceType.point1.z);

      log->Printf("  point2: x=%.17g, z=%.17g",
                  SurfaceType.point2.x,
                  SurfaceType.point2.z);
      log->Printf("  NormalDirectionMode: %d",
                SurfaceType.NormalDirectionMode);
    }
  }

  log->Printf("  KnContactMode: %d",KnContactMode);

  if(KnContactMode==1){
    log->Printf("  KnContact: %.17g",KnContact);
  }
  log->Printf("  KnContactLinearity: %d",KnContactLinearity);

  log->Printf("  unContactMode: %d",unContactMode);

  if(unContactMode==1){
    log->Printf("  unContact: %.17g",unContact);
  }

  log->Printf("  SaveFlexData: %s",
            SaveFlexData ? "true" : "false");
  log->Printf("  SaveFlexPairData: %s",
            SaveFlexPairData ? "true" : "false");
  log->Printf("  FlexRestart.Requested: %s",
      FlexRestartRequested ? "true" : "false");
  const char* requestedSource=
      FlexRestartSourceRequested==1 ? "prescribed" :
      FlexRestartSourceRequested==2 ? "model_transfer" :
      "unspecified";

  log->Printf(
      "  FlexRestart.SourceRequested: %d (%s; %s)",
      FlexRestartSourceRequested,
      requestedSource,
      FlexRestartRequested ? "enabled" : "inactive");

  log->Printf(
      "  FlexRestart.Conservation: %s (%s).",
      FlexRestartConserveMass ? "mass" : "reference_volume",
      FlexRestartRequested
          ? "requested; irrelevant for undeformed field"
          : "inactive");

  log->Printf(
      "  FlexRestart.DensityReference: %s (%s).",
      FlexRestartDensityIsNatural ? "natural" : "reference",
      FlexRestartRequested
          ? "requested; irrelevant for undeformed field"
          : "inactive");

  const int boundaryMode=FlexRestartBoundaryCorrectionMode;

  log->Printf(
      "  BoundaryLayerCorrection: mode=%d; %s",
      boundaryMode,
      boundaryMode==0 ? "disabled" :
      boundaryMode==5
          ? "configured; regularized geometry and slow mixed projection"
          : boundaryMode==4
              ? "enabled; fixed geometric energy weights"
              : "enabled; force assembly only");

  if(boundaryMode>=1 && boundaryMode<=5)
    log->Printf(
        "    Layers: %d (0=all; geometric shells per elastic body)",
        FlexRestartBoundaryCorrectionLayers);

  if(boundaryMode==2)
    log->Printf(
        "    Limit: %.17g (1 < value <= 10)",
        FlexRestartBoundaryCorrectionLimit);

  const char* boundaryForm=
      FlexRestartBoundaryCorrectionBoundaryForm==1 ? "fluid" :
      FlexRestartBoundaryCorrectionBoundaryForm==2 ? "solid" :
      "mixed";

  const char* mode5Status=
      boundaryMode!=5 ? "inactive: mode 5 only" :
      "mode 5 only";

  log->Printf(
      "    InwardReach: %.17g h (0=disabled; %s)",
      FlexRestartBoundaryCorrectionInwardReach,mode5Status);

  log->Printf(
      "    MaxIterations: %u per component (%s)",
      FlexRestartBoundaryCorrectionMaxIterations,mode5Status);

  log->Printf("    QuadraticError: %.17g (0=disabled; initialization only; %s)",
      FlexRestartBoundaryCorrectionQuadraticError,mode5Status);

  log->Printf(
      "    BoundaryForm: %d (%s; %s)",
      FlexRestartBoundaryCorrectionBoundaryForm,
      boundaryForm,mode5Status);
        
  const char* elasticInput=
      FlexRestartElasticInput==1 ? "file" :
      FlexRestartElasticInput==2 ? "special_field" :
      "unspecified";

  log->Printf(
      "  FlexRestart.ElasticInput: %d (%s; %s)",
      FlexRestartElasticInput,
      elasticInput,
      FlexRestartRequested ? "enabled" : "inactive");

  if(FlexRestartElasticInput==1){
    const char* variableType=
        FlexRestartFileVariableType==1
        ? "deformation_gradient"
        : "cauchy_stress";

    log->Printf(
        "  FlexRestart.ElasticFile: %s",
        FlexRestartElasticFile.c_str());

    log->Printf(
        "  FlexRestart.FileVariableType: %d (%s)",
        FlexRestartFileVariableType,
        variableType);
  }
  else if(FlexRestartElasticInput==2){
    const char* specialField=
        FlexRestartSpecialField==1
        ? "undeformed"
        : (FlexRestartSpecialField==2
           ? "hydrostatic_pressure"
           : "sea_level_pressure");

    log->Printf(
        "  FlexRestart.SpecialField: %d (%s)",
        FlexRestartSpecialField,
        specialField);
    if(FlexRestartSpecialField==3)
      log->Printf("  FlexRestart.SeaLevel: %g m",FlexRestartSeaLevel);
  }
}

void UserParameters::ConfigFlexRestartCorrections(
    bool isDirectContinuation,
    bool hasflex,
    JLog2* log)
{
  const bool applicable=isDirectContinuation && hasflex;
  if(log){
    log->Printf(
        "Elastic high-frequency dissipation: %s; KV dissipation: %s.",
        ElasticHighFrequencyDissipation && hasflex
          ? "enabled" : "disabled",
        ElasticViscousDissipation && hasflex
          ? "enabled" : "disabled");

    if(hasflex && ElasticViscousDissipation)
      log->Printf("  KV factor: %g",ElasticViscousFactor);
  }
  
  UseFlexRestartRestorePairs=
      applicable && FlexRestartRestorePairs;

  UseFlexRestartGradientCorrection=
      applicable && FlexRestartGradientCorrection;

  UseFlexRestartHourglassCorrection=
      applicable && FlexRestartHourglassCorrection;

  if(hasflex && ElasticViscousDissipation &&
      (UseFlexRestartRestorePairs || UseFlexRestartGradientCorrection))
    Run_Exceptioon(
        "KV requires directcontinuation.restorepairs=false and "
        "directcontinuation.gradientcorrection=false.");

  const auto reportUnused=
      [&](const char* option,bool requested)
  {
    if(!requested || applicable || !log)
      return;

    log->Printf(
        "User.directcontinuation.%s: unused (%s%s%s).",
        option,
        !isDirectContinuation ? "not direct continuation" : "",
        !isDirectContinuation && !hasflex ? "; " : "",
        !hasflex ? "no flexible structures" : "");
  };

  reportUnused(
      "restorepairs",
      FlexRestartRestorePairs);

  reportUnused(
      "gradientcorrection",
      FlexRestartGradientCorrection);

  reportUnused(
      "hourglasscorrection",
      FlexRestartHourglassCorrection);

  if(FlexRestartBoundaryCorrectionMode!=0 &&
     UseFlexRestartGradientCorrection)
    Run_Exceptioon(
        "Boundary correction requires directcontinuation.gradientcorrection=false.");

  if(FlexRestartBoundaryCorrectionMode==5 &&
     UseFlexRestartHourglassCorrection)
    Run_Exceptioon(
        "Mode 5 requires directcontinuation.hourglasscorrection=false; "
        "its own matched hourglass is used.");

  // Check dependencies only when direct-continuation corrections apply.
  if(UseFlexRestartGradientCorrection &&
     !UseFlexRestartRestorePairs)
  {
    Run_Exceptioon(
        "Direct-continuation gradient correction requires "
        "pair restoration: set restorepairs=true "
        "(legacy parameter: FlexRestartRestorePairs).");
  }
}

const tmatrix3f& UserParameters::GetFlexRestartFstar(
    unsigned pfs,unsigned idp)const
{
  if(pfs>=FlexRestartState.Fstar.size() ||
     pfs>=FlexRestartState.Idp.size())
    Run_Exceptioon("FlexRestart: restart state is unavailable.");

  if(FlexRestartState.Idp[pfs]!=idp)
    Run_Exceptioon("FlexRestart: particle ID mismatch.");

  return FlexRestartState.Fstar[pfs];
}

void UserParameters::ConfigFlexRestartMode(
    bool particleContinuation,
    bool hasflex,
    JLog2* log)
{
  FlexRestartSource=TpFlexRestartSource::None;

  if(FlexRestartRequested && particleContinuation){
    Run_Exceptioon(
        "User restart and SPH restart are mutually exclusive. "
        "Disable User.restart for direct continuation, "
        "or remove -partbegin for User restart.");
  }

  if(FlexRestartRequested && !hasflex){
    Run_Exceptioon(
        "User.restart requires flexible structures.");
  }

  if(hasflex){
    if(particleContinuation){
      FlexRestartSource=
          TpFlexRestartSource::DirectContinuation;
    }
    else if(FlexRestartRequested){
      switch(FlexRestartSourceRequested){
        case 1:
          FlexRestartSource=
              TpFlexRestartSource::Prescribed;
          break;

        case 2:
          FlexRestartSource=
              TpFlexRestartSource::ModelTransfer;
          break;

        default:
          Run_Exceptioon(
              "User.restart requires restartsource "
              "1 (prescribed) or 2 (model_transfer).");
          break;
      }
    }
  }

  if((IsFlexPrescribedRestart() || IsFlexModelTransferRestart()) &&
     SaveFlexPairData && !IsFlexUndeformedRestart())
    Run_Exceptioon(
        "SaveFlexPairData requires the prescribed undeformed field "
        "when User.restart is enabled (source=2, specialfield=1).");

  ConfigFlexRestartCorrections(
      IsFlexDirectContinuation(),
      hasflex,
      log);

  if(log){
    const char* source=
        IsFlexDirectContinuation() ? "direct_continuation" :
        IsFlexPrescribedRestart() ? "prescribed" :
        IsFlexModelTransferRestart() ? "model_transfer" :
        "none";

    log->Printf(
        "FlexRestart: active=%s; source=%s.",
        IsFlexRestart() ? "true" : "false",
        source);
  }
}