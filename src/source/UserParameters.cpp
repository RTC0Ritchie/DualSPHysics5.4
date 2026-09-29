#include "UserParameters.h"
#include "JXml.h"
#include "JLog2.h"
#include "JCaseEParms.h"

#include <map>
#include <string>

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
  xml->CheckElementNames(root,true,"*parameter");

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
  parameters.LoadXml(xml,"case.User");

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
    else{ // Two-point geometry.
      xml->CheckElementNames(surfaceNode,true,"point1 point2");
      const TiXmlElement* p1=
          xml->GetFirstElement(surfaceNode,"point1");
      const TiXmlElement* p2=
          xml->GetFirstElement(surfaceNode,"point2");
      xml->CheckAttributeNames(p1,"x z comment");
      xml->CheckAttributeNames(p2,"x z comment");
      // All four coordinates are required.
      SurfaceType.point1.x=xml->GetAttributeDouble(p1,"x");
      SurfaceType.point1.z=xml->GetAttributeDouble(p1,"z");
      SurfaceType.point2.x=xml->GetAttributeDouble(p2,"x");
      SurfaceType.point2.z=xml->GetAttributeDouble(p2,"z");
      if(SurfaceType.point1.x==SurfaceType.point2.x &&
         SurfaceType.point1.z==SurfaceType.point2.z)
      {
        Run_Exceptioon(
            "User.SurfaceType.point1 and point2 must be different.");
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

  // Print only after all checks succeed.
  VisuConfig(log);

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
    else{
      log->Printf("  point1: x=%.17g, z=%.17g",
                  SurfaceType.point1.x,
                  SurfaceType.point1.z);

      log->Printf("  point2: x=%.17g, z=%.17g",
                  SurfaceType.point2.x,
                  SurfaceType.point2.z);
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
}