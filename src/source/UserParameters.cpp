#include "UserParameters.h"
#include "JXml.h"
#include "JLog2.h"
#include "JCaseEParms.h"
#include "Functions.h"

#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <cmath>

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

  SaveFlexData=false;

  OutputObjects.clear();


  FlexRestartHistory.Clear();
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
  xml->CheckElementNames(root,true,"*parameter output");

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
  // Elastic-state output.
  // ------------------------------------------------------------
  const int saveflexdata=
      parameters.GetValueInt("SaveFlexData",true,0);

  if(saveflexdata!=0 && saveflexdata!=1){
    Run_Exceptioon("User.SaveFlexData must be 0 or 1.");
  }

  SaveFlexData=(saveflexdata==1);

  LoadOutputXml(xml,root);
  // Print only after all checks succeed.
  VisuConfig(log);

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
      xml->CheckElementNames(statics,true,"mass");

      const TiXmlElement* mass=
          statics->FirstChildElement("mass");

      if(mass){
        CheckLeaf(mass,"comment");
        config.Static.Mass=true;
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
      || config.Static.Mass;

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
}

const tmatrix3f& UserParameters::GetFlexRestartDefGrad(
    unsigned pfs,unsigned idp)const
{
  if(pfs>=FlexRestartHistory.DefGrad.size() ||
     pfs>=FlexRestartHistory.Idp.size())
    Run_Exceptioon("Flexible restart history is unavailable.");

  if(FlexRestartHistory.Idp[pfs]!=idp)
    Run_Exceptioon("Flexible restart history particle ID mismatch.");

  return FlexRestartHistory.DefGrad[pfs];
}