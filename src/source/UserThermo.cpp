#include "UserThermo.h"
#include "JXml.h"
#include "JBinaryData.h"
#include "Functions.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

UserThermo::UserThermo(){
  ClassName="UserThermo";
}

void UserThermo::LoadXml(
    const JXml* xml,const TiXmlElement* node){
  Enabled=false;
  PrescribeMode=0;
  InitialTemperature=0.0;
  if(!node)return;

  if(node->NextSiblingElement("thermo"))
    Run_ExceptioonFile(
      "Repeated User.thermo node.",
      xml->ErrGetFileRow(node));

  xml->CheckAttributeNames(node,"enabled comment");
  xml->CheckElementNames(
      node,true,"prescribe initialtemperature");

  const bool enabled=
      xml->GetAttributeBool(node,"enabled",true,false);

  const TiXmlElement* prescribe=
      node->FirstChildElement("prescribe");
  const TiXmlElement* initial=
      node->FirstChildElement("initialtemperature");

  if(enabled && (!prescribe || !initial))
    Run_ExceptioonFile(
      "Enabled thermo requires prescribe and initialtemperature.",
      xml->ErrGetFileRow(node));

  if(prescribe){
    xml->CheckAttributeNames(prescribe,"mode comment");

    if(prescribe->NextSiblingElement("prescribe")
        || prescribe->FirstChildElement())
      Run_ExceptioonFile(
        "prescribe must be a unique leaf.",
        xml->ErrGetFileRow(prescribe));

    if(xml->GetAttributeInt(prescribe,"mode")!=0)
      Run_ExceptioonFile(
        "Only thermo prescribe mode=0 is supported.",
        xml->ErrGetFileRow(prescribe));
  }

  if(initial){
    xml->CheckAttributeNames(
        initial,"value units_comment comment");

    if(initial->NextSiblingElement("initialtemperature")
        || initial->FirstChildElement())
      Run_ExceptioonFile(
        "initialtemperature must be a unique leaf.",
        xml->ErrGetFileRow(initial));

    const double value=
        xml->GetAttributeDouble(initial,"value");

    // Keep the finite-value check valid with -ffast-math.
    static_assert(
      sizeof(double)==sizeof(std::uint64_t)
      && std::numeric_limits<double>::is_iec559,
      "IEEE binary64 required.");

    std::uint64_t bits=0;
    std::memcpy(&bits,&value,sizeof(bits));

    if((bits & UINT64_C(0x7ff0000000000000))
          ==UINT64_C(0x7ff0000000000000)
        || value<0.0)
      Run_ExceptioonFile(
        "Initial temperature must be finite and >=0 K.",
        xml->ErrGetFileRow(initial));

    InitialTemperature=value;
  }

  Enabled=enabled;
}

double UserThermo::GetTemperature(unsigned idp)const{
  if(!Enabled)
    Run_Exceptioon("Thermo is disabled.");

  // Mode 0 assigns the same value to every ID,
  // including newly born particle IDs.
  (void)idp;
  return InitialTemperature;
}

void UserThermo::SaveSnapshot(
    const std::string& directory,const std::string& runCode,
    unsigned part,unsigned step,double time,
    unsigned np,const unsigned* idp,const typecode* code)const{
  if(!Enabled)return;

  if(np && (!idp || !code))
    Run_Exceptioon("Missing thermal particle identities.");

  std::vector<unsigned> ids;
  ids.reserve(np);

  for(unsigned p=0;p<np;p++){
    if(CODE_IsNormal(code[p]))
      ids.push_back(idp[p]);
  }

  std::sort(ids.begin(),ids.end());

  if(std::adjacent_find(ids.begin(),ids.end())!=ids.end())
    Run_Exceptioon(
      "Duplicated normal particle ID in thermal output.");

  const unsigned n=unsigned(ids.size());

  JBinaryData data("JPartThermoBi4");
  data.SetvUint("FormatVer",1);
  data.SetvText("WriterRunCode",runCode);
  data.SetvUint("Cpart",part);
  data.SetvUint("Nstep",step);
  data.SetvDouble("TimeStep",time);
  data.SetvUint("Np",n);
  data.SetvUint("PrescribeMode",PrescribeMode);
  data.SetvDouble("InitialTemperature",InitialTemperature);
  data.SetvText("TemperatureUnits","K");
  data.SetvText("RestartPolicy","reset_from_xml");

  unsigned* savedIds=
      data.CreateArrayUint("Idp",n,false);
  double* temperature=
      data.CreateArrayDouble("Temperature",n,false);

  for(unsigned p=0;p<n;p++){
    savedIds[p]=ids[p];
    temperature[p]=GetTemperature(ids[p]);
  }

  const std::string file=
      fun::GetDirWithSlash(directory)
      +fun::PrintStr("part_Thermo%04u.bi4",part);

  data.SaveFile(file,false,true);
}