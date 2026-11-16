#include "UserThermo.h"
#include "JXml.h"
#include "JBinaryData.h"
#include "Functions.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

bool Finite(double value){
  static_assert(sizeof(double)==sizeof(std::uint64_t)
    && std::numeric_limits<double>::is_iec559,"IEEE binary64 required.");

  std::uint64_t bits=0;
  std::memcpy(&bits,&value,sizeof(bits));

  // Keep this check valid with -ffast-math.
  return (bits & UINT64_C(0x7ff0000000000000))
    !=UINT64_C(0x7ff0000000000000);
}

// Rectangular X-Z grid: rows ordered by Z, then by X, both ascending.
class TemperatureGrid {
  std::vector<double> X,Z,T;

public:
  explicit TemperatureGrid(const std::string& file){
    std::ifstream in(file.c_str());
    in.imbue(std::locale::classic());

    if(!in)
      throw std::runtime_error("Cannot open temperature grid: "+file);

    std::string line;
    std::getline(in,line);
    if(!line.empty() && line.back()=='\r')line.pop_back();

    if(line!="x_m,z_m,temperature_K")
      throw std::runtime_error(
        "Expected x_m,z_m,temperature_K header: "+file);

    size_t column=0;
    bool firstRow=true;

    while(std::getline(in,line)){
      if(line.find_first_not_of(" \t\r")==std::string::npos)continue;

      if(std::count(line.begin(),line.end(),',')!=2)
        throw std::runtime_error("Expected three temperature CSV columns.");

      std::replace(line.begin(),line.end(),',',' ');
      std::istringstream row(line);
      row.imbue(std::locale::classic());

      double x,z,t;
      if(!(row>>x>>z>>t)
          || !Finite(x) || !Finite(z) || !Finite(t) || t<0)
        throw std::runtime_error(
          "Invalid coordinate or Kelvin temperature.");

      row>>std::ws;
      if(!row.eof())
        throw std::runtime_error("Extra data in temperature CSV.");

      if(Z.empty())Z.push_back(z);

      if(z!=Z.back()){
        if(z<Z.back() || column!=X.size())
          throw std::runtime_error(
            "Incomplete or unordered temperature grid.");

        firstRow=false;
        Z.push_back(z);
        column=0;
      }

      if(firstRow){
        if(!X.empty() && x<=X.back())
          throw std::runtime_error(
            "Temperature X coordinates must increase.");

        X.push_back(x);
      }
      else if(column>=X.size() || x!=X[column])
        throw std::runtime_error(
          "Temperature grid must be rectangular.");

      T.push_back(t);
      column++;
    }

    if(in.bad() || X.size()<2 || Z.size()<2 || column!=X.size()
        || T.size()!=X.size()*Z.size())
      throw std::runtime_error(
        "Incomplete temperature grid (minimum 2x2).");
  }

  bool Sample(double x,double z,double& t)const{
    if(!Finite(x) || !Finite(z))
      throw std::runtime_error("Non-finite particle coordinate.");

    // Outside the grid: retain initial/restart temperature.
    if(x<X.front() || x>X.back() || z<Z.front() || z>Z.back())
      return false;

    const size_t ix=std::min(
      size_t(std::upper_bound(X.begin(),X.end(),x)-X.begin()-1),
      X.size()-2);

    const size_t iz=std::min(
      size_t(std::upper_bound(Z.begin(),Z.end(),z)-Z.begin()-1),
      Z.size()-2);

    const double a=(x-X[ix])/(X[ix+1]-X[ix]);
    const double b=(z-Z[iz])/(Z[iz+1]-Z[iz]);
    const size_t p=iz*X.size()+ix;

    t=(1-b)*((1-a)*T[p]+a*T[p+1])
      +b*((1-a)*T[p+X.size()]+a*T[p+X.size()+1]);

    if(!Finite(t) || t<0)
      throw std::runtime_error("Invalid interpolated temperature.");

    return true;
  }
};

}

UserThermo::UserThermo(){
  ClassName="UserThermo";
}

void UserThermo::LoadXml(
    const JXml* xml,const TiXmlElement* node){
  Enabled=false;
  Ready=false;
  PrescribeMode=0;
  InitialTemperature=0;
  InputFile.clear();
  FileMk=0;
  FileRenew=false;
  UpdatedCount=0;
  Temperature.clear();

  if(!node)return;

  if(node->NextSiblingElement("thermo"))
    Run_ExceptioonFile(
      "Repeated thermo node.",xml->ErrGetFileRow(node));

  xml->CheckAttributeNames(node,"enabled comment");
  xml->CheckElementNames(
    node,true,"prescribe initialtemperature restartRenew");

  const bool enabled=
    xml->GetAttributeBool(node,"enabled",true,false);

  if(!enabled)return;

  const TiXmlElement* prescribe=
    node->FirstChildElement("prescribe");
  const TiXmlElement* initial=
    node->FirstChildElement("initialtemperature");

  if(!prescribe || !initial)
    Run_Exceptioon(
      "Enabled thermo requires prescribe and initialtemperature.");

  xml->CheckAttributeNames(prescribe,"mode comment");
  xml->CheckElementNames(prescribe,true,"file");

  if(prescribe->NextSiblingElement("prescribe"))
    Run_Exceptioon("Repeated prescribe node.");

  const int mode=xml->GetAttributeInt(prescribe,"mode");

  if(mode!=0 && mode!=1)
    Run_Exceptioon("Thermo prescribe mode must be 0 or 1.");

  PrescribeMode=unsigned(mode);

  xml->CheckAttributeNames(initial,"value units_comment comment");

  if(initial->NextSiblingElement("initialtemperature")
      || initial->FirstChildElement())
    Run_Exceptioon("initialtemperature must be a unique leaf.");

  InitialTemperature=xml->GetAttributeDouble(initial,"value");

  if(!Finite(InitialTemperature) || InitialTemperature<0)
    Run_Exceptioon("Initial temperature must be finite and >=0 K.");

  const TiXmlElement* file=prescribe->FirstChildElement("file");

  if(mode==1 && !file)
    Run_Exceptioon("Thermo mode 1 requires a file node.");

  if(file){
    xml->CheckAttributeNames(file,"value mk Renew comment");

    if(file->NextSiblingElement("file") || file->FirstChildElement())
      Run_Exceptioon("Thermo file must be a unique leaf.");

    InputFile=xml->GetAttributeStr(file,"value");
    if(InputFile.empty())
      Run_Exceptioon("Empty temperature file name.");

    FileMk=xml->GetAttributeWord(file,"mk");

    const int renew=xml->GetAttributeInt(file,"Renew",true,0);
    if(renew!=0 && renew!=1)
      Run_Exceptioon("file.Renew must be 0 or 1.");

    FileRenew=(renew==1);
  }

  // Compatibility with case1116's remaining phase-1 restartRenew mode=0.
  const TiXmlElement* legacy=node->FirstChildElement("restartRenew");

  if(legacy){
    xml->CheckAttributeNames(legacy,"mode comment");

    if(legacy->NextSiblingElement("restartRenew")
        || legacy->FirstChildElement()
        || xml->GetAttributeInt(legacy,"mode")!=0)
      Run_Exceptioon(
        "Remove old restartRenew; use prescribe/file Renew instead.");
  }

  Enabled=true;
}

void UserThermo::InitializeCpu(
    const std::string& caseDir,
    const std::string& restartDir,
    unsigned restartPart,double restartTime,
    unsigned np,const unsigned* idp,const typecode* code,
    const tdouble3* pos,const word* mk){
  if(!Enabled)return;

  Ready=false;
  UpdatedCount=0;
  Temperature.clear();

  if(np && (!idp || !code || !pos || !mk))
    Run_Exceptioon("Missing CPU thermal particle arrays.");

  std::unordered_map<unsigned,double> restored;

  if(restartPart){
    const std::string file=
      fun::GetDirWithSlash(restartDir)
      +fun::PrintStr("part_Thermo%04u.bi4",restartPart);

    JBinaryData data;
    data.LoadFile(file);

    const unsigned version=data.GetvUint("FormatVer");
    const double savedTime=data.GetvDouble("TimeStep");

    if((version!=1 && version!=2)
        || data.GetName()!="JPartThermoBi4"
        || data.GetvText("TemperatureUnits")!="K"
        || data.GetvUint("Cpart")!=restartPart
        || !Finite(savedTime) || !Finite(restartTime)
        || std::abs(savedTime-restartTime)
          >1e-9*std::max(1.0,std::abs(restartTime)))
      Run_ExceptioonFile("Thermal restart metadata mismatch.",file);

    const unsigned n=data.GetvUint("Np");

    const unsigned* ids=static_cast<const unsigned*>(
      data.GetArrayTpSize(
        "Idp",JBinaryDataDef::DatUint,n,file)->GetDataPointer());

    const double* values=static_cast<const double*>(
      data.GetArrayTpSize(
        "Temperature",JBinaryDataDef::DatDouble,n,file)->GetDataPointer());

    restored.reserve(n);

    for(unsigned p=0;p<n;p++){
      if(!Finite(values[p]) || values[p]<0
          || !restored.emplace(ids[p],values[p]).second)
        Run_ExceptioonFile(
          "Invalid or duplicate thermal restart particle.",file);
    }
  }

  Temperature.reserve(np);

  // First establish the initial/restart temperature for every normal ID.
  for(unsigned p=0;p<np;p++)if(CODE_IsNormal(code[p])){
    double t=InitialTemperature;

    if(restartPart){
      const auto it=restored.find(idp[p]);

      if(it==restored.end())
        Run_Exceptioon(
          "Thermal restart is missing particle ID "
          +std::to_string(idp[p]));

      t=it->second;
    }

    if(!Temperature.emplace(idp[p],t).second)
      Run_Exceptioon(
        "Duplicated normal particle ID during thermal initialization.");
  }

  // Then override only the selected MK within the file's spatial coverage.
  if(PrescribeMode==1 && (!restartPart || FileRenew)){
    const std::string file=
      fun::GetCanonicalPath(caseDir,InputFile);

    try{
      const TemperatureGrid grid(file);

      for(unsigned p=0;p<np;p++)
        if(CODE_IsNormal(code[p]) && mk[p]==FileMk){
          double t;

          if(grid.Sample(pos[p].x,pos[p].z,t)){
            Temperature.at(idp[p])=t;
            UpdatedCount++;
          }
        }
    }
    catch(const std::exception& error){
      Run_ExceptioonFile(error.what(),file);
    }
  }

  Ready=true;
}

double UserThermo::GetTemperature(unsigned idp)const{
  if(!Enabled)
    Run_Exceptioon("Thermo is disabled.");

  if(!Ready)
    Run_Exceptioon(
      "Thermo requires CPU initialization; GPU is not implemented.");

  const auto it=Temperature.find(idp);

  if(it==Temperature.end())
    Run_Exceptioon(
      "Temperature missing for particle ID "+std::to_string(idp));

  return it->second;
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
  data.SetvUint("FormatVer",2);
  data.SetvText("WriterRunCode",runCode);
  data.SetvUint("Cpart",part);
  data.SetvUint("Nstep",step);
  data.SetvDouble("TimeStep",time);
  data.SetvUint("Np",n);
  data.SetvUint("PrescribeMode",PrescribeMode);
  data.SetvDouble("InitialTemperature",InitialTemperature);
  data.SetvText("TemperatureUnits","K");
  data.SetvText("RestartPolicy","inherit_then_file_renew");

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