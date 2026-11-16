#ifndef USERTHERMO_H
#define USERTHERMO_H

#include "JObject.h"
#include "DualSphDef.h"
#include <string>
#include <unordered_map>

class JXml;
class TiXmlElement;

class UserThermo : protected JObject
{
  bool Enabled=false;
  bool Ready=false;
  unsigned PrescribeMode=0;
  double InitialTemperature=0.0;
  std::string InputFile;
  word FileMk=0;
  bool FileRenew=false;
  unsigned UpdatedCount=0;
  std::unordered_map<unsigned,double> Temperature;

public:
  UserThermo();
  void LoadXml(const JXml* xml,const TiXmlElement* node);

  bool IsEnabled()const{ return Enabled; }
  unsigned GetPrescribeMode()const{ return PrescribeMode; }
  double GetInitialTemperature()const{ return InitialTemperature; }
  unsigned GetUpdatedCount()const{ return UpdatedCount; }

  // Shared initialization from host arrays (CPU or downloaded GPU state).
  void InitializeHost(const std::string& caseDir,
    const std::string& restartDir,unsigned restartPart,double restartTime,
    unsigned np,const unsigned* idp,const typecode* code,
    const tdouble3* pos,const word* mk);

  void InitializeCpu(const std::string& caseDir,
    const std::string& restartDir,unsigned restartPart,double restartTime,
    unsigned np,const unsigned* idp,const typecode* code,
    const tdouble3* pos,const word* mk);

  double GetTemperature(unsigned idp)const;

  void SaveSnapshot(
    const std::string& directory,const std::string& runCode,
    unsigned part,unsigned step,double time,
    unsigned np,const unsigned* idp,const typecode* code)const;
};

#endif