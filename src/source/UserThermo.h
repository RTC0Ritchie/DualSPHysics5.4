#ifndef USERTHERMO_H
#define USERTHERMO_H

#include "JObject.h"
#include "DualSphDef.h"
#include <string>

class JXml;
class TiXmlElement;

class UserThermo : protected JObject
{
  bool Enabled=false;
  unsigned PrescribeMode=0;
  double InitialTemperature=0.0;

public:
  UserThermo();

  void LoadXml(
    const JXml* xml,const TiXmlElement* node);

  bool IsEnabled()const{ return Enabled; }

  double GetTemperature(unsigned idp)const;

  void SaveSnapshot(
    const std::string& directory,const std::string& runCode,
    unsigned part,unsigned step,double time,
    unsigned np,const unsigned* idp,const typecode* code)const;
};

#endif