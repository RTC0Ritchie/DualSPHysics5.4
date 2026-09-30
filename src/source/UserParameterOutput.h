#ifndef USERPARAMETEROUTPUT_H
#define USERPARAMETEROUTPUT_H

#include "JObject.h"
#include <string>

class UserParameterOutput : protected JObject
{
public:
  UserParameterOutput();

  // CSV
  void SaveCsv(const std::string& filename);

  // DualSPHysics VTK
  void SaveVTK(const std::string& filename);
};

#endif