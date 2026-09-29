#ifndef USERPARAMETERS_H
#define USERPARAMETERS_H

#include "JObject.h"

class JXml;
class JLog2;

class UserParameters : protected JObject
{
public:
  struct StPoint2d{
    double x=0.0;
    double z=0.0;
  };

  struct StSurface2d{
    // 0: unused; 1: horizontal wall; 2: two-point geometry.
    int value=0;

    double WallZ=0.0;

    StPoint2d point1;
    StPoint2d point2;
  };

  // 1: particle contact; 2: surface contact.
  int FlexBoundContactMode;

  StSurface2d SurfaceType;

  int KnContactMode; // 1: manual; 2: automatic.
  double KnContact;
  int KnContactLinearity; // 1: String (linear); 2: Hertzian (3/2 power).

  int unContactMode; // 1: manual; 2: automatic.
  double unContact;

  UserParameters();

  void Reset();
  void LoadXml(const JXml* xml,JLog2* log);
  void VisuConfig(JLog2* log)const;
};

#endif