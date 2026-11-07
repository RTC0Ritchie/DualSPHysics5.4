#ifndef USERFLEXKINEMATICS_H
#define USERFLEXKINEMATICS_H

#include "TypesDef.h"

// Shared host-side geometry preparation for CPU and GPU elastic bodies.
class UserFlexKinematics{
public:
  struct NormalReport{
    unsigned Checked=0,Corrected=0,Unresolved=0;
  };
  // pos contains one solid body INCLUDING its clamp particles. Normals point
  // toward the fluid; their lengths may be either interface or ghost distances.
  // Repair only confidently inward normals. Preserve lengths, zeros and outward
  // normals. No fluid data, water level, material model or precision policy.
  static NormalReport RepairNormalOrientation(bool twoD,double dp,unsigned n,
    const tdouble3* pos,tfloat3* normal);
};

#endif
