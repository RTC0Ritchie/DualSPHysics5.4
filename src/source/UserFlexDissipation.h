#ifndef USERFLEXDISSIPATION_H
#define USERFLEXDISSIPATION_H

#include "UserFlexKinematics.h"

class UserParameters;

namespace FlexDissipation{

struct Cache;

double Apply(
    const UserParameters& user,
    const FlexKinematics::Cache& k,
    const tdouble3* x,
    const typecode* code,
    const unsigned* rid,
    const StFlexStrucData* material,
    const std::function<double(double)>& dW,
    tfloat3* acceleration,
    double time,
    unsigned step);

}

#endif