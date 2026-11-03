#ifndef _UserGFlexCte_
#define _UserGFlexCte_

#include "JSphGpu_cte.h"

class UserGFlexCte{
private:
  UserGFlexCte()=delete;

public:
  static void Upload(const StCteInteraction* cte);
};

#endif
