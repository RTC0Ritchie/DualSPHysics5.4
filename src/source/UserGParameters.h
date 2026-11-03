#ifndef _UserGParameters_
#define _UserGParameters_

class UserParameters;

// Host-side access to user settings for GPU modules.
// Never pass UserParameters pointers to CUDA kernels.
class UserGParameters{
private:
  UserGParameters()=delete;

public:
  enum class TpFlexPrecisionMode : unsigned{
    Original=0,
    Mixed=1,
    Extended=2
  };

  struct StFlexConfig{
    TpFlexPrecisionMode PrecisionMode=TpFlexPrecisionMode::Original;
  };

  // Returns a small host-side configuration snapshot.
  static StFlexConfig ReadFlex(const UserParameters* user);
};

#endif