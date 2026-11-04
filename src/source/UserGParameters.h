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

  // POD snapshot: no host pointers are sent to device code.
  struct StContactConfig{
    int Mode=1, Surface=0, NormalMode=0;
    int KnMode=1, DampingMode=1, Linearity=1;
    double Kn=0, Damping=0;
    double WallZ=0, X1=0, Z1=0, Slope=0, Nx=0, Nz=1;
  };
  static StContactConfig ReadContact(const UserParameters* user);

  // Returns a small host-side configuration snapshot.
  static StFlexConfig ReadFlex(const UserParameters* user);
};

#endif