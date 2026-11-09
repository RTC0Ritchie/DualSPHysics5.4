#ifndef USER_INOUT_BOUND_MODE_H
#define USER_INOUT_BOUND_MODE_H

#include <map>

class UserParameters;

class UserInOutBoundMode final
{
public:
  // XML request only; not the applied solver state.
  class StConfig
  {
  public:
    bool Enabled=false;

    // -1: unspecified; 0: closed boundary; 1: open inlet.
    int ValveStatue=-1;
  };

  // Missing configuration returns defaults.
  static StConfig Read(const UserParameters* user);

  // The physical valve implementation is not connected yet.
  static bool IsImplemented(){ return false; }

private:
  friend class UserParameters;

  UserInOutBoundMode()=delete;

  static std::map<const UserParameters*,StConfig>& Configs();

  static void Reset(const UserParameters* user);
  static void Store(const UserParameters* user,const StConfig& config);
};

#endif