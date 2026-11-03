#ifndef _UserGFlexBackend_iker_
#define _UserGFlexBackend_iker_

enum class UserGFlexStage{Reference,Geometry,Forces};

template<class Math>
class UserGFlexBackend{
private:
  UserGFlexBackend()=delete;

  template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2>
  static void Interaction_ForcesFlexStrucT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker,TpVisco tvisco>
  static void Interaction_ForcesFlexStruc_gt1(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void Interaction_ForcesFlexStruc_gt0(const StInterParmsFlexStrucg& tfs);

  // CPU counterpart: UserInteractionForcesFlexStruc.cpp.
  template<TpKernel tker,bool simulate2d>
  static void CalcFlexStrucKerCorrT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void CalcFlexStrucKerCorr_gt0(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker,bool simulate2d>
  static void CalcFlexStrucDefGradT(const StInterParmsFlexStrucg& tfs);

  template<TpKernel tker>
  static void CalcFlexStrucDefGrad_ct0(const StInterParmsFlexStrucg& tfs);

public:
  template<UserGFlexStage stage>
  static void Run(const StInterParmsFlexStrucg& tfs){
    if(stage==UserGFlexStage::Reference)CalcFlexStrucKerCorr(tfs);
    else if(stage==UserGFlexStage::Geometry)UpdateFlexStrucGeometry(tfs);
    else Interaction_ForcesFlexStruc(tfs);
  }

  static void CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs);
  static void UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs);
  static void Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs);
};

template<class Math>
template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2> void UserGFlexBackend<Math>::Interaction_ForcesFlexStrucT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    const StDivDataGpu& dvd=tfs.divdatag;
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerInteractionForcesFlexStruc<Math,tker,tvisco,mdbc2> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.viscob,dvd.scelldiv,dvd.nc,dvd.cellzero,dvd.beginendcell+dvd.cellfluid,tfs.dcell,tfs.poscell,tfs.velrhop,tfs.code,tfs.boundmode,tfs.tangenvel,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad,tfs.flexstrucdt,tfs.ace);
  }
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<class Math>
template<TpKernel tker,TpVisco tvisco> void UserGFlexBackend<Math>::Interaction_ForcesFlexStruc_gt1(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.mdbc2!=MDBC2_None)throw "Extra mDBC options are disabled for FastCompilation...";
  Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_None>(tfs);
#else
  if(tfs.mdbc2==MDBC2_None)      Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_None> (tfs);
  else if(tfs.mdbc2==MDBC2_Std)  Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_Std>  (tfs);
  else if(tfs.mdbc2==MDBC2_NoPen)Interaction_ForcesFlexStrucT<tker,tvisco,MDBC2_NoPen>(tfs);
#endif
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<class Math>
template<TpKernel tker> void UserGFlexBackend<Math>::Interaction_ForcesFlexStruc_gt0(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tvisco!=VISCO_Artificial)throw "Extra viscosity options are disabled for FastCompilation...";
  Interaction_ForcesFlexStruc_gt1<tker,VISCO_Artificial> (tfs);
#else
  if(tfs.tvisco==VISCO_Artificial)     Interaction_ForcesFlexStruc_gt1<tker,VISCO_Artificial>(tfs);
  else if(tfs.tvisco==VISCO_Laminar)   Interaction_ForcesFlexStruc_gt1<tker,VISCO_Laminar>   (tfs);
  else if(tfs.tvisco==VISCO_LaminarSPS)Interaction_ForcesFlexStruc_gt1<tker,VISCO_LaminarSPS>(tfs);
#endif
}

//==============================================================================
/// Interaction forces for the flexible structure particles.
/// Fuerzas de interacción para las partículas de estructura flexible.
//==============================================================================
template<class Math>
void UserGFlexBackend<Math>::Interaction_ForcesFlexStruc(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  Interaction_ForcesFlexStruc_gt0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)  Interaction_ForcesFlexStruc_gt0<KERNEL_Wendland>(tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)Interaction_ForcesFlexStruc_gt0<KERNEL_Cubic>   (tfs);
#endif
#endif
}

template<class Math>
template<TpKernel tker,bool simulate2d> void UserGFlexBackend<Math>::CalcFlexStrucKerCorrT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucKerCorr<Math,tker,simulate2d> <<<sgridb,SPHBSIZE>>>
        (tfs.vnpfs,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,const_cast<tmatrix3f*>(tfs.kercorr));
  }
}

template<class Math>
template<TpKernel tker> void UserGFlexBackend<Math>::CalcFlexStrucKerCorr_gt0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucKerCorrT<tker,true>  (tfs);
  else              CalcFlexStrucKerCorrT<tker,false> (tfs);
}

template<class Math>
void UserGFlexBackend<Math>::CalcFlexStrucKerCorr(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  CalcFlexStrucKerCorr_gt0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)     CalcFlexStrucKerCorr_gt0<KERNEL_Wendland> (tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)   CalcFlexStrucKerCorr_gt0<KERNEL_Cubic   > (tfs);
#endif
#endif
}

template<class Math>
template<TpKernel tker,bool simulate2d> void UserGFlexBackend<Math>::CalcFlexStrucDefGradT(const StInterParmsFlexStrucg& tfs){
  if(tfs.vnpfs){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucDefGrad<Math,tker,simulate2d> <<<sgridb,SPHBSIZE,0,tfs.stm>>>
        (tfs.vnpfs,tfs.poscell,tfs.code,tfs.flexstrucdata,tfs.flexstrucridp,tfs.poscell0,tfs.numpairs,tfs.pairidx,tfs.kercorr,tfs.defgrad);
  }
}

template<class Math>
template<TpKernel tker> void UserGFlexBackend<Math>::CalcFlexStrucDefGrad_ct0(const StInterParmsFlexStrucg& tfs){
  if(tfs.simulate2d)CalcFlexStrucDefGradT<tker,true>  (tfs);
  else              CalcFlexStrucDefGradT<tker,false> (tfs);
}

template<class Math>
void UserGFlexBackend<Math>::UpdateFlexStrucGeometry(const StInterParmsFlexStrucg& tfs){
#ifdef FAST_COMPILATION
  if(tfs.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
  CalcFlexStrucDefGrad_ct0<KERNEL_Wendland> (tfs);
#else
  if(tfs.tkernel==KERNEL_Wendland)  CalcFlexStrucDefGrad_ct0<KERNEL_Wendland>(tfs);
#ifndef DISABLE_KERNELS_EXTRA
  else if(tfs.tkernel==KERNEL_Cubic)CalcFlexStrucDefGrad_ct0<KERNEL_Cubic>   (tfs);
#endif
#endif
  if(tfs.vnpfs&&tfs.mdbc2>=MDBC2_Std){
    dim3 sgridb=cusph::usergflex::GetSimpleGridSize(tfs.vnpfs,SPHBSIZE);
    cusph::usergflex::KerCalcFlexStrucNormals<Math> <<<sgridb,SPHBSIZE,0,tfs.stm>>>(tfs.vnpfs,tfs.flexstrucridp,tfs.defgrad,tfs.boundnor0,tfs.boundnor);
  }
}


#endif
