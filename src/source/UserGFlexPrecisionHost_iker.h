#ifndef _UserGFlexPrecisionHost_iker_
#define _UserGFlexPrecisionHost_iker_

class UserGFlexPrecision::Impl{
public:
  unsigned Mode=0,Particles=0;
  size_t Bytes=0;
  std::vector<void*> Buffers;
  UserGFlexState<float> Mixed={};
  UserGFlexState<double> Extended={};
  static void Check(cudaError_t err){
    if(err!=cudaSuccess)throw std::runtime_error(std::string("GPU elastic precision: ")+cudaGetErrorString(err));
  }
  template<class T> void Alloc(T*& p,size_t n){
    if(!n){p=nullptr;return;}
    Check(cudaMalloc(reinterpret_cast<void**>(&p),sizeof(T)*n));
    Buffers.push_back(p);Bytes+=sizeof(T)*n;
  }
  template<class R> void Allocate(UserGFlexState<R>& s,unsigned n,unsigned pairs){
    Alloc(s.Reference,n);Alloc(s.Displacement,n);
    Alloc(s.PairReference,pairs);Alloc(s.PairNorm,pairs);
    if(sizeof(R)==sizeof(double))Alloc(s.PairGradient,pairs);
    Alloc(s.Correction,n);Alloc(s.H,n);Alloc(s.PK,n);Alloc(s.PB,n);Alloc(s.Error,1);
    Check(cudaMemset(s.Error,0,sizeof(unsigned)));
  }
  ~Impl(){for(void* p:Buffers)cudaFree(p);}
};
UserGFlexPrecision::UserGFlexPrecision():Data(nullptr){}
UserGFlexPrecision::~UserGFlexPrecision(){delete Data;}
void UserGFlexPrecision::Reset(){delete Data;Data=nullptr;}
void UserGFlexPrecision::Allocate(unsigned mode,unsigned particles,unsigned pairs){
  Reset();
  if(!mode)return;
  if(mode>2)throw std::runtime_error("Invalid GPU elastic precision mode.");
  Data=new Impl;Data->Mode=mode;Data->Particles=particles;
  try{
    if(mode==1)Data->Allocate(Data->Mixed,particles,pairs);
    else Data->Allocate(Data->Extended,particles,pairs);
  }
  catch(...){Reset();throw;}
}
void* UserGFlexPrecision::State(unsigned mode,unsigned particles)const{
  if(!Data || Data->Mode!=mode || Data->Particles!=particles)
    throw std::runtime_error("GPU elastic precision cache mismatch. Reinitialize before changing mode or topology.");
  return mode==1?static_cast<void*>(&Data->Mixed):static_cast<void*>(&Data->Extended);
}
size_t UserGFlexPrecision::GetAllocMemoryGpu()const{return Data?Data->Bytes:0;}

// Host launch selection only. No precision switch is evaluated on the GPU.
template<class Math> class UserGFlexPrecisionBackend{
  using State=typename Math::State;
  static void Check(){
    const auto e=cudaGetLastError();
    if(e!=cudaSuccess)throw std::runtime_error(std::string("GPU elastic precision launch: ")+cudaGetErrorString(e));
  }
  template<TpKernel kernel,TpVisco visco,TpMdbc2Mode mdbc,bool sim2d>
  static void Force(const StInterParmsFlexStrucg& t,State s){
    const unsigned blocks=(t.vnpfs+SPHBSIZE-1)/SPHBSIZE;
    cusph::usergflex::KerUserGFlexPrecisionForces<Math,kernel,visco,mdbc,sim2d>
      <<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
  }
  template<TpKernel kernel,TpVisco visco,bool sim2d>
  static void Boundary(const StInterParmsFlexStrucg& t,State s){
#ifdef FAST_COMPILATION
    if(t.mdbc2!=MDBC2_None)throw "Extra mDBC options are disabled for FastCompilation...";
    Force<kernel,visco,MDBC2_None,sim2d>(t,s);
#else
    if(t.mdbc2==MDBC2_None)Force<kernel,visco,MDBC2_None,sim2d>(t,s);
    else if(t.mdbc2==MDBC2_Std)Force<kernel,visco,MDBC2_Std,sim2d>(t,s);
    else if(t.mdbc2==MDBC2_NoPen)Force<kernel,visco,MDBC2_NoPen,sim2d>(t,s);
    else throw "Unsupported elastic mDBC mode.";
#endif
  }
  template<UserGFlexStage stage,TpKernel kernel,bool sim2d>
  static void Launch(const StInterParmsFlexStrucg& t,State s){
    const unsigned blocks=(t.vnpfs+SPHBSIZE-1)/SPHBSIZE;
    if(stage==UserGFlexStage::Reference){
      cusph::usergflex::KerUserGFlexPrecision<Math,0,kernel,sim2d><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      cusph::usergflex::KerUserGFlexPrecision<Math,1,kernel,sim2d><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      cusph::usergflex::KerUserGFlexPrecision<Math,2,kernel,sim2d><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      Check();
      // One 4-byte validation download at initialization, never in the time loop.
      unsigned error=0;
      auto e=cudaMemcpyAsync(&error,s.Error,sizeof(error),cudaMemcpyDeviceToHost,t.stm);
      if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
      e=cudaStreamSynchronize(t.stm);
      if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
      if(error)throw std::runtime_error("Singular elastic reference correction matrix.");
    }
    else if(stage==UserGFlexStage::Geometry){
      cusph::usergflex::KerUserGFlexPrecision<Math,3,kernel,sim2d><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      cusph::usergflex::KerUserGFlexPrecision<Math,4,kernel,sim2d><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
      if(t.mdbc2>=MDBC2_Std)
        cusph::usergflex::KerUserGFlexPrecision<Math,5,kernel,sim2d><<<blocks,SPHBSIZE,0,t.stm>>>(t,s);
    }
    else{
#ifdef FAST_COMPILATION
      if(t.tvisco!=VISCO_Artificial)throw "Extra viscosity options are disabled for FastCompilation...";
      Boundary<kernel,VISCO_Artificial,sim2d>(t,s);
#else
      if(t.tvisco==VISCO_Artificial)Boundary<kernel,VISCO_Artificial,sim2d>(t,s);
      else if(t.tvisco==VISCO_Laminar)Boundary<kernel,VISCO_Laminar,sim2d>(t,s);
      else if(t.tvisco==VISCO_LaminarSPS)Boundary<kernel,VISCO_LaminarSPS,sim2d>(t,s);
      else throw "Unsupported elastic viscosity.";
#endif
    }
    Check();
  }
  template<UserGFlexStage stage,TpKernel kernel>
  static void Dimension(const StInterParmsFlexStrucg& t,State s){
    if(t.simulate2d)Launch<stage,kernel,true>(t,s);
    else Launch<stage,kernel,false>(t,s);
  }
public:
  template<UserGFlexStage stage> static void Run(const StInterParmsFlexStrucg& input){
    if(!input.vnpfs)return;
    if(!input.precision || !input.posxy || !input.posz)
      throw std::runtime_error("Missing elastic precision context or double positions.");
    const State s=*static_cast<State*>(input.precision->State(sizeof(typename Math::Real)==4?1:2,input.vnpfs));
    // No host object address is passed into the device kernels.
    auto t=input;t.user=nullptr;t.precision=nullptr;
#ifdef FAST_COMPILATION
    if(t.tkernel!=KERNEL_Wendland)throw "Extra kernels are disabled for FastCompilation...";
    Dimension<stage,KERNEL_Wendland>(t,s);
#else
    if(t.tkernel==KERNEL_Wendland)Dimension<stage,KERNEL_Wendland>(t,s);
#ifndef DISABLE_KERNELS_EXTRA
    else if(t.tkernel==KERNEL_Cubic)Dimension<stage,KERNEL_Cubic>(t,s);
#endif
    else throw "Unsupported elastic kernel.";
#endif
  }
};
#endif
