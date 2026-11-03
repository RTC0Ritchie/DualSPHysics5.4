#ifndef _UserGFlexPrecisionCore_iker_
#define _UserGFlexPrecisionCore_iker_

template<class Math> class UserGFlexPrecisionCore{
public:
  using R=typename Math::Real;
  using Access=UserGFlexAccess<Math>;
  using Matrix=typename Math::Matrix;
  using Vector=typename Math::Vector;
  using State=typename Math::State;

  static __device__ void Reference(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const unsigned p=t.flexstrucridp[i];
    s.Reference[i]=make_double3(t.posxy[p].x,t.posxy[p].y,t.posz[p]);
  }
  static __device__ void Displacement(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const unsigned p=t.flexstrucridp[i];const double3 x=s.Reference[i];
    s.Displacement[i]=make_double3(t.posxy[p].x-x.x,t.posxy[p].y-x.y,t.posz[p]-x.z);
  }
  template<TpKernel kernel> static __device__ void Pairs(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const double3 a=s.Reference[i];
    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const double3 b=s.Reference[j];
      const size_t q=t.pairidx[i]+k-t.pairidx[0];
      const double d[3]={a.x-b.x,a.y-b.y,a.z-b.z};
      const double rr=d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
      s.PairReference[q]={{R(d[0]),R(d[1]),R(d[2])}};s.PairNorm[q]=R(rr);
      if(sizeof(R)==sizeof(double)){
        const double fac=Math::template Fac64<kernel>(rr);
        s.PairGradient[q]={{R(fac*d[0]),R(fac*d[1]),R(fac*d[2])}};
      }
    }
  }
  template<TpKernel kernel,bool sim2d>
  static __device__ void Correction(const StInterParmsFlexStrucg& t,State s,unsigned i){
    double a[3][6]={};const double3 x=s.Reference[i];
    const double vol=t.flexstrucdata[CODE_GetIbodyFlexStruc(t.code[t.flexstrucridp[i]])].vol0;
    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const double3 y=s.Reference[j];
      const double d[3]={x.x-y.x,x.y-y.y,x.z-y.z};
      const auto g=Access::template Gradient<kernel>(t,s,i,k);
      for(int r=0;r<3;r++)for(int c=0;c<3;c++)a[r][c]-=vol*d[r]*double(g.v[c]);
    }
    if(sim2d){for(int k=0;k<3;k++)a[1][k]=a[k][1]=0;a[1][1]=1;}
    for(int r=0;r<3;r++)a[r][r+3]=1;
    // Partial pivoting, without model-dependent tolerance tuning.
    for(int k=0;k<3;k++){
      int pivot=k;for(int r=k+1;r<3;r++)if(fabs(a[r][k])>fabs(a[pivot][k]))pivot=r;
      if(a[pivot][k]==0 || !isfinite(a[pivot][k])){atomicExch(s.Error,1u);return;}
      if(pivot!=k)for(int c=0;c<6;c++){double z=a[k][c];a[k][c]=a[pivot][c];a[pivot][c]=z;}
      const double scale=a[k][k];for(int c=0;c<6;c++)a[k][c]/=scale;
      for(int r=0;r<3;r++)if(r!=k){
        const double scale=a[r][k];for(int c=0;c<6;c++)a[r][c]-=scale*a[k][c];
      }
    }
    tmatrix3f mirror;
    for(int r=0;r<3;r++)for(int c=0;c<3;c++){
      s.Correction[i].v[3*r+c]=R(a[r][c+3]);(&mirror.a11)[3*r+c]=float(a[r][c+3]);
    }
    const_cast<tmatrix3f*>(t.kercorr)[i]=mirror;
  }
  template<TpKernel kernel,bool sim2d>
  static __device__ void Geometry(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const auto mat=t.flexstrucdata[CODE_GetIbodyFlexStruc(t.code[t.flexstrucridp[i]])];
    const double3 ui=s.Displacement[i];Matrix sum={};
    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const double3 uj=s.Displacement[j];
      const R du[3]={R(ui.x-uj.x),R(ui.y-uj.y),R(ui.z-uj.z)};
      const auto g=Access::template Gradient<kernel>(t,s,i,k);
      for(int r=0;r<3;r++)for(int c=0;c<3;c++)
        if(!sim2d || (r!=1 && c!=1))sum.v[3*r+c]-=R(mat.vol0)*du[r]*g.v[c];
    }
    Matrix h=Math::Multiply(sum,Access::Correction(t,s,i));
    if(sim2d)h.v[1]=h.v[3]=h.v[4]=h.v[5]=h.v[7]=0;
    s.H[i]=h;
    const Matrix pk=Math::Stress(h,mat.cmat);
    s.PK[i]=pk;s.PB[i]=Math::Multiply(pk,s.Correction[i]);
    tmatrix3f f;
    for(int k=0;k<9;k++)(&f.a11)[k]=float(h.v[k]);
    f.a11+=1;f.a22+=1;f.a33+=1;t.defgrad[i]=f;
  }
  static __device__ void Normals(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const float3 n=t.boundnor0[i];if(n.x==0 && n.y==0 && n.z==0)return;
    Matrix f=s.H[i];f.v[0]+=1;f.v[4]+=1;f.v[8]+=1;
    const R* a=f.v;
    // cofactor(F) n = det(F) F^-T n. Preserve the sign of det(F).
    R r[3]={
      (a[4]*a[8]-a[5]*a[7])*R(n.x)+(a[5]*a[6]-a[3]*a[8])*R(n.y)+(a[3]*a[7]-a[4]*a[6])*R(n.z),
      (a[2]*a[7]-a[1]*a[8])*R(n.x)+(a[0]*a[8]-a[2]*a[6])*R(n.y)+(a[1]*a[6]-a[0]*a[7])*R(n.z),
      (a[1]*a[5]-a[2]*a[4])*R(n.x)+(a[2]*a[3]-a[0]*a[5])*R(n.y)+(a[0]*a[4]-a[1]*a[3])*R(n.z)};
    const R mag=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    const R scale=sqrt(R(n.x)*n.x+R(n.y)*n.y+R(n.z)*n.z)/mag*(Math::DeterminantF(s.H[i])<0?-1:1);
    t.boundnor[t.flexstrucridp[i]]=make_float3(float(r[0]*scale),float(r[1]*scale),float(r[2]*scale));
  }

  template<TpKernel tker,TpVisco tvisco,TpMdbc2Mode mdbc2,bool sim2d>
  static __device__ void Forces(const StInterParmsFlexStrucg& t,State s,unsigned i){
    const unsigned p1=t.flexstrucridp[i];const typecode codep1=t.code[p1];
    if(!CODE_IsFlexStrucFlex(codep1))return;
    const auto mat=t.flexstrucdata[CODE_GetIbodyFlexStruc(codep1)];
    const float vol0p1=mat.vol0,rho0p1=mat.rho0,mass0p1=vol0p1*rho0p1;
    float4 velrhop1=t.velrhop[p1];
    const float pressp1=cufsph::ComputePressCte(velrhop1.w);
    const R rhop1=R(rho0p1)/Math::DeterminantF(s.H[i]);
    const float csp1=float(sqrt(R(mat.youngmod)*(R(1)-R(mat.poissratio))
      /(rhop1*(R(1)+R(mat.poissratio))*(R(1)-R(2)*R(mat.poissratio)))));
    R acc[3]={};
    const auto& dvd=t.divdatag;
    const int scelldiv=dvd.scelldiv;const int4 nc=dvd.nc;const int3 cellzero=dvd.cellzero;
    const int2* beginendcellfluid=dvd.beginendcell+dvd.cellfluid;
    const auto dcell=t.dcell;const auto code=t.code;const auto velrhop=t.velrhop;
    const auto boundmode=t.boundmode;const auto tangenvel=t.tangenvel;
    const float visco=t.viscob;
      //-Modify values if using mDBC.
      float massp2=CTE.massf;
      if(mdbc2>=MDBC2_Std){
        if(boundmode[p1]==BMODE_MDBC2OFF)massp2=0;
        const float3 tangentvelp1=tangenvel[p1];
        velrhop1.x=tangentvelp1.x;
        velrhop1.y=tangentvelp1.y;
        velrhop1.z=tangentvelp1.z;
      }

      //-Obtains neighborhood search limits.
      int ini1,fin1,ini2,fin2,ini3,fin3;
      cunsearch::InitCte(dcell[p1],scelldiv,nc,cellzero,ini1,fin1,ini2,fin2,ini3,fin3);

      //-Flexible structure-Fluid interaction.
      for(int c3=ini3;c3<fin3;c3+=nc.w){
        for(int c2=ini2;c2<fin2;c2+=nc.x){
          unsigned pini,pfin=0;  cunsearch::ParticleRange(c2,c3,ini1,fin1,beginendcellfluid,pini,pfin);
          if(pfin){
            for(int p2=pini;p2<pfin;p2++){
              if(CODE_IsFluid(code[p2])){
                const double dx=t.posxy[p1].x-t.posxy[p2].x;
                const double dy=t.posxy[p1].y-t.posxy[p2].y;
                const double dz=t.posz[p1]-t.posz[p2];
                const float drx=float(dx),dry=float(dy),drz=float(dz);
                const float rr2=float(dx*dx+dy*dy+dz*dz);
                if(rr2<=CTE.kernelsize2&&rr2>=ALMOSTZERO){
                  //-Computes kernel.
                  const float fac=cufsph::GetKernel_Fac<tker>(rr2);
                  const float frx=fac*drx,fry=fac*dry,frz=fac*drz; //-Gradients.
                  //-Obtains velocity of particle p2 and compute difference.
                  const float4 velrhop2=velrhop[p2];
                  const float dvx=velrhop1.x-velrhop2.x,dvy=velrhop1.y-velrhop2.y,dvz=velrhop1.z-velrhop2.z;
                  //-Pressure derivative (Momentum equation).
                  const float pressp2=cufsph::ComputePressCte(velrhop2.w);
                  const float prs=(pressp1+pressp2)/(velrhop1.w*velrhop2.w)+(tker==KERNEL_Cubic?cufsph::GetKernelCubic_Tensil(rr2,velrhop1.w,pressp1,velrhop2.w,pressp2):0);
                  const float p_vpm=-prs*massp2*(CTE.massf/mass0p1);
                  acc[0]+=p_vpm*frx; acc[1]+=p_vpm*fry; acc[2]+=p_vpm*frz;                  
                  //-Artificial viscosity.
                  if(tvisco==VISCO_Artificial){
                    const float dot=drx*dvx+dry*dvy+drz*dvz;
                    if(dot<0){
                      const float dot_rr2=dot/(rr2+CTE.eta2);
                      const float amubar=CTE.kernelh*dot_rr2;
                      const float robar=(velrhop1.w+velrhop2.w)*0.5f;
                      const float pi_visc=(-visco*CTE.cs0*amubar/robar)*massp2*(CTE.massf/mass0p1);
                      acc[0]-=pi_visc*frx; acc[1]-=pi_visc*fry; acc[2]-=pi_visc*frz;
                    }
                  }
                  //-Laminar viscosity.
                  else{
                    const float robar2=(velrhop1.w+velrhop2.w);
                    const float temp=4.f*visco/((rr2+CTE.eta2)*robar2);
                    const float vtemp=massp2*temp*(drx*frx+dry*fry+drz*frz)*(CTE.massf/mass0p1);
                    acc[0]+=vtemp*dvx; acc[1]+=vtemp*dvy; acc[2]+=vtemp*dvz;
                  }
                }
              }
            }
          }
        }
      }


    for(unsigned k=0;k<t.numpairs[i];k++){
      const unsigned j=t.pairidx[i][k];const size_t flat=t.pairidx[i]+k-t.pairidx[0];
      const auto g=Access::template Gradient<tker>(t,s,i,k);
      const Matrix pi=s.PB[i];
      Matrix pj=s.PB[j];
      // Original law evaluates both stresses with particle i's constitutive tensor.
      // Keep that convention for pairs joining different material/body codes.
      if(CODE_GetIbodyFlexStruc(t.code[t.flexstrucridp[j]])!=CODE_GetIbodyFlexStruc(codep1))
        pj=Math::Multiply(Math::Stress(s.H[j],mat.cmat),s.Correction[j]);
      for(int r=0;r<3;r++){
        R force=0;for(int c=0;c<3;c++)force+=(pi.v[3*r+c]+pj.v[3*r+c])*g.v[c];
        acc[r]+=force*R(vol0p1)/R(rho0p1);
      }
      if(mat.hgfactor){
        const unsigned p2=t.flexstrucridp[j];
        const double dd[3]={t.posxy[p1].x-t.posxy[p2].x,t.posxy[p1].y-t.posxy[p2].y,t.posz[p1]-t.posz[p2]};
        const double rr64=dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2];
        const R rr2=R(rr64),rr=sqrt(rr2),d[3]={R(dd[0]),R(dd[1]),R(dd[2])};
        const R wab=sizeof(R)==sizeof(double)?R(Math::template Wab64<tker>(rr64))
          :R(cufsph::GetKernel_Wab<tker>(float(rr2)));
        const auto ref=s.PairReference[flat];const auto ui=s.Displacement[i],uj=s.Displacement[j];
        const R du[3]={R(ui.x-uj.x),R(ui.y-uj.y),R(ui.z-uj.z)};
        const Matrix hi=s.H[i],hj=s.H[j];R delta=0;
        for(int r=0;r<3;r++){
          if(sim2d && r==1)continue;
          R vi=0,vj=0;
          for(int c=0;c<3;c++){
            if(sim2d && c==1)continue;
            vi+=hi.v[3*r+c]*ref.v[c];vj+=hj.v[3*r+c]*ref.v[c];
          }
          delta+=((du[r]-vi)+(du[r]-vj))*d[r]/rr;
        }
        // Mixed keeps the original float HG denominator; extended keeps it double.
        const R rr0=ref.v[0]*ref.v[0]+ref.v[1]*ref.v[1]+ref.v[2]*ref.v[2];
        const R scale=R(mat.hgfactor)*R(vol0p1)*R(vol0p1)*wab*R(mat.youngmod)
          /(rr0*rr*R(mass0p1))*R(.5)*delta;
        for(int r=0;r<3;r++)acc[r]-=scale*d[r];
      }
    }
    float3 a=t.ace[p1];a.x=float(R(a.x)+acc[0]);a.y=float(R(a.y)+acc[1]);a.z=float(R(a.z)+acc[2]);
    t.ace[p1]=a;
    if(csp1>t.flexstrucdt[i])t.flexstrucdt[i]=csp1;
  }
};

// Thin CUDA entry points; all physics lives in the policy-independent class.
template<class Math,int stage,TpKernel kernel,bool sim2d>
__global__ void KerUserGFlexPrecision(StInterParmsFlexStrucg t,typename Math::State s){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=t.vnpfs)return;
  using Core=UserGFlexPrecisionCore<Math>;
  if(stage==0)Core::Reference(t,s,i);
  if(stage==1)Core::template Pairs<kernel>(t,s,i);
  if(stage==2)Core::template Correction<kernel,sim2d>(t,s,i);
  if(stage==3)Core::Displacement(t,s,i);
  if(stage==4)Core::template Geometry<kernel,sim2d>(t,s,i);
  if(stage==5)Core::Normals(t,s,i);
}
template<class Math,TpKernel kernel,TpVisco visco,TpMdbc2Mode mdbc,bool sim2d>
__global__ void KerUserGFlexPrecisionForces(StInterParmsFlexStrucg t,typename Math::State s){
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<t.vnpfs)UserGFlexPrecisionCore<Math>::template Forces<kernel,visco,mdbc,sim2d>(t,s,i);
}
#endif
