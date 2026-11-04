#include "JSphGpuSingle.h"
#include "UserParameters.h"
#include "UserGParameters.h"
#include "UserGFlexPrecision.h"
#include "Functions.h"
#include "FunctionsMath.h"
#include "FunSphKernel.h"
#include "JLog2.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// CPU counterpart: UserFlexBoundaryCorrection.cpp, modes 1-4.
// Only initialization reads GPU reference arrays. All step forces stay on GPU.
class UserGFlexBoundaryHost{
public:
  template<class T> static void Copy(std::vector<T>& dst,size_t n,const T* source){
    dst.resize(n);if(!n)return;
    const auto error=cudaMemcpy(dst.data(),source,n*sizeof(T),cudaMemcpyDeviceToHost);
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
  }
  template<class M> static M Cap(const M& input,double limit){
    double a[3][3]={
      {input.a11,
       0.5*(double(input.a12)+input.a21),
       0.5*(double(input.a13)+input.a31)},
      {0.5*(double(input.a21)+input.a12),
       input.a22,
       0.5*(double(input.a23)+input.a32)},
      {0.5*(double(input.a31)+input.a13),
       0.5*(double(input.a32)+input.a23),
       input.a33}
    };

    double v[3][3]={{1,0,0},{0,1,0},{0,0,1}};

    for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
        if(!std::isfinite(a[i][j]))
          throw std::runtime_error(
              "Boundary correction: non-finite kernel matrix.");

    // Symmetric Jacobi eigensolver; valid for 2D and 3D.
    for(int iter=0;iter<32;iter++){
      int p=0,q=1;

      if(std::fabs(a[0][2])>std::fabs(a[p][q])){
        p=0;
        q=2;
      }
      if(std::fabs(a[1][2])>std::fabs(a[p][q])){
        p=1;
        q=2;
      }

      const double scale=std::max(1.0,
          std::max(std::fabs(a[0][0]),
          std::max(std::fabs(a[1][1]),std::fabs(a[2][2]))));

      if(std::fabs(a[p][q])<=1.e-12*scale)break;

      const double angle=0.5*std::atan2(
          2.0*a[p][q],a[q][q]-a[p][p]);

      const double c=std::cos(angle),s=std::sin(angle);
      const double app=a[p][p],aqq=a[q][q],apq=a[p][q];

      for(int k=0;k<3;k++)if(k!=p && k!=q){
        const double akp=a[k][p],akq=a[k][q];
        a[k][p]=a[p][k]=c*akp-s*akq;
        a[k][q]=a[q][k]=s*akp+c*akq;
      }

      a[p][p]=c*c*app-2.0*c*s*apq+s*s*aqq;
      a[q][q]=s*s*app+2.0*c*s*apq+c*c*aqq;
      a[p][q]=a[q][p]=0.0;

      for(int k=0;k<3;k++){
        const double vkp=v[k][p],vkq=v[k][q];
        v[k][p]=c*vkp-s*vkq;
        v[k][q]=s*vkp+c*vkq;
      }
    }

    bool changed=false;


    // Upper clipping only; smaller eigenvalues are retained.
    for(int k=0;k<3;k++){
      if(!std::isfinite(a[k][k]))
        throw std::runtime_error(
            "Boundary correction: non-finite kernel eigenvalue.");

      if(a[k][k]>limit){
        a[k][k]=limit;
        changed=true;
      }
    }

    if(!changed)return input;

    double b[3][3]={};

    for(int i=0;i<3;i++)
      for(int j=0;j<3;j++)
        for(int k=0;k<3;k++)
          b[i][j]+=v[i][k]*a[k][k]*v[j][k];

    M result;
    for(int i=0;i<3;i++)for(int j=0;j<3;j++)(&result.a11)[3*i+j]=b[i][j];
    return result;
  }
};

void JSphGpuSingle::InitFlexRestartBoundaryCorrection()
{
  auto& data=User->FlexRestartBoundaryData;
  data=UserParameters::StFlexRestartBoundaryData{};

  const int mode=User->FlexRestartBoundaryCorrectionMode;
  if(mode==0)return;

  if(mode<1 || mode>4)
    Run_Exceptioon("Invalid boundary correction mode.");



  if(User->UseFlexRestartGradientCorrection)
    Run_Exceptioon(
        "Boundary correction cannot use transported direct-continuation gradients.");


  const unsigned np=CaseNflexstruc;
  if(!np)Run_Exceptioon("GPU boundary correction requires elastic particles.");
  std::vector<unsigned> ridp,counts,pairs;
  std::vector<typecode> code;
  std::vector<double2> xy;
  std::vector<double> z;
  std::vector<tmatrix3f> correction;
  UserGFlexBoundaryHost::Copy(ridp,np,FlexStrucRidpg);
  UserGFlexBoundaryHost::Copy(code,Npb,Code_g->cptr());
  UserGFlexBoundaryHost::Copy(counts,np,NumPairsg);
  size_t totalPairs=0;for(unsigned n:counts)totalPairs+=n;
  UserGFlexBoundaryHost::Copy(pairs,totalPairs,PairIdxBufferg);
  UserGFlexBoundaryHost::Copy(xy,Npb,Posxy_g->cptr());
  UserGFlexBoundaryHost::Copy(z,Npb,Posz_g->cptr());
  UserGFlexBoundaryHost::Copy(correction,np,KerCorrg);
  std::vector<tdouble3> reference(np);
  std::vector<const unsigned*> pairPointers(np);
  size_t offset=0;
  for(unsigned i=0;i<np;i++){
    reference[i]=TDouble3(xy[ridp[i]].x,xy[ridp[i]].y,z[ridp[i]]);
    pairPointers[i]=pairs.data()+offset;offset+=counts[i];
  }
  const auto Pos0c=reference.data();const auto KerCorrc=correction.data();
  const auto NumPairsc=counts.data();const auto PairIdxc=pairPointers.data();
  const auto FlexStrucRidpc=ridp.data();
  std::vector<tmatrix3d> forceKernel64;
  if(!np || !Pos0c || !KerCorrc || !NumPairsc || !PairIdxc)
    Run_Exceptioon(
        "Boundary correction: reference geometry is unavailable.");

  data.Body.resize(np);
  data.Selected.assign(np,1);

  for(unsigned i=0;i<np;i++){
    data.Body[i]=CODE_GetIbodyFlexStruc(
        code[FlexStrucRidpc[i]]);

    if(data.Body[i]>=FlexStrucCount)
      Run_Exceptioon(
          "Boundary correction: invalid elastic body.");
  }

  const int layers=User->FlexRestartBoundaryCorrectionLayers;

  if(layers>0){
    // Peel geometric shells in X0, independently for each elastic body.
    // A missing angular sector identifies an exposed particle.
    const double cutoff2=1.8*1.8*Dp*Dp;
    const double pi=std::acos(-1.0);
    const double angleTolerance=1.e-10;

    std::vector<std::vector<unsigned>> neighbours(np);
    for(unsigned i=0;i<np;i++){
      for(unsigned k=0;k<NumPairsc[i];k++){
        const unsigned j=PairIdxc[i][k];
        if(j>=np || j==i || data.Body[i]!=data.Body[j])continue;

        const tdouble3 r=Pos0c[j]-Pos0c[i];
        const double rr2=r.x*r.x+r.z*r.z+
            (Simulate2D ? 0.0 : r.y*r.y);
        if(rr2>0.0 && rr2<=cutoff2)
          neighbours[i].push_back(j);
      }
    }

    // 98 approximately distributed directions for the 3D empty-cone test.
    std::vector<tdouble3> probeDirections;
    if(!Simulate2D){
      for(int x=-2;x<=2;x++)
        for(int y=-2;y<=2;y++)
          for(int z=-2;z<=2;z++){
            if(std::max(std::abs(x),
                std::max(std::abs(y),std::abs(z)))!=2)continue;
            const double length=std::sqrt(double(x*x+y*y+z*z));
            probeDirections.push_back(
                TDouble3(x/length,y/length,z/length));
          }
    }

    data.Selected.assign(np,0);
    std::vector<byte> queued(np,0);
    std::vector<unsigned> candidates(np),shell,next;
    for(unsigned i=0;i<np;i++)candidates[i]=i;

    std::vector<double> angles;
    std::vector<tdouble3> directions;
    angles.reserve(64);
    directions.reserve(64);

    for(int layer=0;layer<layers && !candidates.empty();layer++){
      shell.clear();

      // Evaluate the complete shell before removing any of its particles.
      for(const unsigned i:candidates){
        angles.clear();
        directions.clear();

        for(const unsigned j:neighbours[i]){
          if(data.Selected[j])continue;
          const tdouble3 r=Pos0c[j]-Pos0c[i];

          if(Simulate2D)angles.push_back(std::atan2(r.z,r.x));
          else{
            const double length=std::sqrt(
                r.x*r.x+r.y*r.y+r.z*r.z);
            directions.push_back(
                TDouble3(r.x/length,r.y/length,r.z/length));
          }
        }

        bool exposed=false;
        if(Simulate2D){
          if(angles.size()<2)exposed=true;
          else{
            std::sort(angles.begin(),angles.end());
            double gap=angles.front()+2.0*pi-angles.back();
            for(size_t k=1;k<angles.size();k++)
              gap=std::max(gap,angles[k]-angles[k-1]);

            // A >=120 degree empty sector tolerates sloped/staircase edges.
            exposed=(gap>=2.0*pi/3.0-angleTolerance);
          }
        }
        else{
          // An empty cone with a 60 degree half-angle marks a 3D surface.
          for(const tdouble3& probe:probeDirections){
            bool occupied=false;
            for(const tdouble3& direction:directions){
              const double dot=probe.x*direction.x+
                  probe.y*direction.y+probe.z*direction.z;
              if(dot>=0.5-angleTolerance){
                occupied=true;
                break;
              }
            }
            if(!occupied){
              exposed=true;
              break;
            }
          }
        }

        if(exposed)shell.push_back(i);
      }

      if(shell.empty())break;
      for(const unsigned i:shell)data.Selected[i]=1;

      // Only neighbours of the removed shell can become newly exposed.
      next.clear();
      for(const unsigned i:shell){
        for(const unsigned j:neighbours[i]){
          if(!data.Selected[j] && !queued[j]){
            queued[j]=1;
            next.push_back(j);
          }
        }
      }
      for(const unsigned i:next)queued[i]=0;
      candidates.swap(next);
    }
  }



  if(mode==2){
    data.ForceKernel.resize(np);

    for(unsigned i=0;i<np;i++)
      data.ForceKernel[i]=data.Selected[i]
          ? UserGFlexBoundaryHost::Cap(KerCorrc[i],User->FlexRestartBoundaryCorrectionLimit) : KerCorrc[i];

    if(UserGParameters::ReadFlex(User.get()).PrecisionMode==UserGParameters::TpFlexPrecisionMode::Extended){
      FlexPrecision->DownloadCorrection64(np,forceKernel64);
      for(unsigned i=0;i<np;i++)if(data.Selected[i])
        forceKernel64[i]=UserGFlexBoundaryHost::Cap(forceKernel64[i],User->FlexRestartBoundaryCorrectionLimit);
    }
  }



  if(mode==3 && !User->IsFlexRestart())
    Log->Print(
        "  Mode 3: ordinary stress-free reference; "
        "prestress compensation is zero.");



  if(mode==4){
    // Geometry-only calibration. Material reference volumes are NOT used here.
    // C maps scalar weights to the assembled constant-stress area vectors.
    struct Entry{ unsigned row; tdouble3 c; };
    std::vector<std::vector<Entry>> columns(np);
    std::vector<tdouble3> target(np,TDouble3(0)),residual(np,TDouble3(0));
    std::vector<double> beta(np,1.0),diagonal(np,0.0);
    const double vg=Simulate2D ? Dp*Dp : Dp*Dp*Dp;
    const double scale=vg*Dp; // vg^2 / dp^(dimension-1)
    const double eta2=1.e-6,betaMin=0.05;

    const auto dot=[](const tdouble3& a,const tdouble3& b){
      return a.x*b.x+a.y*b.y+a.z*b.z;
    };

    const auto gradient=[&](unsigned i,unsigned j){
      const tdouble3 r=Pos0c[i]-Pos0c[j];
      const float x=float(r.x),y=Simulate2D ? 0.f : float(r.y),z=float(r.z);
      const float rr=x*x+y*y+z*z;
      const float fac=TKernel==KERNEL_Wendland
          ? fsph::GetKernel_Fac<KERNEL_Wendland>(CSP,rr)
          : fsph::GetKernel_Fac<KERNEL_Cubic>(CSP,rr);
      return TDouble3(double(fac)*x,double(fac)*y,double(fac)*z);
    };

    for(unsigned i=0;i<np;i++){
      double a[3][3]={};

      for(unsigned k=0;k<NumPairsc[i];k++){
        const unsigned j=PairIdxc[i][k];
        if(j>=np || j==i || data.Body[i]!=data.Body[j])continue;

        const tdouble3 r=Pos0c[j]-Pos0c[i],g=gradient(i,j);
        const double rv[3]={r.x,Simulate2D ? 0.0 : r.y,r.z};
        const double gv[3]={g.x,g.y,g.z};

        for(unsigned u=0;u<3;u++)for(unsigned v=0;v<3;v++)
          a[u][v]+=vg*rv[u]*gv[v];
      }

      if(Simulate2D)a[1][1]=1.0;

      const tmatrix3d m=TMatrix3d(a[0][0],a[0][1],a[0][2],
          a[1][0],a[1][1],a[1][2],a[2][0],a[2][1],a[2][2]);

      const double det=fmath::Determinant3x3(m);
      if(!std::isfinite(det) || det<=1.e-12)
        Run_Exceptioon("Mode 4: singular geometric kernel moment.");

      const tmatrix3d b=fmath::InverseMatrix3x3(m,det);
      tdouble3 center=TDouble3(0);

      for(unsigned k=0;k<NumPairsc[i];k++){
        const unsigned j=PairIdxc[i][k];
        if(j>=np || j==i || data.Body[i]!=data.Body[j])continue;

        const tdouble3 g=gradient(i,j);
        const tdouble3 raw=TDouble3(scale*g.x,scale*g.y,scale*g.z);

        const tdouble3 q=TDouble3(
            scale*(b.a11*g.x+b.a12*g.y+b.a13*g.z),
            scale*(b.a21*g.x+b.a22*g.y+b.a23*g.z),
            scale*(b.a31*g.x+b.a32*g.y+b.a33*g.z));

        target[i]=target[i]-raw;
        target[j]=target[j]+raw;
        center=center-q;
        columns[i].push_back(Entry{j,q});
      }

      columns[i].push_back(Entry{i,center});

      for(const Entry& e:columns[i]){
        residual[e.row]=residual[e.row]+e.c;
        diagonal[i]+=dot(e.c,e.c);
      }
    }

    double target2=0.0;
    for(unsigned i=0;i<np;i++){
      residual[i]=residual[i]-target[i];
      target2+=dot(target[i],target[i]);
    }

    // Strictly convex bounded least squares; fixed particles stay at beta=1.
    // Include ALL residual rows, including the unmodified transition region.
    bool converged=false;
    unsigned sweeps=0;
    double projected=0.0;

    for(;sweeps<200000;sweeps++){
      for(unsigned i=0;i<np;i++)if(data.Selected[i]){
        double grad=eta2*(beta[i]-1.0);
        for(const Entry& e:columns[i])
          grad+=dot(e.c,residual[e.row]);

        const double next=std::max(betaMin,std::min(1.0,
            beta[i]-grad/(diagonal[i]+eta2)));
        const double change=next-beta[i];
        beta[i]=next;

        for(const Entry& e:columns[i])
          residual[e.row]=residual[e.row]+TDouble3(
              change*e.c.x,change*e.c.y,change*e.c.z);
      }

      projected=0.0;
      for(unsigned i=0;i<np;i++)if(data.Selected[i]){
        double grad=eta2*(beta[i]-1.0);
        for(const Entry& e:columns[i])
          grad+=dot(e.c,residual[e.row]);

        const double next=std::max(betaMin,std::min(1.0,
            beta[i]-grad/(diagonal[i]+eta2)));
        projected=std::max(projected,std::fabs(next-beta[i]));
      }

      if(projected<1.e-9){
        converged=true;
        break;
      }
    }

    if(!converged)
      Run_Exceptioon(fun::PrintStr(
          "Mode 4: geometric beta optimization did not converge "
          "(projected=%.9g).",projected));

    data.Beta.resize(np);
    double r2=0.0,rmax=0.0,bmin=1.0,bmax=0.0;

    for(unsigned i=0;i<np;i++){
      data.Beta[i]=float(beta[i]);
      const double rr=dot(residual[i],residual[i]);
      r2+=rr;
      rmax=std::max(rmax,std::sqrt(rr));
      bmin=std::min(bmin,beta[i]);
      bmax=std::max(bmax,beta[i]);
    }

    Log->Printf(
        "  Mode 4: geometry-only beta, sweeps=%u, "
        "range=[%.9g,%.9g], projected=%.3g.",
        sweeps+1,bmin,bmax,projected);

    Log->Printf(
        "  Mode 4: area residual L2=%.6g, "
        "max/dp^(d-1)=%.6g; eta=0.001, bounds=[0.05,1].",
        std::sqrt(r2/std::max(target2,1.e-30)),rmax);
  }

  std::vector<unsigned> total(FlexStrucCount,0);
  std::vector<unsigned> selected(FlexStrucCount,0);

  for(unsigned i=0;i<np;i++){
    total[data.Body[i]]++;
    if(data.Selected[i])selected[data.Body[i]]++;
  }

  for(unsigned b=0;b<FlexStrucCount;b++)
    Log->Printf(
        "  Boundary correction: mode=%d, body=%u, scope=%u/%u particles.",
        mode,b,selected[b],total[b]);
  FlexPrecision->UploadBoundary(unsigned(mode),np,data.Selected.data(),
      data.Beta.empty()?nullptr:data.Beta.data(),
      data.ForceKernel.empty()?nullptr:data.ForceKernel.data(),
      forceKernel64.empty()?nullptr:forceKernel64.data(),
      FlexStrucDatag,Code_g->cptr(),FlexStrucRidpg);
}

