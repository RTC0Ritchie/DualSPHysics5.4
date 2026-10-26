#include "UserFlexBoundaryCorrection.h"
#include "OmpDefs.h"
#include "JSphCpu.h"
#include "UserParameters.h"
#include "Functions.h"
#include "FunctionsMath.h"
#include "FunSphKernel.h"
#include "JLog2.h"
#include "UserFlexInline.h"

#include <exception>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>
#include <queue>
#include <limits>

namespace FlexBoundaryCorrection{

using Vec=std::array<double,3>;
using Mat=std::vector<double>;

struct Edge{
  unsigned j;
  Vec R,D,raw;
};

struct RegularizedState;
struct Cache{
  std::shared_ptr<RegularizedState> reg;
  unsigned maxIterations=2000;
  double quadraticError=0;
  std::vector<tdouble3> quadraticReference;
  std::vector<unsigned> quadraticBody;
  std::vector<byte> quadraticSelected;
  double kernelH=0,eventTime=0;
  bool eventPending=false,localReady=false,stepClockReady=false;
  ContactGeometry pendingContact;
  std::vector<double> mass,referenceNorm2;
  AuxiliaryLoads loads;
  int dim=2;

  std::vector<double> volume;
  std::vector<std::vector<Edge>> rows;

  bool mixed=false;
  double dp=0;


  // Fixed positive interpolation weights onto existing surface particles.
  std::vector<std::vector<std::pair<unsigned,double>>> blend;

  std::vector<double> eta;


  std::vector<byte> active;
  // Continuous inward mobility; zero rows retain the local kernel gradient.
  std::vector<double> inwardWeight;

  // Runtime graph and reusable work arrays.
  bool runtimeReady=false,hgReady=false,fixedReady=false;
  std::vector<size_t> offset;
  std::vector<std::vector<size_t>> incoming;
  std::vector<tdouble3> flux,hgFlux;
  std::vector<Vec> hgR;
  std::vector<double> hgWeight,gate;
  std::vector<tmatrix3f> totalPK;
  std::vector<byte> used;
  std::map<std::array<long long,3>,std::vector<Vec>> fixedCells;
  std::vector<tdouble3> fixedSaved;
};

namespace RegularizedMG{

using Vector=std::vector<double>;
using Basis=std::vector<Vector>;
using Point=std::array<double,3>;
using Row=std::vector<std::pair<unsigned,double>>;

static void Require(bool ok,const char* message){
  if(!ok)throw std::runtime_error(message);
}

static double Inner(const Vector& a,const Vector& b){
  double s=0;

  for(size_t i=0;i<a.size();i++)
    s+=a[i]*b[i];

  return s;
}

static void Remove(Vector& x,const Basis& q){
  for(int pass=0;pass<2;pass++)
    for(const auto& v:q){
      const double a=Inner(x,v);

      for(size_t i=0;i<x.size();i++)
        x[i]-=a*v[i];
    }
}

static Basis Orthonormal(const Basis& b){
  Basis q;

  for(auto x:b){
    const double original=std::sqrt(Inner(x,x));

    Remove(x,q);

    const double length=std::sqrt(Inner(x,x));

    if(length<=1.e-12*original || length==0)
      continue;

    for(double& v:x)
      v/=length;

    q.push_back(std::move(x));
  }

  return q;
}

struct Sparse{
  unsigned cols=0;
  std::vector<Row> row;
  struct Packed{
    std::vector<size_t> offset;
    std::vector<unsigned> index;
    std::vector<double> value;
  };
  std::shared_ptr<const Packed> packed;
  // Rows are immutable after packing; copies share the contiguous storage.
  void Pack(){
    if(packed)return;
    auto data=std::make_shared<Packed>();data->offset.reserve(row.size()+1);data->offset.push_back(0);
    size_t count=0;for(const auto& r:row)count+=r.size();data->index.reserve(count);data->value.reserve(count);
    for(const auto& r:row){for(const auto& e:r){data->index.push_back(e.first);data->value.push_back(e.second);}data->offset.push_back(data->index.size());}
    packed=data;
  }

  Sparse()=default;

  Sparse(unsigned n,unsigned m)
    :cols(m),row(n){}

  Vector Apply(const Vector& x)const{
    Vector y(row.size(),0);
    if(packed){
      const auto& p=*packed;
      for(size_t i=0;i<row.size();i++){
        double sum=0;for(size_t e=p.offset[i];e<p.offset[i+1];e++)sum+=p.value[e]*x[p.index[e]];y[i]=sum;
      }
      return y;
    }

    for(size_t i=0;i<row.size();i++){
      double sum=0;
      for(const auto& e:row[i])sum+=e.second*x[e.first];
      y[i]=sum;
    }

    return y;
  }

  Sparse Transpose()const{
    Sparse t(cols,unsigned(row.size()));

    for(unsigned i=0;i<row.size();i++)
      for(const auto& e:row[i])
        t.row[e.first].emplace_back(i,e.second);

    return t;
  }

  Sparse Multiply(const Sparse& b)const{
    Require(
        cols==b.row.size(),
        "MG sparse dimensions.");

    Sparse c(unsigned(row.size()),b.cols);

    Vector value(b.cols,0);
    std::vector<size_t> mark(b.cols,row.size());
    std::vector<unsigned> used;

    for(size_t i=0;i<row.size();i++){
      used.clear();

      for(const auto& e:row[i])
        for(const auto& f:b.row[e.first]){
          const unsigned j=f.first;

          if(mark[j]!=i){
            mark[j]=i;
            value[j]=0;
            used.push_back(j);
          }

          value[j]+=e.second*f.second;
        }

      std::sort(used.begin(),used.end());

      for(unsigned j:used)
        if(value[j]!=0)
          c.row[i].emplace_back(j,value[j]);
    }

    return c;
  }
};

class Solver{
  struct Level{
    Sparse a,p,pt;
    Vector weight,lower;
    Basis nullspace;
  };

  std::vector<Level> levels;
  Basis nullspace;

  static void Partition(
      const std::vector<Point>& x,int d,
      std::vector<unsigned> ids,
      std::vector<std::vector<unsigned>>& groups)
  {
    if(ids.size()<=32){
      groups.push_back(std::move(ids));
      return;
    }

    int axis=0;
    double longest=-1;

    for(int k=0;k<d;k++){
      double lo=x[ids[0]][k],hi=lo;

      for(unsigned i:ids){
        lo=std::min(lo,x[i][k]);
        hi=std::max(hi,x[i][k]);
      }

      if(hi-lo>longest){
        longest=hi-lo;
        axis=k;
      }
    }

    const size_t mid=ids.size()/2;

    std::nth_element(
        ids.begin(),ids.begin()+mid,ids.end(),
        [&](unsigned i,unsigned j){
          return x[i][axis]<x[j][axis] ||
              (x[i][axis]==x[j][axis] && i<j);
        });

    std::vector<unsigned> right(
        ids.begin()+mid,ids.end());

    ids.resize(mid);

    Partition(x,d,std::move(ids),groups);
    Partition(x,d,std::move(right),groups);
  }

  void Build(
      Sparse a,Basis b,
      std::vector<Point> x,int d)
  {
    const unsigned n=unsigned(a.row.size());

    Level l;
    l.a=std::move(a);
    l.a.Pack();

    Vector diag(n,0);

    for(unsigned i=0;i<n;i++)
      for(const auto& e:l.a.row[i])
        if(e.first==i)
          diag[i]=e.second;

    for(double v:diag)
      Require(
          std::isfinite(v)&&v>0,
          "MG invalid diagonal.");

    if(n<128){
      // The shifted system is positive definite; retain all coarse modes.
      l.lower.assign(size_t(n)*n,0);

      for(unsigned i=0;i<n;i++)
        for(const auto& e:l.a.row[i])
          l.lower[size_t(i)*n+e.first]=e.second;

      const double scale=
          *std::max_element(diag.begin(),diag.end());

      // Stabilize affine gauge modes before Cholesky factorization.
      for(unsigned i=0;i<n;i++)
        for(unsigned j=0;j<=i;j++){
          double v=.5*(
              l.lower[size_t(i)*n+j]+
              l.lower[size_t(j)*n+i]);

          for(const auto& q:l.nullspace)
            v+=scale*q[i]*q[j];

          for(unsigned k=0;k<j;k++)
            v-=l.lower[size_t(i)*n+k]*
                l.lower[size_t(j)*n+k];

          if(i==j){
            Require(
                std::isfinite(v)&&v>1.e-16*scale,
                "MG coarse factorization failed.");

            l.lower[size_t(i)*n+j]=std::sqrt(v);
          }
          else{
            l.lower[size_t(i)*n+j]=
                v/l.lower[size_t(j)*n+j];
          }
        }

      levels.push_back(std::move(l));
      return;
    }

    double bound=0;

    for(unsigned i=0;i<n;i++){
      double sum=0;

      for(const auto& e:l.a.row[i])
        sum+=std::abs(e.second)/
            std::sqrt(diag[i]*diag[e.first]);

      bound=std::max(bound,sum);
    }

    l.weight.resize(n);

    for(unsigned i=0;i<n;i++)
      l.weight[i]=1.4/(bound*diag[i]);

    std::vector<unsigned> ids(n);

    for(unsigned i=0;i<n;i++)
      ids[i]=i;

    std::vector<std::vector<unsigned>> groups;
    Partition(x,d,std::move(ids),groups);

    Sparse t(n,0);
    Basis bc(b.size());
    std::vector<Point> xc;

    for(const auto& group:groups){
      Basis local(b.size(),Vector(group.size()));
      Point center={{0,0,0}};

      for(size_t j=0;j<group.size();j++){
        for(size_t k=0;k<b.size();k++)
          local[k][j]=b[k][group[j]];

        for(int k=0;k<d;k++)
          center[k]+=x[group[j]][k]/group.size();
      }

      const Basis q=Orthonormal(local);

      for(const auto& col:q){
        for(size_t j=0;j<group.size();j++)
          t.row[group[j]].emplace_back(t.cols,col[j]);

        for(size_t k=0;k<b.size();k++)
          bc[k].push_back(Inner(col,local[k]));

        xc.push_back(center);
        t.cols++;
      }
    }

    Require(
        t.cols>0 && t.cols<n,
        "MG coarsening stalled.");

    Sparse at=l.a.Multiply(t);
    l.p=Sparse(n,t.cols);

    // Smooth the tentative interpolation.
    for(unsigned i=0;i<n;i++){
      std::map<unsigned,double> values;

      for(const auto& e:t.row[i])
        values[e.first]+=e.second;

      for(const auto& e:at.row[i])
        values[e.first]-=l.weight[i]*e.second;

      for(const auto& e:values)
        if(e.second!=0)
          l.p.row[i].push_back(e);
    }

    l.pt=l.p.Transpose();
    l.p.Pack();l.pt.Pack();

    Sparse coarse=
        l.pt.Multiply(l.a.Multiply(l.p));

    levels.push_back(std::move(l));

    Build(
        std::move(coarse),
        std::move(bc),
        std::move(xc),d);
  }

  Vector Cycle(size_t level,const Vector& rhs)const{
    const auto& l=levels[level];
    const size_t n=rhs.size();

    Vector x(n,0);

    if(!l.lower.empty()){
      x=rhs;
      // No gauge projection for the shifted system.

      for(size_t i=0;i<n;i++){
        for(size_t j=0;j<i;j++)
          x[i]-=l.lower[i*n+j]*x[j];

        x[i]/=l.lower[i*n+i];
      }

      for(size_t i=n;i-->0;){
        for(size_t j=i+1;j<n;j++)
          x[i]-=l.lower[j*n+i]*x[j];

        x[i]/=l.lower[i*n+i];
      }

      // No gauge projection for the shifted system.
      return x;
    }

    const auto smooth=[&](){
      const auto ax=l.a.Apply(x);

      for(size_t i=0;i<n;i++)
        x[i]+=l.weight[i]*(rhs[i]-ax[i]);
    };

    for(int k=0;k<3;k++)
      smooth();

    Vector r=l.a.Apply(x);

    for(size_t i=0;i<n;i++)
      r[i]=rhs[i]-r[i];

    const auto correction=
        l.p.Apply(Cycle(level+1,l.pt.Apply(r)));

    for(size_t i=0;i<n;i++)
      x[i]+=correction[i];

    for(int k=0;k<3;k++)
      smooth();

    return x;
  }

public:
  Solver(
      Sparse a,
      const std::vector<Point>& x,int d)
  {
    Require(
        !x.empty() && a.row.size()==x.size(),
        "MG empty or invalid dimensions.");

    Basis b(d+1,Vector(x.size(),1));

    for(int k=0;k<d;k++){
      double mean=0;

      for(const auto& p:x)
        mean+=p[k]/x.size();

      for(size_t i=0;i<x.size();i++)
        b[k+1][i]=x[i][k]-mean;
    }

    nullspace=Orthonormal(b);

    Build(std::move(a),nullspace,x,d);
  }

  Vector Apply(Vector r)const{
    // No gauge projection for the shifted system.
    r=Cycle(0,r);
    // No gauge projection for the shifted system.
    return r;
  }


};

}


struct RegularizedState{
  RegularizedMG::Sparse matrix;
  std::vector<double> referenceDiagonal;
  std::shared_ptr<RegularizedMG::Solver> solver;
  std::vector<unsigned> ids;
  std::vector<byte> active;
  std::vector<Mat> inverse;
  std::vector<Vec> fluid,solid,baseArea;
  std::vector<std::vector<Vec>> particular;
  RegularizedMG::Vector lambda[3];
  double rhsNorm[3]={0,0,0};
  double initialResidual=0,lastEvent=0;
};

namespace{

void Check(bool ok,const char* message){
  if(!ok)
    throw std::runtime_error(std::string("Mode 5: ")+message);
}

void PrepareRuntime(Cache& c){
  if(c.runtimeReady)return;
  const unsigned n=c.rows.size();

  c.offset.assign(n+1,0);
  c.incoming.assign(n,{});
  for(unsigned i=0;i<n;i++)
    c.offset[i+1]=c.offset[i]+c.rows[i].size();

  for(unsigned i=0;i<n;i++)
    for(size_t e=0;e<c.rows[i].size();e++)
      c.incoming[c.rows[i][e].j].push_back(c.offset[i]+e);

  const size_t ne=c.offset[n];
  c.flux.resize(ne);
  c.hgFlux.resize(ne);
  c.totalPK.resize(n);
  c.gate.assign(n,0);
  c.used.assign(n,0);

  if(c.mixed)
    for(unsigned i=0;i<n;i++)
      for(const auto& w:c.blend[i]){
        c.used[w.first]=1;
      }

  c.runtimeReady=true;
}

tdouble3 GatherFlux(
    const Cache& c,unsigned i,const std::vector<tdouble3>& flux)
{
  tdouble3 q=TDouble3(0);
  for(size_t e=c.offset[i];e<c.offset[i+1];e++)
    q=q+flux[e];
  for(size_t e:c.incoming[i])
    q=q-flux[e];
  return q;
}

tdouble3 ApplyPK(const tmatrix3f& p,const Vec& D,int d){
  return d==2
      ? TDouble3(p.a11*D[0]+p.a13*D[1],0,
                 p.a31*D[0]+p.a33*D[1])
      : TDouble3(p.a11*D[0]+p.a12*D[1]+p.a13*D[2],
                 p.a21*D[0]+p.a22*D[1]+p.a23*D[2],
                 p.a31*D[0]+p.a32*D[1]+p.a33*D[2]);
}

void PrepareHourglass(
    Cache& c,const tdouble3* reference,
    const std::function<double(double)>& kernel)
{
  if(c.hgReady)return;

  c.hgR.resize(c.offset.back());
  c.hgWeight.resize(c.offset.back());

  for(unsigned i=0;i<c.rows.size();i++)
    for(size_t e=0;e<c.rows[i].size();e++){
      const unsigned j=c.rows[i][e].j;
      const tdouble3 r=reference[j]-reference[i];
      const Vec R=c.dim==2
          ? Vec{{r.x,r.z,0}}
          : Vec{{r.x,r.y,r.z}};

      double rr=0;
      for(int k=0;k<c.dim;k++)
        rr+=R[k]*R[k];

      Check(
          std::isfinite(rr)&&rr>0,
          "invalid reference hourglass pair.");

      const double W=kernel(rr);
      Check(
          std::isfinite(W)&&W>=0,
          "invalid reference hourglass kernel.");

      const size_t id=c.offset[i]+e;
      c.hgR[id]=R;
      c.hgWeight[id]=0.5*c.volume[i]*c.volume[j]*W/rr;
    }

  c.hgReady=true;
}

Vec Coordinates(const tdouble3& x,int d){
  return d==2
      ? Vec{{x.x,x.z,0}}
      : Vec{{x.x,x.y,x.z}};
}

double Dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
  double s=0;
  for(size_t k=0;k<a.size();k++)s+=a[k]*b[k];
  return s;
}

// Small dense systems only: dimensions 2/3 or 6/12.
std::vector<double> Solve(
    Mat a,std::vector<double> b,int n)
{
  double scale=0;
  for(double v:a)scale=std::max(scale,std::abs(v));

  Check(scale>0,"singular local geometry.");

  for(int k=0;k<n;k++){
    int p=k;

    for(int i=k+1;i<n;i++)
      if(std::abs(a[i*n+k])>std::abs(a[p*n+k]))
        p=i;

    Check(
        std::abs(a[p*n+k])>1.e-14*scale,
        "singular local moment system.");

    for(int j=0;j<n;j++)
      std::swap(a[k*n+j],a[p*n+j]);

    std::swap(b[k],b[p]);

    const double pivot=a[k*n+k];

    for(int j=k;j<n;j++)a[k*n+j]/=pivot;
    b[k]/=pivot;

    for(int i=0;i<n;i++)if(i!=k){
      const double v=a[i*n+k];

      for(int j=k;j<n;j++)
        a[i*n+j]-=v*a[k*n+j];

      b[i]-=v*b[k];
    }
  }

  return b;
}

Mat Inverse(const Mat& a,int d){
  Mat inv(d*d);

  for(int k=0;k<d;k++){
    std::vector<double> b(d,0);
    b[k]=1;
    b=Solve(a,b,d);

    for(int i=0;i<d;i++)
      inv[i*d+k]=b[i];
  }

  return inv;
}

struct Face{
  unsigned i;
  Vec normal;
  double area;
};

void CorrectArea(
    const std::vector<unsigned>& ids,
    const std::vector<Vec>& X,
    const std::vector<double>& v,int d,
    const std::vector<Face>& faces,
    std::vector<Vec>& target)
{
  const int m=d+d*d;

  Check(
      !faces.empty(),
      "no directed boundary area for an elastic component.");

  Vec center={{0,0,0}};
  double volume=0;

  for(unsigned i:ids){
    volume+=v[i];
    for(int k=0;k<d;k++)
      center[k]+=v[i]*X[i][k];
  }

  for(int k=0;k<d;k++)center[k]/=volume;

  double length=1;

  for(unsigned i:ids)
    for(int k=0;k<d;k++)
      length=std::max(
          length,std::abs(X[i][k]-center[k]));

  std::vector<Mat> columns(faces.size(),Mat(m,0));
  std::vector<double> b(m,0),lambda(m,0);
  std::vector<double> amplitude(faces.size());

  for(int k=0;k<d;k++)
    b[d+k*d+k]=volume/length;

  for(size_t f=0;f<faces.size();f++){
    const Face& face=faces[f];

    for(int k=0;k<d;k++)
      columns[f][k]=face.normal[k];

    for(int a=0;a<d;a++)
      for(int k=0;k<d;k++)
        columns[f][d+a*d+k]=
            (X[face.i][a]-center[a])/length*
            face.normal[k];
  }

  const auto objective=
      [&](const std::vector<double>& l)->double
  {
    double value=Dot(b,l);

    for(size_t f=0;f<faces.size();f++){
      const double a=std::max(
          0.,faces[f].area-Dot(columns[f],l));
      value+=.5*a*a;
    }

    return value;
  };

  bool converged=false;

  for(unsigned it=0;it<300;it++){
    std::vector<double> residual=b;
    for(double& x:residual)x=-x;

    Mat hessian(m*m,0);

    for(size_t f=0;f<faces.size();f++){
      amplitude[f]=std::max(
          0.,faces[f].area-Dot(columns[f],lambda));

      for(int a=0;a<m;a++)
        residual[a]+=columns[f][a]*amplitude[f];

      if(amplitude[f]>0)
        for(int a=0;a<m;a++)
          for(int k=0;k<m;k++)
            hessian[a*m+k]+=
                columns[f][a]*columns[f][k];
    }

    if(std::sqrt(Dot(residual,residual))<=
       1.e-12*std::max(1.,std::sqrt(Dot(b,b))))
    {
      converged=true;
      break;
    }

    double diagonal=0;

    for(int k=0;k<m;k++)
      diagonal=std::max(diagonal,hessian[k*m+k]);

    for(int k=0;k<m;k++)
      hessian[k*m+k]+=
          1.e-10*std::max(1.,diagonal);

    const auto step=Solve(hessian,residual,m);
    const double decrease=Dot(residual,step);
    const double old=objective(lambda);

    bool accepted=false;
    double scale=1;

    for(int trial=0;trial<50;trial++,scale*=.5){
      auto next=lambda;

      for(int k=0;k<m;k++)
        next[k]+=scale*step[k];

      if(objective(next)<=
         old-1.e-4*scale*decrease+
         1.e-14*std::max(1.,std::abs(old)))
      {
        lambda.swap(next);
        accepted=true;
        break;
      }
    }

    Check(
        accepted,
        "positive area projection failed; "
        "inspect reference geometry/volumes.");
  }

  Check(
      converged,
      "positive area constraints did not converge.");

  for(size_t f=0;f<faces.size();f++)
    for(int k=0;k<d;k++)
      target[faces[f].i][k]+=
          amplitude[f]*faces[f].normal[k];
}

Cache& Get(
    const std::shared_ptr<Cache>& c,unsigned n)
{
  Check(
      c && c->rows.size()==n,
      "missing or inconsistent cache.");

  return *c;
}

}

std::string ValidateOptions(const Options& o){
  if(o.Layers<0)
    return "Boundary correction layers must be nonnegative.";

  if(!std::isfinite(o.InwardReach)||o.InwardReach<0)
    return
        "Boundary correction inwardreach must be finite "
        "and nonnegative.";

  if(!std::isfinite(o.QuadraticError)||o.QuadraticError<0)
    return "Mode 5 quadraticerror must be finite and nonnegative.";
  if(o.MaxIterations==0)return "Mode 5 maxiterations must be positive.";

  if(o.BoundaryForm<1||o.BoundaryForm>3)
    return
        "Boundary correction boundaryform must be 1, 2 or 3.";

  return std::string();
}

namespace{
using Array=std::vector<double>;
// Expand along short same-body edges, rather than through empty gaps.
Array InwardWeights(
    const Options& o,int d,double dp,unsigned n,
    const tdouble3* reference,const unsigned* body,
    const byte* selected,const unsigned* count,
    const unsigned* const* pairs)
{
  Array distance(
      n,std::numeric_limits<double>::infinity());
  Array weight(n,0);

  using Item=std::pair<double,unsigned>;
  std::priority_queue<
      Item,std::vector<Item>,std::greater<Item>> queue;

  for(unsigned i=0;i<n;i++)if(selected[i]){
    distance[i]=0;
    weight[i]=1;
    queue.push({0,i});
  }

  if(o.InwardReach==0)return weight;

  const double reach=o.InwardReach*o.KernelH;

  while(!queue.empty()){
    const auto item=queue.top();
    queue.pop();

    const unsigned i=item.second;

    if(item.first!=distance[i]||item.first>=reach)
      continue;

    const Vec xi=Coordinates(reference[i],d);

    for(unsigned e=0;e<count[i];e++){
      const unsigned j=pairs[i][e];

      if(j>=n||body[j]!=body[i])continue;

      const Vec xj=Coordinates(reference[j],d);
      double rr=0;

      for(int k=0;k<d;k++)
        rr+=(xj[k]-xi[k])*(xj[k]-xi[k]);

      const double r=std::sqrt(rr);
      if(!(r>0&&r<=1.8*dp))continue;

      const double next=item.first+r;

      if(next<reach&&next<distance[j]){
        distance[j]=next;
        queue.push({next,j});
      }
    }
  }

  for(unsigned i=0;i<n;i++)if(distance[i]<reach){
    const double t=distance[i]/reach;
    weight[i]=(1-t)*(1-t)*(1+2*t);
  }

  return weight;
}

}
std::shared_ptr<Cache> BuildTemplate(
    const Options& o,bool twoD,double dp,unsigned n,
    const tdouble3* reference,const unsigned* body,
    const byte* selected,const float* volume,
    const unsigned* count,const unsigned* const* pairs,
    const std::function<double(double)>& kernelFac,
    std::vector<Vec>& physicalTarget)
{


  const auto error=ValidateOptions(o);
  if(!error.empty())throw std::invalid_argument(error);

  if(o.InwardReach>0 || o.QuadraticError>0)
    Check(
        std::isfinite(o.KernelH)&&o.KernelH>0,
        "positive inwardreach requires "
        "the solver kernel length.");

  Check(
      n && reference && body && selected && volume &&
      count && pairs && kernelFac,
      "incomplete initialization data.");

  Check(
      std::isfinite(dp)&&dp>0,
      "invalid particle spacing.");
  
  Check(
      o.ParticleMass && o.ParticleMassCount==n,
      "particle mass is unavailable for geometry initialization.");

  for(unsigned i=0;i<n;i++)
    Check(
        std::isfinite(double(o.ParticleMass[i])) &&
        o.ParticleMass[i]>0,
        "invalid particle mass for geometry initialization.");

  auto c=std::make_shared<Cache>();
  const int d=twoD?2:3;

  c->dim=d;
  c->volume.assign(volume,volume+n);
  c->rows.resize(n);

  const auto weights=InwardWeights(
      o,d,dp,n,reference,body,selected,count,pairs);

  c->active.resize(n);
  c->inwardWeight=weights;

  for(unsigned i=0;i<n;i++){
    c->active[i]=weights[i]>0 && count[i]>0;
  }

  std::vector<Vec> X(n);
  std::vector<Vec> rawArea(n,Vec{{0,0,0}});
  std::vector<Vec> target(n,Vec{{0,0,0}});
  std::vector<double> v(n);
  std::map<unsigned,std::vector<unsigned>> groups;

  const double unitVolume=std::pow(dp,d);
  const double unitGradient=std::pow(dp,d+1);

  for(unsigned i=0;i<n;i++){
    Check(
        std::isfinite(volume[i])&&volume[i]>0,
        "invalid reference volume.");

    X[i]=Coordinates(reference[i],d);

    for(int k=0;k<d;k++){
      Check(
          std::isfinite(X[i][k]),
          "nonfinite reference coordinate.");
      X[i][k]/=dp;
    }

    v[i]=volume[i]/unitVolume;

    // Disconnected clamp markers have no reconstruction or elastic force.
    if(count[i])groups[body[i]].push_back(i);
  }

  for(unsigned i=0;i<n;i++)if(count[i]){
    Check(pairs[i],"missing neighbour row.");

    Mat moment(d*d,0);

    for(unsigned p=0;p<count[i];p++){
      const unsigned j=pairs[i][p];

      Check(
          j<n && j!=i && body[i]==body[j],
          "invalid same-body neighbour.");

      Edge e;
      e.j=j;
      e.R={{0,0,0}};
      e.D={{0,0,0}};
      e.raw={{0,0,0}};

      double rr=0;

      for(int k=0;k<d;k++){
        e.R[k]=X[j][k]-X[i][k];
        rr+=e.R[k]*e.R[k];
      }

      Check(rr>0,"duplicate reference particles.");

      const double fac=
          -kernelFac(rr*dp*dp)*dp*unitGradient;

      Check(
          std::isfinite(fac),
          "nonfinite kernel gradient.");

      for(int k=0;k<d;k++)
        e.raw[k]=fac*e.R[k];

      for(int a=0;a<d;a++)
        for(int b=0;b<d;b++){
          moment[a*d+b]+=
              v[j]*e.R[a]*e.raw[b];

        }

      c->rows[i].push_back(e);
    }

    const Mat inv=Inverse(moment,d);


    for(auto& e:c->rows[i])
      for(int b=0;b<d;b++)
        for(int a=0;a<d;a++)
          e.D[b]+=e.raw[a]*inv[a*d+b];
  }

  // Each directed row contributes to both endpoints.
  // Reciprocity of the supplied neighbour list is not assumed.
  for(unsigned i=0;i<n;i++)
    for(const auto& e:c->rows[i])
      for(int k=0;k<d;k++){
        const double a=v[i]*v[e.j]*e.raw[k];
        rawArea[i][k]-=a;
        rawArea[e.j][k]+=a;
      }

  for(const auto& group:groups){
    const auto& ids=group.second;
    std::vector<Face> faces;

    if(o.BoundaryForm==1){
      for(unsigned i:ids)
        for(int k=0;k<d;k++)
          if(std::abs(rawArea[i][k])>1.e-12){
            Face f;
            f.i=i;
            f.normal={{0,0,0}};
            f.normal[k]=rawArea[i][k]>0?1:-1;
            f.area=std::abs(rawArea[i][k]);
            faces.push_back(f);
          }
    }
    else{
      using Key=std::array<long long,3>;
      std::map<Key,std::vector<unsigned>> occupied;

      std::array<Vec,3> frame={
          Vec{{1,0,0}},Vec{{0,1,0}},Vec{{0,0,1}}
      };

      // Recover an approximate lattice orientation for the surface target.
      {
        const auto& row=c->rows[ids.front()];

        double best=1.e300;
        Vec first={{0,0,0}};

        for(const auto& e:row){
          double rr=0;
          for(int k=0;k<d;k++)rr+=e.R[k]*e.R[k];

          if(rr<best){
            best=rr;
            first=e.R;
          }
        }

        if(best>0&&best<1.e299){
          for(int k=0;k<d;k++)
            frame[0][k]=first[k]/std::sqrt(best);

          if(d==2){
            frame[1]={{-frame[0][1],frame[0][0],0}};
          }
          else{
            double nextBest=1.e300;
            Vec second={{0,0,0}};

            for(const auto& e:row){
              double rr=0,pr=0;

              for(int k=0;k<d;k++){
                rr+=e.R[k]*e.R[k];
                pr+=e.R[k]*frame[0][k];
              }

              if(pr*pr<.25*rr&&rr<nextBest){
                nextBest=rr;
                second=e.R;
              }
            }

            if(nextBest<1.e299){
              double pr=0,rr=0;

              for(int k=0;k<d;k++)
                pr+=second[k]*frame[0][k];

              for(int k=0;k<d;k++){
                second[k]-=pr*frame[0][k];
                rr+=second[k]*second[k];
              }

              for(int k=0;k<d;k++)
                frame[1][k]=second[k]/std::sqrt(rr);

              frame[2]={{
                  frame[0][1]*frame[1][2]-frame[0][2]*frame[1][1],
                  frame[0][2]*frame[1][0]-frame[0][0]*frame[1][2],
                  frame[0][0]*frame[1][1]-frame[0][1]*frame[1][0]
              }};
            }
            else{
              frame={
                  Vec{{1,0,0}},Vec{{0,1,0}},Vec{{0,0,1}}
              };
            }
          }
        }
      }

      const auto lattice=[&](unsigned i){
        Vec q={{0,0,0}};

        for(int a=0;a<d;a++)
          for(int k=0;k<d;k++)
            q[a]+=frame[a][k]*X[i][k];

        return q;
      };

      Vec origin=lattice(ids.front());

      for(unsigned i:ids)
        for(int k=0;k<d;k++)
          origin[k]=std::min(origin[k],lattice(i)[k]);

      for(unsigned i:ids){
        Key key={{0,0,0}};

        for(int k=0;k<d;k++){
          const double t=lattice(i)[k]-origin[k];

          Check(
              std::abs(t)<1.e9,
              "reference coordinate range exceeds voxel capacity.");

          key[k]=std::llround(t);

        }

        auto& owners=occupied[key];


        owners.push_back(i);
      }

      for(const auto& cell:occupied)
        for(int k=0;k<d;k++)
          for(int sign:{-1,1}){
            Key next=cell.first;
            next[k]+=sign;

            if(!occupied.count(next)){
              for(unsigned owner:cell.second){
                Face f;
                f.i=owner;
                f.normal={{0,0,0}};

                for(int a=0;a<d;a++)
                  f.normal[a]=sign*frame[k][a];

                f.area=1./cell.second.size();
                faces.push_back(f);
              }
            }
          }
    }

    CorrectArea(ids,X,v,d,faces,target);
  }

  physicalTarget=target;
  for(auto& a:physicalTarget)for(int k=0;k<d;k++)a[k]*=std::pow(dp,d-1);
  for(auto& row:c->rows)for(auto& e:row){
    for(int k=0;k<d;k++){e.R[k]*=dp;e.D[k]/=unitGradient;}
  }
  c->dp=dp;
  return c;
}
void BuildBlend(Cache& c,unsigned n,int d,double dp,const tdouble3* reference,
    const unsigned* body,const float* volume,const std::function<double(double)>& kernelFac){
  std::vector<byte> surface(n,0);

  // Identify potential surface owners from missing short-range directions.
  // Existing particles only; no ghost particles are introduced.
  for(unsigned i=0;i<n;i++)if(!c.rows[i].empty()){
    const Vec xi=Coordinates(reference[i],d);

    for(int ax=-1;ax<=1;ax++)
      for(int ay=-1;ay<=1;ay++)
        for(int az=(d==2?0:-1);az<=(d==2?0:1);az++){

          if(!ax&&!ay&&!az)continue;

          const Vec q={{double(ax),double(ay),double(az)}};
          const double qq=ax*ax+ay*ay+az*az;
          bool covered=false;

          for(const auto& e:c.rows[i]){
            const Vec xj=Coordinates(reference[e.j],d);
            double rr=0,pr=0;

            for(int k=0;k<d;k++){
              const double r=xj[k]-xi[k];
              rr+=r*r;
              pr+=q[k]*r;
            }

            if(rr>0 && rr<=3.24*dp*dp &&
                pr>0 && pr*pr>=.49*qq*rr){
              covered=true;
              break;
            }
          }

          if(!covered)surface[i]=1;
        }
  }

  struct SurfaceNode{unsigned id;int left=-1,right=-1,axis=0;};
  std::vector<SurfaceNode> tree;
  std::vector<Vec> coordinates(n);
  std::map<unsigned,std::vector<unsigned>> surfaceIds;
  for(unsigned i=0;i<n;i++){
    coordinates[i]=Coordinates(reference[i],d);
    if(surface[i])surfaceIds[body[i]].push_back(i);
  }
  std::function<int(std::vector<unsigned>&,size_t,size_t,int)> build;
  build=[&](std::vector<unsigned>& ids,size_t begin,size_t end,int depth)->int{
    if(begin==end)return -1;
    const size_t mid=(begin+end)/2;const int axis=depth%d;
    std::nth_element(ids.begin()+begin,ids.begin()+mid,ids.begin()+end,
        [&](unsigned a,unsigned b){return coordinates[a][axis]<coordinates[b][axis] ||
            (coordinates[a][axis]==coordinates[b][axis]&&a<b);});
    const int node=int(tree.size());SurfaceNode entry;entry.id=ids[mid];entry.axis=axis;tree.push_back(entry);
    const int left=build(ids,begin,mid,depth+1),right=build(ids,mid+1,end,depth+1);
    tree[node].left=left;tree[node].right=right;return node;
  };
  std::map<unsigned,int> roots;
  for(auto& item:surfaceIds)roots[item.first]=build(item.second,0,item.second.size(),0);
  std::function<void(int,const Vec&,double&,unsigned&)> nearest;
  nearest=[&](int node,const Vec& x,double& best,unsigned& owner){
    if(node<0)return;const auto& entry=tree[node];const auto& y=coordinates[entry.id];double rr=0;
    for(int k=0;k<d;k++)rr+=(y[k]-x[k])*(y[k]-x[k]);
    if(rr<best||(rr==best&&entry.id<owner)){best=rr;owner=entry.id;}
    const double delta=x[entry.axis]-y[entry.axis];
    nearest(delta<0?entry.left:entry.right,x,best,owner);
    if(delta*delta<=best)nearest(delta<0?entry.right:entry.left,x,best,owner);
  };

  for(unsigned i=0;i<n;i++){
    if(!c.active[i])continue;

    const Vec xi=Coordinates(reference[i],d);

    const auto add=[&](unsigned j){
      if(!surface[j] || body[j]!=body[i])return;

      const Vec xj=Coordinates(reference[j],d);
      double rr=0;

      for(int k=0;k<d;k++)
        rr+=(xj[k]-xi[k])*(xj[k]-xi[k]);

      // A small positive radius also handles Cubic's Fac(0) singular form.
      const double w=volume[j]*std::abs(
          kernelFac(std::max(rr,1.e-12*dp*dp)));

      if(std::isfinite(w)&&w>0)
        c.blend[i].push_back({j,w});
    };

    add(i);
    for(const auto& e:c.rows[i])add(e.j);

    if(c.blend[i].empty()){
      // Deep selected rows or irregular local support:
      // use the nearest surface owner of the same elastic body.
      double best=1.e300;
      unsigned owner=i;

      const auto root=roots.find(body[i]);
      if(root!=roots.end())nearest(root->second,xi,best,owner);

      c.blend[i].push_back({owner,1});
    }

    double sum=0;
    for(const auto& w:c.blend[i])sum+=w.second;
    for(auto& w:c.blend[i])w.second/=sum;
  }

}
void UpdateMixed(
    Cache& c,const ContactGeometry& contact,
    const tdouble3* position,const unsigned* ridp)
{
  const int d=c.dim;
  const double width=c.dp;
  const double cell=2*c.dp;

  using Key=std::array<long long,3>;

  const auto key=[&](const Vec& x){
    Key k={{0,0,0}};

    for(int a=0;a<d;a++)
      k[a]=static_cast<long long>(std::floor(x[a]/cell));

    return k;
  };

  PrepareRuntime(c);

  if(contact.Kind==1){
    bool same=c.fixedReady&&
        c.fixedSaved.size()==contact.Fixed.size();

    if(same)
      for(size_t i=0;i<contact.Fixed.size();i++){
        const auto a=c.fixedSaved[i],b=contact.Fixed[i];
        if(a.x!=b.x||a.y!=b.y||a.z!=b.z){
          same=false;
          break;
        }
      }

    if(!same){
      c.fixedCells.clear();
      for(const auto& p:contact.Fixed){
        const Vec x=Coordinates(p,d);
        c.fixedCells[key(x)].push_back(x);
      }
      c.fixedSaved=contact.Fixed;
      c.fixedReady=true;
    }
  }

  const auto& cells=c.fixedCells;
  const auto& used=c.used;
  auto& gate=c.gate;
  std::fill(gate.begin(),gate.end(),0.0);

  #ifdef OMP_USE
    #pragma omp parallel for schedule(guided)
  #endif
  for(unsigned i=0;i<c.rows.size();i++){
    const Vec x=Coordinates(position[ridp[i]],d);

    if(!used[i])continue;


    double remaining=1;

    // Smooth union of local proximity gates.
    // It avoids switching the identity of a single nearest boundary particle.
    const auto include=[&](double gap){
      const double t=std::max(0.,std::min(1.,gap/width));
      const double s=1-t*t*(3-2*t);

      remaining*=1-s;
    };

    if(contact.Kind==2){
      const Vec p=Coordinates(contact.Point,d);
      const Vec normal=Coordinates(contact.Normal,d);

      double gap=-.5*c.dp;

      for(int k=0;k<d;k++)
        gap+=(x[k]-p[k])*normal[k];

      include(gap);
    }
    else if(contact.Kind==1){
      const Key base=key(x);

      for(int ax=-1;ax<=1;ax++)
        for(int ay=-1;ay<=1;ay++)
          for(int az=(d==2?0:-1);az<=(d==2?0:1);az++){

            const Key k={{
                base[0]+ax,base[1]+ay,base[2]+az
            }};

            const auto it=cells.find(k);
            if(it==cells.end())continue;

            for(const Vec& p:it->second){
              Vec normal={{0,0,0}};
              double rr=0;

              for(int a=0;a<d;a++){
                normal[a]=x[a]-p[a];
                rr+=normal[a]*normal[a];
              }

              const double r=std::sqrt(rr);
              if(r>=2*c.dp)continue;

              if(r>0)
                for(int a=0;a<d;a++)
                  normal[a]/=r;

              include(r-c.dp);
            }
          }
    }

    gate[i]=1-remaining;
  }

  // Sample the target state without changing the held gradient operator.
  #ifdef OMP_USE
    #pragma omp parallel for schedule(guided)
  #endif
  for(unsigned i=0;i<c.rows.size();i++){
    double eta=0;

    for(const auto& w:c.blend[i])
      eta+=w.second*gate[w.first];

    c.eta[i]=eta;
  }
}

static std::vector<Vec> NodalArea(const Cache& c,const std::vector<std::vector<Vec>>& q){
  std::vector<Vec> a(c.rows.size(),Vec{{0,0,0}});
  for(unsigned i=0;i<c.rows.size();i++)for(size_t e=0;e<c.rows[i].size();e++)
    for(int k=0;k<c.dim;k++){a[i][k]-=q[i][e][k];a[c.rows[i][e].j][k]+=q[i][e][k];}
  return a;
}
static unsigned SolveRegularized(Cache& c,const std::vector<double>& eta,
    std::vector<std::vector<Vec>>& q){
  auto& s=*c.reg;const unsigned n=c.rows.size(),m=s.ids.size();unsigned total=0;
  std::vector<Vec> lambda(n,Vec{{0,0,0}});
  for(int k=0;k<c.dim;k++){
    RegularizedMG::Vector rhs(m);
    for(unsigned a=0;a<m;a++){
      const unsigned i=s.ids[a];const double target=s.fluid[i][k]+eta[i]*(s.solid[i][k]-s.fluid[i][k]);
      rhs[a]=s.baseArea[i][k]-target;
    }
    const double bn=std::sqrt(RegularizedMG::Inner(rhs,rhs));s.rhsNorm[k]=std::max(s.rhsNorm[k],bn);
    auto& x=s.lambda[k];if(x.size()!=m)x.assign(m,0);
    auto r=s.matrix.Apply(x);for(unsigned a=0;a<m;a++)r[a]=rhs[a]-r[a];
    const double tolerance=1.e-4*std::max(1.e-15,s.rhsNorm[k]);unsigned it=0;
    if(std::sqrt(RegularizedMG::Inner(r,r))>tolerance){
      auto z=s.solver->Apply(r),p=z;double rz=RegularizedMG::Inner(r,z);
      for(;it<c.maxIterations;it++){
        const auto ap=s.matrix.Apply(p);const double pap=RegularizedMG::Inner(p,ap);
        Check(pap>0 && rz>0 && std::isfinite(pap+rz),"regularized CG breakdown");const double step=rz/pap;
        for(unsigned a=0;a<m;a++){x[a]+=step*p[a];r[a]-=step*ap[a];}
        if(std::sqrt(RegularizedMG::Inner(r,r))<=tolerance){it++;break;}
        z=s.solver->Apply(r);const double next=RegularizedMG::Inner(r,z);
        for(unsigned a=0;a<m;a++)p[a]=z[a]+next/rz*p[a];rz=next;
      }
      auto residual=s.matrix.Apply(x);for(unsigned a=0;a<m;a++)residual[a]-=rhs[a];
      Check(std::sqrt(RegularizedMG::Inner(residual,residual))<=2*tolerance,"initial projection reached maxiterations before convergence");
    }
    total+=it;for(unsigned a=0;a<m;a++)lambda[s.ids[a]][k]=x[a];
  }
  q=s.particular;
  for(unsigned i=0;i<n;i++)if(s.active[i]){
    for(int k=0;k<c.dim;k++){
      Vec moment={{0,0,0}};
      for(const auto& e:c.rows[i])for(int a=0;a<c.dim;a++)
        moment[a]+=e.R[a]/c.dp*(lambda[e.j][k]-lambda[i][k]);
      Vec coef={{0,0,0}};
      for(int a=0;a<c.dim;a++)for(int t=0;t<c.dim;t++)coef[a]+=s.inverse[i][a*c.dim+t]*moment[t];
      for(size_t e=0;e<c.rows[i].size();e++){
        double correction=lambda[c.rows[i][e].j][k]-lambda[i][k];
        for(int a=0;a<c.dim;a++)correction-=c.rows[i][e].R[a]/c.dp*coef[a];
        q[i][e][k]-=c.inwardWeight[i]*correction;
      }
    }
  }
  return total;
}
// Both anchors reproduce affine fields. Blend them before the area solve,
// then use the same inward mobility in the Schur matrix and its adjoint.
// A post-solve taper would invalidate the area constraint being solved.
static void BuildInitialSystem(Cache& c,const tdouble3* reference){
  auto& s=*c.reg;const unsigned n=c.rows.size();const int dim=c.dim;
  s.active=c.active;s.inverse.resize(n);s.particular.resize(n);
  std::vector<std::vector<Vec>> R(n);std::vector<Mat>& inverse=s.inverse;
  std::vector<byte> used(n,0);
  for(unsigned i=0;i<n;i++){
    s.particular[i].resize(c.rows[i].size());
    Mat moment(dim*dim,0);R[i].resize(c.rows[i].size());
    for(size_t e=0;e<c.rows[i].size();e++){
      const auto& edge=c.rows[i][e];R[i][e]=edge.R;
      for(int a=0;a<dim;a++)R[i][e][a]/=c.dp;
      for(int a=0;a<dim;a++)for(int k=0;k<dim;k++)moment[a*dim+k]+=R[i][e][a]*R[i][e][k];
      for(int k=0;k<dim;k++)s.particular[i][e][k]=c.volume[i]*c.volume[edge.j]*edge.D[k];
      if(s.active[i]){used[i]=1;used[edge.j]=1;}
    }
    if(s.active[i]&&!c.rows[i].empty()){
      inverse[i]=Inverse(moment,dim);
      for(size_t e=0;e<c.rows[i].size();e++)for(int k=0;k<dim;k++){
        double q=0;for(int a=0;a<dim;a++)q+=R[i][e][a]*inverse[i][a*dim+k];
        const double weight=c.inwardWeight[i];
        s.particular[i][e][k]=weight*c.volume[i]/c.dp*q
            +(1.-weight)*s.particular[i][e][k];
      }
    }
  }
  for(unsigned i=0;i<n;i++)if(used[i])s.ids.push_back(i);
  const unsigned m=s.ids.size();
  Check(m>0,"no active geometry correction rows");
  std::vector<double> diagonal(m,0);
  std::vector<unsigned> index(n,~0u);for(unsigned a=0;a<m;a++)index[s.ids[a]]=a;
  std::vector<std::map<unsigned,double>> assembled(m);
  for(unsigned i=0;i<n;i++)if(s.active[i]){
    const unsigned degree=c.rows[i].size(),count=degree+1;
    std::vector<unsigned> nodes(count);std::vector<Vec> C(count,Vec{{0,0,0}});
    nodes[0]=index[i];
    for(unsigned e=0;e<degree;e++){
      nodes[e+1]=index[c.rows[i][e].j];C[e+1]=R[i][e];
      for(int k=0;k<dim;k++)C[0][k]-=R[i][e][k];
    }
    for(unsigned a=0;a<count;a++){
      Vec coef={{0,0,0}};for(int k=0;k<dim;k++)for(int t=0;t<dim;t++)coef[k]+=inverse[i][k*dim+t]*C[a][t];
      for(unsigned j=0;j<count;j++){
        double v=(a==0&&j==0)?double(degree):((a==0||j==0)?-1.:(a==j?1.:0.));
        for(int k=0;k<dim;k++)v-=coef[k]*C[j][k];
        assembled[nodes[a]][nodes[j]]+=c.inwardWeight[i]*v;
        if(a==j)diagonal[nodes[a]]+=v;
      }
    }
  }
  RegularizedMG::Sparse current(m,m);
  for(unsigned a=0;a<m;a++){
    Check(diagonal[a]>0 && std::isfinite(diagonal[a]),"nonpositive geometry diagonal");
    assembled[a][a]+=1.e-7*diagonal[a];current.row[a].reserve(assembled[a].size());
    for(const auto& e:assembled[a])current.row[a].push_back(e);
    assembled[a].clear();
  }

  // Retain the unweighted reference scale for the soft area penalty.
  // Scaling this penalty by mobility would undo the taper near its inner end.
  s.referenceDiagonal=diagonal;
  s.matrix=std::move(current);s.matrix.Pack();
  std::vector<RegularizedMG::Point> coords(m);
  for(unsigned a=0;a<m;a++)coords[a]=Coordinates(reference[s.ids[a]],dim);
  s.solver=std::make_shared<RegularizedMG::Solver>(s.matrix,coords,dim);
  s.baseArea=NodalArea(c,s.particular);
}
namespace{
using Apply=std::function<Array(const Array&)>;

double Length(const Array& x){
  return std::sqrt(Dot(x,x));
}

// Minimum-norm least squares.
Array LeastSquares(
    const Apply& forward,const Apply& adjoint,
    Array rhs,unsigned maxit)
{
  Array x=adjoint(rhs);
  std::fill(x.begin(),x.end(),0);

  const double normal0=Length(adjoint(rhs));
  double beta=Length(rhs);

  if(beta==0||normal0<1.e-14)return x;

  Array u=rhs;
  for(double& v:u)v/=beta;

  Array z=adjoint(u);
  double alpha=Length(z);

  if(alpha==0)return x;
  for(double& v:z)v/=alpha;

  Array direction=z;
  double rhobar=alpha,phibar=beta;

  for(unsigned it=0;it<maxit;it++){
    Array next=forward(z);

    for(size_t k=0;k<u.size();k++)
      next[k]-=alpha*u[k];

    u.swap(next);
    beta=Length(u);

    if(beta>0)
      for(double& v:u)v/=beta;

    next=adjoint(u);

    for(size_t k=0;k<z.size();k++)
      next[k]-=beta*z[k];

    z.swap(next);
    alpha=Length(z);

    if(alpha>0)
      for(double& v:z)v/=alpha;

    const double rho=std::hypot(rhobar,beta);
    if(!(rho>0&&std::isfinite(rho)))break;

    const double cs=rhobar/rho;
    const double sn=beta/rho;
    const double theta=sn*alpha;

    rhobar=-cs*alpha;

    const double phi=cs*phibar;
    phibar*=sn;

    for(size_t k=0;k<x.size();k++){
      x[k]+=phi/rho*direction[k];
      direction[k]=z[k]-theta/rho*direction[k];
    }

    if(it%10==9||alpha==0||beta==0){
      Array residual=forward(x);

      for(size_t k=0;k<rhs.size();k++)
        residual[k]-=rhs[k];

      if(Length(adjoint(residual))<=1.e-9*normal0)
        break;
    }
  }

  return x;
}

// Symmetric eigensystem. Eigenvectors are stored by columns.
void SymmetricEigen(
    Mat a,int n,Array& values,Mat& vectors)
{
  vectors.assign(n*n,0);
  for(int i=0;i<n;i++)vectors[i*n+i]=1;

  double scale=0;
  for(double v:a)scale=std::max(scale,std::abs(v));

  for(unsigned sweep=0;sweep<60;sweep++){
    double off=0;

    for(int p=0;p<n;p++)
      for(int q=p+1;q<n;q++){
        const double b=a[p*n+q];
        off=std::max(off,std::abs(b));

        if(std::abs(b)<=1.e-14*scale)continue;

        const double tau=
            (a[q*n+q]-a[p*n+p])/(2*b);

        const double t=std::copysign(1.,tau)/
            (std::abs(tau)+std::hypot(1.,tau));

        const double c=1/std::sqrt(1+t*t);
        const double s=t*c;

        a[p*n+p]-=t*b;
        a[q*n+q]+=t*b;
        a[p*n+q]=a[q*n+p]=0;

        for(int k=0;k<n;k++)if(k!=p&&k!=q){
          const double ap=a[k*n+p];
          const double aq=a[k*n+q];

          a[k*n+p]=a[p*n+k]=c*ap-s*aq;
          a[k*n+q]=a[q*n+k]=s*ap+c*aq;
        }

        for(int k=0;k<n;k++){
          const double vp=vectors[k*n+p];
          const double vq=vectors[k*n+q];

          vectors[k*n+p]=c*vp-s*vq;
          vectors[k*n+q]=s*vp+c*vq;
        }
      }

    if(off<=1.e-13*scale)break;
  }

  values.resize(n);
  for(int i=0;i<n;i++)values[i]=a[i*n+i];
}

struct QuadraticGeometry{
  int d,q,m;
  double h,dp;

  std::vector<Vec> X;
  Array depth;

  std::vector<std::pair<int,int>> powers;
  std::vector<std::vector<unsigned>> near;
};

QuadraticGeometry MakeQuadraticGeometry(
    int d,double dp,double h,unsigned n,
    const tdouble3* reference,const unsigned* body)
{
  QuadraticGeometry g;
  g.d=d;
  g.dp=dp;
  g.h=h;

  for(int a=0;a<d;a++)
    for(int b=a;b<d;b++)
      g.powers.push_back({a,b});

  g.q=g.powers.size();
  g.m=d*(d+g.q);
  g.X.resize(n);
  g.near.resize(n);

  using Key=std::array<long long,3>;
  const double radius=2.01*dp;

  const auto key=[&](const Vec& x){
    Key k={{0,0,0}};

    for(int a=0;a<d;a++)
      k[a]=std::floor(x[a]/radius);

    return k;
  };

  std::map<Key,std::vector<unsigned>> cells;

  for(unsigned i=0;i<n;i++){
    g.X[i]=Coordinates(reference[i],d);
    cells[key(g.X[i])].push_back(i);
  }

  for(unsigned i=0;i<n;i++){
    const Key base=key(g.X[i]);

    for(int a=-1;a<=1;a++)
      for(int b=-1;b<=1;b++)
        for(int c=(d==2?0:-1);c<=(d==2?0:1);c++){
          const Key k={{
              base[0]+a,base[1]+b,base[2]+c
          }};

          auto it=cells.find(k);
          if(it==cells.end())continue;

          for(unsigned j:it->second)
            if(body[j]==body[i]){
              double rr=0;

              for(int k=0;k<d;k++)
                rr+=(g.X[j][k]-g.X[i][k])*
                    (g.X[j][k]-g.X[i][k]);

              if(rr<=radius*radius)
                g.near[i].push_back(j);
            }
        }
  }

  // Approximate surface distance is used only for peer selection.
  std::vector<byte> surface(n,0);

  for(unsigned i=0;i<n;i++){
    bool exposed=false;

    for(int a=-1;a<=1;a++)
      for(int b=-1;b<=1;b++)
        for(int c=(d==2?0:-1);c<=(d==2?0:1);c++){
          if(!a&&!b&&!c)continue;

          const Vec q={{
              double(a),double(b),double(c)
          }};
          const double qq=a*a+b*b+c*c;
          bool covered=false;

          for(unsigned j:g.near[i])if(i!=j){
            double rr=0,pr=0;

            for(int k=0;k<d;k++){
              const double r=g.X[j][k]-g.X[i][k];
              rr+=r*r;
              pr+=q[k]*r;
            }

            if(rr<=3.24*dp*dp&&pr>0&&
                pr*pr>=.25*qq*rr){
              covered=true;
              break;
            }
          }

          if(!covered)exposed=true;
        }

    if(exposed)surface[i]=1;
  }

  struct SurfaceNode{unsigned id;int left=-1,right=-1,axis=0;};
  std::vector<SurfaceNode> tree;
  std::vector<Vec> coordinates(n);
  std::map<unsigned,std::vector<unsigned>> surfaceIds;
  for(unsigned i=0;i<n;i++){
    coordinates[i]=Coordinates(reference[i],d);
    if(surface[i])surfaceIds[body[i]].push_back(i);
  }
  std::function<int(std::vector<unsigned>&,size_t,size_t,int)> build;
  build=[&](std::vector<unsigned>& ids,size_t begin,size_t end,int depth)->int {
    if(begin==end)return -1;
    const size_t mid=(begin+end)/2;const int axis=depth%d;
    std::nth_element(ids.begin()+begin,ids.begin()+mid,ids.begin()+end,
        [&](unsigned a,unsigned b){return coordinates[a][axis]<coordinates[b][axis] ||
            (coordinates[a][axis]==coordinates[b][axis]&&a<b);});
    const int node=int(tree.size());SurfaceNode entry;entry.id=ids[mid];entry.axis=axis;tree.push_back(entry);
    const int left=build(ids,begin,mid,depth+1),right=build(ids,mid+1,end,depth+1);
    tree[node].left=left;tree[node].right=right;return node;
  };
  std::map<unsigned,int> roots;
  for(auto& item:surfaceIds)roots[item.first]=build(item.second,0,item.second.size(),0);
  std::function<void(int,const Vec&,double&,unsigned&)> nearest;
  nearest=[&](int node,const Vec& x,double& best,unsigned& owner){
    if(node<0)return;const auto& entry=tree[node];const auto& y=coordinates[entry.id];double rr=0;
    for(int k=0;k<d;k++)rr+=(y[k]-x[k])*(y[k]-x[k]);
    if(rr<best||(rr==best&&entry.id<owner)){best=rr;owner=entry.id;}
    const double delta=x[entry.axis]-y[entry.axis];
    nearest(delta<0?entry.left:entry.right,x,best,owner);
    if(delta*delta<=best)nearest(delta<0?entry.right:entry.left,x,best,owner);
  };

  g.depth.assign(n,0);

  for(unsigned i=0;i<n;i++){
    double best=
        std::numeric_limits<double>::infinity();

    unsigned owner=i;
    const auto root=roots.find(body[i]);
    if(root!=roots.end())nearest(root->second,g.X[i],best,owner);

    if(std::isfinite(best))
      g.depth[i]=std::sqrt(best);
  }

  return g;
}

Array QuadraticMoments(
    const Cache& c,const QuadraticGeometry& g)
{
  Array M(c.rows.size()*g.q*g.d,0);

  for(unsigned i=0;i<c.rows.size();i++)
    for(const auto& e:c.rows[i])
      for(int p=0;p<g.q;p++){
        const int a=g.powers[p].first;
        const int b=g.powers[p].second;

        const double q=
            (g.X[e.j][a]-g.X[i][a])*
            (g.X[e.j][b]-g.X[i][b])/g.h;

        for(int k=0;k<g.d;k++)
          M[(i*g.q+p)*g.d+k]+=
              c.volume[e.j]*q*e.D[k];
      }

  return M;
}

// Fixed nu=.33 reference metric, consistent with the earlier diagnostics.
// This score is not the actual material's effective modulus loss.
std::pair<double,double> QuadraticScore(
    const Cache& c,const QuadraticGeometry& g,
    const Array& M,unsigned center)
{
  const int d=g.d;
  const int m=g.m;
  const int stride=d+g.q;

  Mat exact(m*m,0),discrete(m*m,0);

  const double lambda=.33/(1.33*.34);
  const double mu=.5/1.33;

  for(unsigned j:g.near[center]){
    Array grad(m*d*d,0),trial(m*d*d,0);
    Vec y={{0,0,0}};

    for(int k=0;k<d;k++)
      y[k]=(g.X[j][k]-g.X[center][k])/g.h;

    for(int a=0;a<d;a++){
      for(int k=0;k<d;k++)
        grad[((a*stride+k)*d+a)*d+k]=1;

      for(int p=0;p<g.q;p++){
        const int b=g.powers[p].first;
        const int k=g.powers[p].second;
        const int t=a*stride+d+p;

        if(b==k)
          grad[(t*d+a)*d+b]=y[b];
        else{
          grad[(t*d+a)*d+b]=y[k];
          grad[(t*d+a)*d+k]=y[b];
        }
      }
    }

    trial=grad;

    for(int a=0;a<d;a++)
      for(int p=0;p<g.q;p++)
        for(int k=0;k<d;k++)
          trial[((a*stride+d+p)*d+a)*d+k]+=
              (g.powers[p].first==g.powers[p].second
                  ? .5 : 1.)*
              M[(j*g.q+p)*d+k];

    const auto accumulate=[&](const Array& G,Mat& K){
      Array strain(m*d*d),trace(m,0);

      for(int a=0;a<m;a++)
        for(int b=0;b<d;b++)
          for(int k=0;k<d;k++){
            strain[(a*d+b)*d+k]=.5*(
                G[(a*d+b)*d+k]+
                G[(a*d+k)*d+b]);

            if(b==k)
              trace[a]+=strain[(a*d+b)*d+k];
          }

      for(int a=0;a<m;a++)
        for(int b=0;b<m;b++){
          double v=lambda*trace[a]*trace[b];

          for(int k=0;k<d*d;k++)
            v+=2*mu*
                strain[a*d*d+k]*strain[b*d*d+k];

          K[a*m+b]+=c.volume[j]*v;
        }
    };

    accumulate(grad,exact);
    accumulate(trial,discrete);
  }

  Array ev;
  Mat U;

  SymmetricEigen(exact,m,ev,U);

  const double largest=
      *std::max_element(ev.begin(),ev.end());

  std::vector<int> valid;

  for(int k=0;k<m;k++)
    if(ev[k]>largest*1.e-10)
      valid.push_back(k);

  const int r=valid.size();
  if(!r)return {0,0};

  Mat whitened(r*r,0);

  for(int a=0;a<r;a++)
    for(int b=0;b<r;b++){
      double sum=0;

      for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
          sum+=U[i*m+valid[a]]*
              discrete[i*m+j]*
              U[j*m+valid[b]];

      whitened[a*r+b]=
          sum/std::sqrt(ev[valid[a]]*ev[valid[b]]);
    }

  SymmetricEigen(whitened,r,ev,U);

  double worst=0,mean=0;

  for(double v:ev){
    const double loss=std::max(0.,1-v);
    worst=std::max(worst,loss);
    mean+=loss;
  }

  return {worst,mean/r};
}

// Reduce full second moments while preserving affine moments and nodal areas.
bool CorrectQuadraticPatch(
    Cache& c,const QuadraticGeometry& g,
    const std::vector<unsigned>& ids)
{
  struct Ref{
    unsigned i,e;
  };

  std::vector<Ref> edges;

  for(unsigned i:ids)
    for(unsigned e=0;e<c.rows[i].size();e++)
      edges.push_back({i,e});

  const size_t ne=edges.size();
  if(!ne)return false;

  std::map<unsigned,unsigned> touched;

  for(const auto& r:edges){
    touched.emplace(r.i,0);
    touched.emplace(c.rows[r.i][r.e].j,0);
  }

  const size_t nc=ids.size()*g.d+touched.size();

  // Skip an oversized patch instead of allocating unbounded workspace.
  if(ne*nc>32000000)return false;

  std::vector<Array> basis;

  const auto insert=[&](Array row){
    const double initial=Length(row);
    if(initial==0)return;

    for(double& x:row)x/=initial;

    for(int pass=0;pass<2;pass++)
      for(const auto& b:basis){
        const double v=Dot(row,b);

        for(size_t k=0;k<ne;k++)
          row[k]-=v*b[k];
      }

    const double norm=Length(row);
    if(norm<1.e-10)return;

    for(double& x:row)x/=norm;
    basis.push_back(std::move(row));
  };

  for(unsigned i:ids)
    for(int k=0;k<g.d;k++){
      Array row(ne,0);

      for(size_t e=0;e<ne;e++)
        if(edges[e].i==i){
          const unsigned j=c.rows[i][edges[e].e].j;

          row[e]=std::sqrt(c.inwardWeight[i])*
              (g.X[j][k]-g.X[i][k])/g.h;
        }

      insert(std::move(row));
    }

  const double unitV=std::pow(g.dp,g.d);

  for(const auto& item:touched){
    Array row(ne,0);

    for(size_t e=0;e<ne;e++){
      const unsigned i=edges[e].i;
      const unsigned j=c.rows[i][edges[e].e].j;

      row[e]=std::sqrt(c.inwardWeight[i])*c.volume[i]/unitV*
          ((j==item.first?1.:0)-
           (i==item.first?1.:0));
    }

    insert(std::move(row));
  }

  const auto project=[&](Array z){
    for(int pass=0;pass<2;pass++)
      for(const auto& b:basis){
        const double v=Dot(z,b);

        for(size_t k=0;k<ne;k++)
          z[k]-=v*b[k];
      }

    return z;
  };

  std::vector<unsigned> local(c.rows.size(),0);

  for(unsigned k=0;k<ids.size();k++)
    local[ids[k]]=k;

  std::vector<Array> q(ne,Array(g.q));

  for(size_t e=0;e<ne;e++){
    const unsigned i=edges[e].i;
    const unsigned j=c.rows[i][edges[e].e].j;

    for(int p=0;p<g.q;p++){
      const int a=g.powers[p].first;
      const int b=g.powers[p].second;

      q[e][p]=std::sqrt(c.inwardWeight[i])*
          std::sqrt(c.volume[i]/unitV)*
          (a==b?1.:std::sqrt(2.))*
          (g.X[j][a]-g.X[i][a])*
          (g.X[j][b]-g.X[i][b])/(g.h*g.h);
    }
  }

  const auto Q=[&](const Array& z){
    Array y(ids.size()*g.q,0);

    for(size_t e=0;e<ne;e++)
      for(int p=0;p<g.q;p++)
        y[local[edges[e].i]*g.q+p]+=q[e][p]*z[e];

    return y;
  };

  const Apply forward=[&](const Array& z){
    return Q(project(z));
  };

  const Apply adjoint=[&](const Array& y){
    Array z(ne,0);

    for(size_t e=0;e<ne;e++)
      for(int p=0;p<g.q;p++)
        z[e]+=q[e][p]*y[local[edges[e].i]*g.q+p];

    return project(z);
  };

  std::vector<Array> increment(g.d);

  for(int k=0;k<g.d;k++){
    // Evaluate the current quadratic moment directly. The inward mobility
    // cancels analytically; do not divide by a weight that can approach zero.
    Array rhs(ids.size()*g.q,0);
    for(size_t e=0;e<ne;e++){
      const auto& ref=edges[e];
      const auto& edge=c.rows[ref.i][ref.e];
      const double scale=std::sqrt(c.volume[ref.i]/unitV)*
          c.volume[edge.j]*edge.D[k]/g.h;
      for(int p=0;p<g.q;p++){
        const int a=g.powers[p].first,b=g.powers[p].second;
        rhs[local[ref.i]*g.q+p]+=scale*(a==b?1.:std::sqrt(2.))*
            (g.X[edge.j][a]-g.X[ref.i][a])*
            (g.X[edge.j][b]-g.X[ref.i][b]);
      }
    }

    const double before=Length(rhs);

    for(double& x:rhs)x=-x;

    increment[k]=project(
        LeastSquares(forward,adjoint,rhs,c.maxIterations));

    Array after=Q(increment[k]);

    for(size_t a=0;a<after.size();a++)
      after[a]-=rhs[a];

    if(!std::isfinite(Length(after))||
        Length(after)>before*(1+1.e-8))
      return false;

    for(const auto& b:basis)
      if(std::abs(Dot(b,increment[k]))>
          1.e-8*(1+Length(increment[k])))
        return false;
  }

  // Audit the original constraints, not just the orthogonal basis.
  for(int k=0;k<g.d;k++){
    Array area(c.rows.size(),0);
    Array moment(ids.size()*g.d,0);

    for(size_t e=0;e<ne;e++){
      const unsigned i=edges[e].i;
      const unsigned j=c.rows[i][edges[e].e].j;
      const double z=std::sqrt(c.inwardWeight[i])*increment[k][e];

      area[i]-=c.volume[i]/unitV*z;
      area[j]+=c.volume[i]/unitV*z;

      for(int a=0;a<g.d;a++)
        moment[local[i]*g.d+a]+=
            (g.X[j][a]-g.X[i][a])/g.h*z;
    }

    const double tolerance=
        1.e-8*(1+Length(increment[k]));

    for(double v:area)
      if(std::abs(v)>tolerance)return false;

    for(double v:moment)
      if(std::abs(v)>tolerance)return false;
  }

  for(size_t e=0;e<ne;e++){
    const auto& r=edges[e];
    auto& edge=c.rows[r.i][r.e];

    for(int k=0;k<g.d;k++)
      edge.D[k]+=
          std::sqrt(c.inwardWeight[r.i])*increment[k][e]/
          (c.volume[edge.j]*g.h);
  }

  return true;
}

void ApplyQuadratic(
    const Options& o,bool twoD,double dp,unsigned n,
    const tdouble3* reference,const unsigned* body,
    const byte* selected,
    std::vector<std::shared_ptr<Cache>> templates,
    Diagnostics& report)
{
  if(o.QuadraticError==0)return;

  if(templates.size()==2&&templates[0]==templates[1])
    templates.resize(1);

  const auto g=MakeQuadraticGeometry(
      twoD?2:3,dp,o.KernelH,n,reference,body);

  Array loss(n,0);
  std::vector<byte> scored(n,0);
  for(unsigned i=0;i<n;i++)if(selected[i]){
    scored[i]=1;for(unsigned j:g.near[i])scored[j]=1;
  }

  for(const auto& c:templates){
    const Array M=QuadraticMoments(*c,g);

    for(unsigned i=0;i<n;i++)
      if(scored[i]&&!c->rows[i].empty())
        loss[i]=std::max(
            loss[i],QuadraticScore(*c,g,M,i).first);
  }

  report.QuadraticAboveThreshold=0;
  for(unsigned i=0;i<n;i++)
    if(selected[i]&&loss[i]>o.QuadraticError)
      report.QuadraticAboveThreshold++;
  std::vector<unsigned> candidates;

  for(unsigned i=0;i<n;i++)
    if(selected[i]&&loss[i]>o.QuadraticError){
      Array peer;
      bool maximum=true;

      for(unsigned j:g.near[i])
        if(loss[j]>loss[i]+1.e-12||
            (std::abs(loss[j]-loss[i])<=1.e-12&&j<i))
          maximum=false;

      if(!maximum)continue;

      for(unsigned j=0;j<n;j++)
        if(body[j]==body[i]&&selected[j]&&
            std::abs(g.depth[j]-g.depth[i])<=dp){
          double rr=0;

          for(int k=0;k<g.d;k++)
            rr+=(g.X[j][k]-g.X[i][k])*
                (g.X[j][k]-g.X[i][k]);

          if(rr>std::pow(2.01*dp,2)&&
              rr<=std::pow(2.01*dp+2*o.KernelH,2))
            peer.push_back(loss[j]);
        }

      if(peer.empty())continue;

      std::sort(peer.begin(),peer.end());

      if(loss[i]-peer[peer.size()/2]>o.QuadraticError)
        candidates.push_back(i);
    }

  std::sort(
      candidates.begin(),candidates.end(),
      [&](unsigned a,unsigned b){
        return loss[a]>loss[b]||
            (loss[a]==loss[b]&&a<b);
      });

  std::vector<unsigned> seeds;
  const double radius=2.01*dp+2*o.KernelH;

  for(unsigned seed:candidates){
    bool overlap=false;

    for(unsigned j:seeds)if(body[j]==body[seed]){
      double rr=0;

      for(int k=0;k<g.d;k++)
        rr+=(g.X[j][k]-g.X[seed][k])*
            (g.X[j][k]-g.X[seed][k]);

      if(rr<4*radius*radius)overlap=true;
    }

    if(overlap)continue;

    seeds.push_back(seed);
    report.QuadraticTried++;

    std::vector<unsigned> ids,watch;

    for(unsigned j=0;j<n;j++)
      if(body[j]==body[seed]){
        double rr=0;

        for(int k=0;k<g.d;k++)
          rr+=(g.X[j][k]-g.X[seed][k])*
              (g.X[j][k]-g.X[seed][k]);

        if(rr<=radius*radius&&templates[0]->active[j])
          ids.push_back(j);

        if(rr<=std::pow(radius+2.01*dp,2)&&
            !templates[0]->rows[j].empty())
          watch.push_back(j);
      }

    if(ids.empty())continue;

    // The common increment minimizes the joint endpoint objective.
    Cache average=*templates[0];

    for(unsigned i:ids)
      for(unsigned e=0;e<average.rows[i].size();e++)
        for(int k=0;k<g.d;k++){
          double value=0;

          for(const auto& c:templates)
            value+=c->rows[i][e].D[k];

          average.rows[i][e].D[k]=value/templates.size();
        }

    const auto initial=average.rows;

    if(!CorrectQuadraticPatch(average,g,ids))
      continue;

    bool accept=true;
    std::vector<std::vector<std::vector<Edge>>> old;

    for(const auto& c:templates){
      old.push_back(c->rows);

      const Array beforeM=QuadraticMoments(*c,g);
      double beforeMax=0,beforeMean=0;

      for(unsigned j:watch){
        const auto s=QuadraticScore(*c,g,beforeM,j);
        beforeMax=std::max(beforeMax,s.first);
        beforeMean+=s.second;
      }

      for(unsigned i:ids)
        for(unsigned e=0;e<c->rows[i].size();e++)
          for(int k=0;k<g.d;k++)
            c->rows[i][e].D[k]+=
                average.rows[i][e].D[k]-
                initial[i][e].D[k];

      for(unsigned i:ids){
        Vec sum={{0,0,0}};double norm=0;
        for(const auto& e:c->rows[i])for(int k=0;k<g.d;k++){
          const double value=c->volume[e.j]*e.D[k];
          norm+=value*value;sum[k]+=value;
        }
        for(int k=0;k<g.d;k++)norm+=sum[k]*sum[k];
        accept=accept&&std::isfinite(norm)&&norm<=4*c->referenceNorm2[i];
      }

      const Array afterM=QuadraticMoments(*c,g);
      double afterMax=0,afterMean=0;

      for(unsigned j:watch){
        const auto s=QuadraticScore(*c,g,afterM,j);
        afterMax=std::max(afterMax,s.first);
        afterMean+=s.second;
      }

      accept=accept&&
          afterMax<=beforeMax+1.e-9&&
          afterMean<beforeMean-1.e-10;
    }

    if(accept)
      report.QuadraticAccepted++;
    else
      for(size_t k=0;k<templates.size();k++)
        templates[k]->rows.swap(old[k]);
  }
}

}

static void FinishInitialProjection(Cache& c){
  auto& s=*c.reg;std::vector<std::vector<Vec>> q;
  const unsigned iterations=SolveRegularized(c,c.eta,q);
  const auto area=NodalArea(c,q);double rr=0,tt=0,affine=0;
  for(unsigned i=0;i<c.rows.size();i++){
    Vec sum={{0,0,0}};double norm=0;Mat moment(c.dim*c.dim,0);
    for(size_t e=0;e<c.rows[i].size();e++)for(int k=0;k<c.dim;k++){
      auto& edge=c.rows[i][e];edge.D[k]=q[i][e][k]/(c.volume[i]*c.volume[edge.j]);
      const double g=q[i][e][k]/c.volume[i];norm+=g*g;sum[k]+=g;
      for(int a=0;a<c.dim;a++)moment[a*c.dim+k]+=edge.R[a]*g;
    }
    if(!c.rows[i].empty())for(int a=0;a<c.dim;a++)for(int k=0;k<c.dim;k++)
      affine=std::max(affine,std::abs(moment[a*c.dim+k]-(a==k?1.:0.)));
    for(int k=0;k<c.dim;k++){
      norm+=sum[k]*sum[k];const double target=s.fluid[i][k]+c.eta[i]*(s.solid[i][k]-s.fluid[i][k]);
      rr+=(area[i][k]-target)*(area[i][k]-target);tt+=target*target;
    }
    c.referenceNorm2[i]=norm;
  }
  Check(std::isfinite(rr)&&affine<1.e-7,"invalid regularized initial geometry");
  s.initialResidual=std::sqrt(rr/std::max(tt,1.e-30));c.localReady=true;
  std::fprintf(stderr,"Mode 5 initialized: iterations=%u area=%.9g affine=%.9g\n",iterations,s.initialResidual,affine);
  if(c.quadraticError>0){
    Options options;options.QuadraticError=c.quadraticError;
    options.KernelH=c.kernelH;options.MaxIterations=c.maxIterations;
    Diagnostics report;
    // The alias is borrowed only for this synchronous initialization call.
    std::shared_ptr<Cache> alias(&c,[](Cache*){});
    ApplyQuadratic(options,c.dim==2,c.dp,unsigned(c.rows.size()),
        c.quadraticReference.data(),c.quadraticBody.data(),
        c.quadraticSelected.data(),{alias},report);
    std::vector<Vec> checkedArea(c.rows.size(),Vec{{0,0,0}});
    double affineChange=0,areaChange=0;
    for(unsigned i=0;i<c.rows.size();i++){
      Mat moment(c.dim*c.dim,0);
      for(const auto& edge:c.rows[i])for(int k=0;k<c.dim;k++){
        const double flux=c.volume[i]*c.volume[edge.j]*edge.D[k];
        checkedArea[i][k]-=flux;checkedArea[edge.j][k]+=flux;
        for(int a=0;a<c.dim;a++)moment[a*c.dim+k]+=edge.R[a]*flux/c.volume[i];
      }
      if(!c.rows[i].empty())for(int a=0;a<c.dim;a++)for(int k=0;k<c.dim;k++)
        affineChange=std::max(affineChange,std::abs(moment[a*c.dim+k]-(a==k?1.:0.)));
    }
    for(unsigned i=0;i<c.rows.size();i++)for(int k=0;k<c.dim;k++){
      const double delta=checkedArea[i][k]-area[i][k];areaChange+=delta*delta;
    }
    if(!std::isfinite(areaChange)||!std::isfinite(affineChange)||
        affineChange>1.e-7||std::sqrt(areaChange)>1.e-7*std::max(std::sqrt(tt),std::pow(c.dp,c.dim-1))){
      for(unsigned i=0;i<c.rows.size();i++)for(size_t e=0;e<c.rows[i].size();e++)
        for(int k=0;k<c.dim;k++)c.rows[i][e].D[k]=q[i][e][k]/(c.volume[i]*c.volume[c.rows[i][e].j]);
      report.QuadraticAccepted=0;
      std::fprintf(stderr,"Mode 5 quadratic: optional pass rolled back after constraint audit.\n");
    }
    std::fprintf(stderr,"Mode 5 quadratic: threshold=%.9g above=%u attempted=%u accepted=%u\n",
        c.quadraticError,report.QuadraticAboveThreshold,report.QuadraticTried,report.QuadraticAccepted);
  }
  // Keep the pre-patch row norm as the dynamic amplification reference.
  std::vector<tdouble3>().swap(c.quadraticReference);
  std::vector<unsigned>().swap(c.quadraticBody);
  std::vector<byte>().swap(c.quadraticSelected);
  // Initial Krylov workspace is not needed by event projections.
  s.solver.reset();s.matrix=RegularizedMG::Sparse();s.particular.clear();s.inverse.clear();s.baseArea.clear();
  for(auto& x:s.lambda)x.clear();
}
static unsigned ThreadIndex(){
#ifdef OMP_USE
  return unsigned(omp_get_thread_num());
#else
  return 0;
#endif
}
// Preserve current deformation inside the area-increment projection.
// The inward mobility must match in assembly, matrix-free audit and writeback;
// it changes the admissible increment, not the current deformation gradient.
static void UpdateIncremental(Cache& c,const ContactGeometry& contact,const tdouble3* position,const unsigned* ridp,double waveSpeed){
  c.eventPending=false;auto& s=*c.reg;const double elapsed=c.eventTime-s.lastEvent;
  const double tau=10*c.kernelH/std::max(waveSpeed,1.e-12);if(elapsed<.5*tau)return;
  auto start=std::chrono::steady_clock::now();const auto previousEta=c.eta;UpdateMixed(c,contact,position,ridp);
  double change=0;for(unsigned i=0;i<c.rows.size();i++)change=std::max(change,std::abs(c.eta[i]-previousEta[i]));
  const double fraction=-std::expm1(-elapsed/tau);std::vector<double> targetEta=c.eta;
  for(unsigned i=0;i<c.rows.size();i++)targetEta[i]=previousEta[i]+fraction*(targetEta[i]-previousEta[i]);c.eta=previousEta;s.lastEvent=c.eventTime;
  if(change<1.e-6)return;
  const unsigned n=c.rows.size(),m=s.ids.size();const int dim=c.dim;
  std::vector<std::vector<Vec>> R(n),dq(n);std::vector<Mat> inverse(n);
  for(unsigned i=0;i<n;i++)if(s.active[i]){
    const auto xi=Coordinates(position[ridp[i]],dim);Mat moment(dim*dim,0);R[i].resize(c.rows[i].size());dq[i].resize(c.rows[i].size(),Vec{{0,0,0}});
    for(size_t e=0;e<c.rows[i].size();e++){
      const auto xj=Coordinates(position[ridp[c.rows[i][e].j]],dim);auto& r=R[i][e];
      for(int a=0;a<dim;a++)r[a]=(xj[a]-xi[a])/c.dp;
      for(int a=0;a<dim;a++)for(int k=0;k<dim;k++)moment[a*dim+k]+=r[a]*r[k];
    }
    inverse[i]=Inverse(moment,dim);
  }
  const auto& diagonal=s.referenceDiagonal;
  const auto project=[&](const RegularizedMG::Vector& x,std::vector<std::vector<Vec>>* result,int component){
    RegularizedMG::Vector lambda(n,0),out(n,0);for(unsigned a=0;a<m;a++)lambda[s.ids[a]]=x[a];
    for(unsigned i=0;i<n;i++)if(s.active[i]){
      Vec moment={{0,0,0}},coef=moment;
      for(size_t e=0;e<c.rows[i].size();e++)for(int a=0;a<dim;a++)moment[a]+=R[i][e][a]*(lambda[c.rows[i][e].j]-lambda[i]);
      for(int a=0;a<dim;a++)for(int k=0;k<dim;k++)coef[a]+=inverse[i][a*dim+k]*moment[k];
      for(size_t e=0;e<c.rows[i].size();e++){
        const unsigned j=c.rows[i][e].j;double v=lambda[j]-lambda[i];for(int a=0;a<dim;a++)v-=R[i][e][a]*coef[a];
        v*=c.inwardWeight[i];out[i]-=v;out[j]+=v;if(result)(*result)[i][e][component]=v;
      }
    }
    RegularizedMG::Vector y(m);for(unsigned a=0;a<m;a++)y[a]=out[s.ids[a]]+1.e-7*diagonal[a]*x[a];return y;
  };

  const auto assemblyStart=std::chrono::steady_clock::now();
  std::vector<unsigned> index(n,~0u);for(unsigned a=0;a<m;a++)index[s.ids[a]]=a;
  std::vector<std::map<unsigned,double>> assembled(m);
  for(unsigned i=0;i<n;i++)if(s.active[i]){
    const unsigned degree=c.rows[i].size(),count=degree+1;
    std::vector<unsigned> nodes(count);std::vector<Vec> C(count,Vec{{0,0,0}});
    nodes[0]=index[i];
    for(unsigned e=0;e<degree;e++){
      nodes[e+1]=index[c.rows[i][e].j];C[e+1]=R[i][e];
      for(int k=0;k<dim;k++)C[0][k]-=R[i][e][k];
    }
    for(unsigned a=0;a<count;a++){
      Vec coef={{0,0,0}};for(int k=0;k<dim;k++)for(int t=0;t<dim;t++)coef[k]+=inverse[i][k*dim+t]*C[a][t];
      for(unsigned j=0;j<count;j++){
        double v=(a==0&&j==0)?double(degree):((a==0||j==0)?-1.:(a==j?1.:0.));
        for(int k=0;k<dim;k++)v-=coef[k]*C[j][k];
        assembled[nodes[a]][nodes[j]]+=c.inwardWeight[i]*v;
      }
    }
  }
  RegularizedMG::Sparse current(m,m);
  for(unsigned a=0;a<m;a++){
    assembled[a][a]+=1.e-7*diagonal[a];current.row[a].reserve(assembled[a].size());
    for(const auto& e:assembled[a])current.row[a].push_back(e);
    assembled[a].clear();
  }
  const double assemblySeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-assemblyStart).count();
  const auto mgStart=std::chrono::steady_clock::now();
  std::vector<RegularizedMG::Point> coords(m);
  for(unsigned a=0;a<m;a++)coords[a]=Coordinates(position[ridp[s.ids[a]]],dim);
  current.Pack();
  RegularizedMG::Solver currentSolver(current,coords,dim);
  const double mgSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-mgStart).count();
  double matvecSeconds[3]={0,0,0},preconditionSeconds[3]={0,0,0};
  const auto multiply=[&](const RegularizedMG::Vector& v){auto t=std::chrono::steady_clock::now();auto y=current.Apply(v);matvecSeconds[ThreadIndex()]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return y;};
  const auto precondition=[&](const RegularizedMG::Vector& v){auto t=std::chrono::steady_clock::now();auto y=currentSolver.Apply(v);preconditionSeconds[ThreadIndex()]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return y;};
  unsigned iterations=0;
  std::exception_ptr solveErrors[3];
  int solveThreads=1;
  #ifdef OMP_USE
    solveThreads=std::min(dim,omp_get_max_threads());
  #endif
  #ifdef OMP_USE
    #pragma omp parallel for num_threads(solveThreads) reduction(+:iterations)
  #endif
  for(int k=0;k<dim;k++){
    try{
    RegularizedMG::Vector rhs(m),x(m,0);
    for(unsigned a=0;a<m;a++){const unsigned i=s.ids[a];rhs[a]=(targetEta[i]-previousEta[i])*(s.solid[i][k]-s.fluid[i][k]);}
    const double tolerance=1.e-4*std::max(std::sqrt(RegularizedMG::Inner(rhs,rhs)),1.e-15);

    auto r=rhs;if(std::sqrt(RegularizedMG::Inner(r,r))<=tolerance)continue;
    auto z=precondition(r),p=z;double rz=RegularizedMG::Inner(r,z);unsigned it=0;
    for(;it<c.maxIterations;it++){
      auto ap=multiply(p);double pap=RegularizedMG::Inner(p,ap);Check(pap>0 && rz>0 && std::isfinite(pap+rz),"incremental projection CG breakdown");const double step=rz/pap;
      for(unsigned a=0;a<m;a++){x[a]+=step*p[a];r[a]-=step*ap[a];}
      if(std::sqrt(RegularizedMG::Inner(r,r))<=tolerance){it++;break;}
      z=precondition(r);const double next=RegularizedMG::Inner(r,z);for(unsigned a=0;a<m;a++)p[a]=z[a]+next/rz*p[a];rz=next;
    }
    auto residual=project(x,&dq,k);for(unsigned a=0;a<m;a++)residual[a]-=rhs[a];
    Check(std::sqrt(RegularizedMG::Inner(residual,residual))<2*tolerance,"dynamic projection reached maxiterations before convergence");iterations+=it;
    }catch(...){solveErrors[k]=std::current_exception();}
  }
  for(int k=0;k<dim;k++)if(solveErrors[k])std::rethrow_exception(solveErrors[k]);
  size_t nnz=0;for(const auto& row:current.row)nnz+=row.size();
  double accepted=0,maxGain=0,jump=0;
  for(double alpha=1;alpha>=1./128;alpha*=.5){
    double gain=0;bool valid=true;
    for(unsigned i=0;i<n&&valid;i++)if(s.active[i]){
      Vec sum={{0,0,0}};double norm=0;
      for(size_t e=0;e<c.rows[i].size();e++)for(int k=0;k<dim;k++){
        const double g=c.volume[c.rows[i][e].j]*c.rows[i][e].D[k]+alpha*dq[i][e][k]/c.volume[i];norm+=g*g;sum[k]+=g;
      }
      for(int k=0;k<dim;k++)norm+=sum[k]*sum[k];gain=std::max(gain,std::sqrt(norm/std::max(c.referenceNorm2[i],1.e-30)));if(norm>4*c.referenceNorm2[i])valid=false;
    }
    if(valid){accepted=alpha;maxGain=gain;break;}
  }
  std::vector<Vec> area(n,Vec{{0,0,0}});
  for(unsigned i=0;i<n;i++){
    Mat moment(dim*dim,0);
    for(size_t e=0;e<c.rows[i].size();e++){
      auto& edge=c.rows[i][e];
      for(int k=0;k<dim;k++){
        if(s.active[i]){
          edge.D[k]+=accepted*dq[i][e][k]/(c.volume[i]*c.volume[edge.j]);
          for(int a=0;a<dim;a++)moment[a*dim+k]+=c.dp*R[i][e][a]*accepted*dq[i][e][k]/c.volume[i];
        }
        const double q=c.volume[i]*c.volume[edge.j]*edge.D[k];area[i][k]-=q;area[edge.j][k]+=q;
      }
    }
    for(double a:moment)jump=std::max(jump,std::abs(a));c.eta[i]=previousEta[i]+accepted*(targetEta[i]-previousEta[i]);
  }
  double rr=0,tt=0;for(unsigned i=0;i<n;i++)for(int k=0;k<dim;k++){
    const double target=s.fluid[i][k]+c.eta[i]*(s.solid[i][k]-s.fluid[i][k]);rr+=(area[i][k]-target)*(area[i][k]-target);tt+=target*target;
  }
  //std::fprintf(stderr,"Mode 5 update time=%.9g iterations=%u alpha=%.9g etaStep=%.9g lag=%.9g residual=%.9g gain=%.9g Fjump=%.9g seconds=%.9g\n",c.eventTime,iterations,accepted,accepted*fraction*change,(1-accepted*fraction)*change,std::sqrt(rr/std::max(tt,1.e-30)),maxGain,jump,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
}

Diagnostics Initialize(const Options& o,bool twoD,double dp,unsigned n,
    const tdouble3* reference,const unsigned* body,const byte* selected,
    const float* volume,const unsigned* count,const unsigned* const* pairs,
    const std::function<double(double)>& kernelFac,std::shared_ptr<Cache>& handle){
  Options endpoint=o;endpoint.BoundaryForm=o.BoundaryForm==2?2:1;
  std::vector<Vec> fluid,solid;
  auto c=BuildTemplate(endpoint,twoD,dp,n,reference,body,selected,volume,count,pairs,kernelFac,fluid);
  if(o.BoundaryForm==3){
    endpoint.BoundaryForm=2;
    BuildTemplate(endpoint,twoD,dp,n,reference,body,selected,volume,count,pairs,kernelFac,solid);
  }else solid=fluid;
  c->reg=std::make_shared<RegularizedState>();c->reg->fluid=std::move(fluid);c->reg->solid=std::move(solid);
  c->maxIterations=o.MaxIterations;c->kernelH=o.KernelH;c->mixed=o.BoundaryForm==3;
  c->quadraticError=o.QuadraticError;
  if(o.QuadraticError>0){
    c->quadraticReference.assign(reference,reference+n);
    c->quadraticBody.assign(body,body+n);
    c->quadraticSelected.assign(selected,selected+n);
  }
  c->mass.assign(o.ParticleMass,o.ParticleMass+n);c->referenceNorm2.assign(n,0);
  c->eta.assign(n,0);c->blend.resize(n);
  if(c->mixed)BuildBlend(*c,n,c->dim,dp,reference,body,volume,kernelFac);
  BuildInitialSystem(*c,reference);
  if(!c->mixed)FinishInitialProjection(*c);
  Diagnostics report;
  report.AreaError=c->reg->initialResidual;
  report.Approximation=c->mixed?"mixed projection completed at first contact geometry evaluation":"regularized area target with exact affine moments";
  handle=c;return report;
}
void DefGrad(
    const std::shared_ptr<Cache>& handle,unsigned n,
    const tdouble3* position,const unsigned* ridp,
    tmatrix3f* result,const ContactGeometry* contact)
{
  auto& c=Get(handle,n);
  const int d=c.dim;
  PrepareRuntime(c);

  if(c.mixed){
    Check(
        contact!=nullptr,
        "mixed contact geometry was not supplied.");

    c.pendingContact=*contact;
    if(!c.localReady){
      UpdateMixed(c,*contact,position,ridp);
      FinishInitialProjection(c);
    }
  }

  #ifdef OMP_USE
    #pragma omp parallel for schedule(guided)
  #endif
  for(unsigned i=0;i<n;i++){
    double f[3][3]={{0}};

    const Vec xi=Coordinates(position[ridp[i]],d);

    for(const auto& e:c.rows[i]){
      const Vec xj=Coordinates(position[ridp[e.j]],d);

      const Vec D=e.D;
      for(int a=0;a<d;a++)
        for(int b=0;b<d;b++)
          f[a][b]+=
              c.volume[e.j]*(xj[a]-xi[a])*D[b];


    }

    if(c.rows[i].empty())
      for(int k=0;k<d;k++)f[k][k]=1;

    if(d==2)
      result[i]={
          float(f[0][0]),0,float(f[0][1]),
          0,1,0,
          float(f[1][0]),0,float(f[1][1])
      };
    else
      result[i]={
          float(f[0][0]),float(f[0][1]),float(f[0][2]),
          float(f[1][0]),float(f[1][1]),float(f[1][2]),
          float(f[2][0]),float(f[2][1]),float(f[2][2])
      };
  }
}

void ElasticForce(
    const std::shared_ptr<Cache>& handle,unsigned n,
    const tmatrix3f* pk,std::vector<tdouble3>& force)
{
  auto& c=Get(handle,n);
  PrepareRuntime(c);
  force.resize(n);

  #ifdef OMP_USE
    #pragma omp parallel for schedule(guided)
  #endif
  for(unsigned i=0;i<n;i++){
    for(size_t e=0;e<c.rows[i].size();e++){
      const auto& edge=c.rows[i][e];
      c.flux[c.offset[i]+e]=ApplyPK(pk[i],edge.D,c.dim)*
          (c.volume[i]*c.volume[edge.j]);
    }
  }

  #ifdef OMP_USE
    #pragma omp parallel for schedule(guided)
  #endif
  for(unsigned i=0;i<n;i++)
    force[i]=GatherFlux(c,i,c.flux);

  // Held numerical states: no instantaneous gate derivative applies.
}

const AuxiliaryLoads& GetAuxiliaryLoads(const std::shared_ptr<Cache>& handle){
  Check(bool(handle),"uninitialized auxiliary loads.");
  return handle->loads;
}

void ElasticForceWithHourglass(
    const std::shared_ptr<Cache>& handle,unsigned n,
    const tdouble3* reference,const tdouble3* position,const unsigned* ridp,
    const tmatrix3f* deformation,const tmatrix3f* referencePK,
    const double* hgModulus,const byte* clamp,
    const std::function<double(double)>& referenceKernel,
    std::vector<tdouble3>& force)
{
  auto& c=Get(handle,n);
  const int d=c.dim;
  Check(reference&&position&&ridp&&deformation&&referencePK&&
      hgModulus&&clamp&&bool(referenceKernel),
      "incomplete hourglass data.");

  PrepareRuntime(c);

  bool anyHg=false;
  for(unsigned i=0;i<n;i++){
    Check(std::isfinite(hgModulus[i])&&hgModulus[i]>=0,
        "hourglass modulus must be finite and nonnegative.");
    anyHg=anyHg || hgModulus[i]>0;
  }

  if(anyHg)
    PrepareHourglass(c,reference,referenceKernel);

  c.loads.ClampForce.resize(n);
  double energy=0;
  int invalid=0;

  if(anyHg){
    #ifdef OMP_USE
      #pragma omp parallel for schedule(guided) reduction(+:energy) reduction(|:invalid)
    #endif
    for(unsigned i=0;i<n;i++){
      c.totalPK[i]=referencePK[i];

      for(size_t e=c.offset[i];e<c.offset[i+1];e++)
        c.hgFlux[e]=TDouble3(0);

      if(hgModulus[i]==0)continue;

      const auto& f=deformation[i];
      const double F[3][3]={
          {f.a11,d==2?f.a13:f.a12,f.a13},
          {d==2?f.a31:f.a21,d==2?f.a33:f.a22,f.a23},
          {f.a31,f.a32,f.a33}};

      double H[3][3]={{0}};
      const Vec xi=Coordinates(position[ridp[i]],d);

      for(size_t eindex=0;eindex<c.rows[i].size();eindex++){
        const size_t id=c.offset[i]+eindex;
        const unsigned j=c.rows[i][eindex].j;
        const Vec xj=Coordinates(position[ridp[j]],d);
        const Vec& R=c.hgR[id];

        Vec r={{0,0,0}},e=r;
        double r2=0;
        for(int a=0;a<d;a++){
          r[a]=xj[a]-xi[a];
          r2+=r[a]*r[a];
        }

        // Report errors after leaving the OpenMP region.
        if(!std::isfinite(r2)||r2<=0){
          invalid=1;
          continue;
        }

        const double k=hgModulus[i]*c.hgWeight[id];
        if(k==0)continue;

        // Penalize the complete non-affine displacement residual.
        double residual2=0;
        for(int a=0;a<d;a++){
          e[a]=r[a];
          for(int b=0;b<d;b++)
            e[a]-=F[a][b]*R[b];
          residual2+=e[a]*e[a];
        }

        energy+=0.5*k*residual2;

        // Differentiate the same energy through the held gradient.
        Vec q={{0,0,0}};
        for(int a=0;a<d;a++){
          q[a]=k*e[a];
          for(int b=0;b<d;b++)
            H[a][b]-=q[a]*R[b]/c.volume[i];
        }

        c.hgFlux[id]=d==2
            ? TDouble3(q[0],0,q[1])
            : TDouble3(q[0],q[1],q[2]);
      }

      auto& p=c.totalPK[i];
      p.a11+=float(H[0][0]);
      if(d==2){
        p.a13+=float(H[0][1]);
        p.a31+=float(H[1][0]);
        p.a33+=float(H[1][1]);
      }
      else{
        p.a12+=float(H[0][1]);
        p.a13+=float(H[0][2]);
        p.a21+=float(H[1][0]);
        p.a22+=float(H[1][1]);
        p.a23+=float(H[1][2]);
        p.a31+=float(H[2][0]);
        p.a32+=float(H[2][1]);
        p.a33+=float(H[2][2]);
      }
    }

    Check(!invalid,"collapsed or invalid hourglass pair.");
    c.loads.HourglassEnergy=energy;

    // Use the same held gradient for elastic and hourglass forces.
    ElasticForce(handle,n,c.totalPK.data(),force);
  }
  else{
    c.loads.HourglassEnergy=0;
    ElasticForce(handle,n,referencePK,force);
  }

  double fx=0,fy=0,fz=0,mx=0,my=0,mz=0;

  #ifdef OMP_USE
    #pragma omp parallel for schedule(guided) reduction(+:fx,fy,fz,mx,my,mz)
  #endif
  for(unsigned i=0;i<n;i++){
    if(anyHg)
      force[i]=force[i]+GatherFlux(c,i,c.hgFlux);

    c.loads.ClampForce[i]=clamp[i]?force[i]:TDouble3(0);

    if(clamp[i]){
      const auto q=force[i],x=position[ridp[i]];
      fx+=q.x;
      fy+=q.y;
      fz+=q.z;
      mx+=x.y*q.z-x.z*q.y;
      my+=x.z*q.x-x.x*q.z;
      mz+=x.x*q.y-x.y*q.x;
    }
  }

  c.loads.ClampTotal=TDouble3(fx,fy,fz);
  c.loads.ClampMoment=TDouble3(mx,my,mz);
}

}

void FlexBoundaryCorrection::BeginAcceptedStep(const UserParameters& user,double time){
  const auto c=user.FlexRestartBoundaryData.Correction;
  if(c && c->mixed){
    if(!c->stepClockReady){c->reg->lastEvent=time;c->stepClockReady=true;}
    c->eventTime=time;c->eventPending=true;
  }
}

void FlexBoundaryCorrection::DefGradForSolver(
    const UserParameters& user,unsigned np,unsigned npb,
    const tdouble3* pos,const typecode* code,
    const unsigned* ridp,tmatrix3f* result)
{
  FlexBoundaryCorrection::ContactGeometry contact;

  if(user.FlexRestartBoundaryCorrectionBoundaryForm==3){
    if(user.FlexBoundContactMode==1){
      contact.Kind=1;
      contact.Fixed.reserve(npb);

      for(unsigned p=0;p<npb;p++)
        if(CODE_IsNotOut(code[p]) &&
            CODE_IsFixed(code[p]) &&
            !CODE_IsFlexStrucAny(code[p])){
          contact.Fixed.push_back(pos[p]);
        }
    }
    else if(user.FlexBoundContactMode==2){
      const auto& surface=user.SurfaceType;

      if(surface.value==1){
        contact.Kind=2;
        contact.Point=TDouble3(0,0,surface.WallZ);
      }
      else if(surface.value==2){
        const double dx=
            surface.point2.x-surface.point1.x;
        const double dz=
            surface.point2.z-surface.point1.z;
        const double slope=dz/dx;

        contact.Kind=2;
        contact.Point=TDouble3(
            surface.point1.x,0,surface.point1.z);

        // Match the existing vertical-gap contact convention.
        contact.Normal=TDouble3(-slope,0,1);

        // Match the existing normal-distance contact convention.
        if(surface.NormalDirectionMode==2)
          contact.Normal=contact.Normal*
              (1./std::sqrt(1.+slope*slope));
      }
    }
  }
    FlexBoundaryCorrection::DefGrad(
      user.FlexRestartBoundaryData.Correction,
      np,pos,ridp,result,&contact);

}

void FlexBoundaryCorrection::ElasticForceForSolver(
    const UserParameters& user,unsigned np,
    const tdouble3* reference,const tdouble3* position,
    const typecode* code,const StFlexStrucData* materialData,
    const unsigned* ridp,const tmatrix3f* deformation,
    const tmatrix3f* referencePK,
    const std::function<double(double)>& kernel,
    std::vector<tdouble3>& force)
{
  std::vector<double> hgModulus(np);
  std::vector<byte> clamp(np);

  for(unsigned i=0;i<np;i++){
    const typecode ci=code[ridp[i]];
    const auto& material=materialData[CODE_GetIbodyFlexStruc(ci)];

    hgModulus[i]=double(material.hgfactor)*material.youngmod;
    clamp[i]=CODE_IsFlexStrucClamp(ci)?1:0;
  }

  auto& cache=*user.FlexRestartBoundaryData.Correction;
  if(cache.mixed && cache.eventPending){
    double speed=0;
    for(unsigned i=0;i<np;i++){
      const auto& m=materialData[CODE_GetIbodyFlexStruc(code[ridp[i]])];
      const auto& f=deformation[i];
      const double J=fmath::Determinant3x3(f);
      Check(std::isfinite(J)&&J>0,"invalid deformation in mixed wave-speed estimate");
      const double rho=cache.mass[i]/(cache.volume[i]*J);
      const double c2=double(m.youngmod)*(1-m.poissratio)/
          (rho*(1+m.poissratio)*(1-2*m.poissratio));
      speed=std::max(speed,std::sqrt(c2));
    }
    UpdateIncremental(cache,cache.pendingContact,position,ridp,speed);
  }

  ElasticForceWithHourglass(
      user.FlexRestartBoundaryData.Correction,np,
      reference,position,ridp,deformation,referencePK,
      hgModulus.data(),clamp.data(),kernel,force);
}

//============================================================================
/// Correct the boundary forces for flexible structure particles.
//=============================================================================
void JSphCpu::InitFlexRestartBoundaryCorrection()
{
  auto& data=User->FlexRestartBoundaryData;
  data=UserParameters::StFlexRestartBoundaryData{};

  const int mode=User->FlexRestartBoundaryCorrectionMode;
  if(mode==0)return;

  if(mode<1 || mode>5)
    Run_Exceptioon("Invalid boundary correction mode.");

  if(mode==5){
    if(FlexStrucCount!=1)
      Run_Exceptioon("Mode 5 currently requires exactly one elastic body.");

    // Flexible particles themselves are encoded as moving boundaries.
    // Fixed clamps are allowed; moving clamps are prescribed moving supports.
    if(CaseNfloat || FtCount)
      Run_Exceptioon("Mode 5 does not support floating rigid bodies.");

    const typecode* code=Code_c->cptr();
    for(unsigned p=0;p<Np;p++){
      if(!CODE_IsNotOut(code[p]))continue;
      if(CODE_IsFloating(code[p]))
        Run_Exceptioon("Mode 5 does not support floating rigid bodies.");
      if(CODE_IsMoving(code[p])&&!CODE_IsFlexStrucFlex(code[p]))
        Run_Exceptioon(
            "Mode 5 does not support moving rigid boundaries or moving clamps.");
    }
  }

  if(User->UseFlexRestartGradientCorrection)
    Run_Exceptioon(
        "Boundary correction cannot use transported direct-continuation gradients.");


  const unsigned np=CaseNflexstruc;
  if(!np || !Pos0c || !KerCorrc || !NumPairsc || !PairIdxc)
    Run_Exceptioon(
        "Boundary correction: reference geometry is unavailable.");

  data.Body.resize(np);
  data.Selected.assign(np,1);

  for(unsigned i=0;i<np;i++){
    data.Body[i]=CODE_GetIbodyFlexStruc(
        Code_c->cptr()[FlexStrucRidpc[i]]);

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

  if(mode==5){
    FlexBoundaryCorrection::Options options;
    options.KernelH=CSP.kernelh;
    options.Layers=User->FlexRestartBoundaryCorrectionLayers;
    options.InwardReach=
        User->FlexRestartBoundaryCorrectionInwardReach;
    options.MaxIterations=User->FlexRestartBoundaryCorrectionMaxIterations;
    options.QuadraticError=User->FlexRestartBoundaryCorrectionQuadraticError;
    options.BoundaryForm=
        User->FlexRestartBoundaryCorrectionBoundaryForm;

    std::vector<float> referenceVolume(np),particleMass(np);

    if(User->IsFlexRestart()){
      const auto& state=User->FlexRestartState;

      if(!state.HasRestartStateLayout() || state.Idp.size()!=np)
        Run_Exceptioon(
            "Mode 5: elastic restart state is unavailable.");

      referenceVolume=state.ReferenceVolume;
      particleMass=state.ParticleMass;
    }
    else{
      for(unsigned i=0;i<np;i++){
        const auto& material=FlexStrucDatac[data.Body[i]];

        referenceVolume[i]=material.vol0;
        particleMass[i]=material.vol0*material.rho0;
      }
    }

    options.ParticleMass=particleMass.data();
    options.ParticleMassCount=np;

    if(TKernel!=KERNEL_Wendland && TKernel!=KERNEL_Cubic)
      Run_Exceptioon(
          "Mode 5: unsupported kernel.");

    const auto fac=[&](double rr)->double {
      return TKernel==KERNEL_Wendland
          ? fsph::GetKernel_Fac<KERNEL_Wendland>(CSP,float(rr))
          : fsph::GetKernel_Fac<KERNEL_Cubic>(CSP,float(rr));
    };

    try{
      const auto report=FlexBoundaryCorrection::Initialize(
          options,Simulate2D,Dp,np,
          Pos0c,data.Body.data(),data.Selected.data(),
          referenceVolume.data(),
          NumPairsc,PairIdxc,fac,data.Correction);

      Log->Printf("  Mode 5: boundary form=%d; layers=%d; maxiterations=%u.",
          options.BoundaryForm,options.Layers,options.MaxIterations);
      if(!report.Approximation.empty())
        Log->Printf(
            "  Mode 5 approximation: %s",
            report.Approximation.c_str());
    }
    catch(const std::exception& e){
      Run_Exceptioon(e.what());
    }

    return;
  }

  const auto capKernel=[&](const tmatrix3f& input)->tmatrix3f
  {
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
          Run_Exceptioon(
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
    const double limit=User->FlexRestartBoundaryCorrectionLimit;

    // Upper clipping only; smaller eigenvalues are retained.
    for(int k=0;k<3;k++){
      if(!std::isfinite(a[k][k]))
        Run_Exceptioon(
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

    return TMatrix3f(
        float(b[0][0]),float(b[0][1]),float(b[0][2]),
        float(b[1][0]),float(b[1][1]),float(b[1][2]),
        float(b[2][0]),float(b[2][1]),float(b[2][2]));
  };

  if(mode==2){
    data.ForceKernel.resize(np);

    for(unsigned i=0;i<np;i++)
      data.ForceKernel[i]=data.Selected[i]
          ? capKernel(KerCorrc[i]) : KerCorrc[i];
  }

  if(mode==3)
    data.Prestress.assign(np,TMatrix3f(0));

  if(mode==3 && !User->IsFlexRestart())
    Log->Print(
        "  Mode 3: ordinary stress-free reference; "
        "prestress compensation is zero.");

  if(mode==3 && User->IsFlexRestart()){
    const auto& state=User->FlexRestartState;

    if(!state.HasRestartStateLayout() || state.Idp.size()!=np)
      Run_Exceptioon(
          "Boundary correction: initial elastic state is unavailable.");

    for(unsigned i=0;i<np;i++)if(data.Selected[i]){
      const unsigned id=Idp_c->cptr()[FlexStrucRidpc[i]];
      const tmatrix3f& fs=User->GetFlexRestartFstar(i,id);

      const tmatrix3f pn=CalcFlexStrucPK1Stress(
          fs,FlexStrucDatac[data.Body[i]].cmat);

      data.Prestress[i]=ScaleFlexMatrix(
          fmath::MulMatrix3x3(pn,fmath::TrasMatrix3x3(fs)),
          1.f/state.Jstar[i]);
    }
  }

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
}

tmatrix3f JSphCpu::BuildFlexRestartBoundaryForceMatrix(
    unsigned pfs,unsigned body,
    const tmatrix3f& frelative,const tmatrix3f& preference,
    const tmatrix3f& kercorr)const
{
  const tmatrix3f original=
      fmath::MulMatrix3x3(preference,kercorr);

  const int mode=User->FlexRestartBoundaryCorrectionMode;
  if(mode==0)return original;

  const auto& data=User->FlexRestartBoundaryData;

  if(pfs>=data.Body.size() || pfs>=data.Selected.size() ||
     data.Body[pfs]!=body)
    Run_Exceptioon(
        "Boundary correction: missing or inconsistent particle cache.");

  switch(mode){
    case 1:
      // Selected particles use identity correction in force assembly.
      return data.Selected[pfs] ? preference : original;

    case 2:
      if(pfs>=data.ForceKernel.size())
        Run_Exceptioon(
            "Boundary correction: force kernel cache is unavailable.");

      return fmath::MulMatrix3x3(
          preference,data.ForceKernel[pfs]);

    case 3:{
      if(!data.Selected[pfs])return original;

      if(pfs>=data.Prestress.size())
        Run_Exceptioon(
            "Boundary correction: prestress cache is unavailable.");

      // Q = P*B + F_relative*P_initial*(I-B).
      const tmatrix3f transported=
          fmath::MulMatrix3x3(frelative,data.Prestress[pfs]);

      const tmatrix3f corrected=
          fmath::MulMatrix3x3(transported,kercorr);

      tmatrix3f result=original;

      result.a11+=transported.a11-corrected.a11;
      result.a12+=transported.a12-corrected.a12;
      result.a13+=transported.a13-corrected.a13;

      result.a21+=transported.a21-corrected.a21;
      result.a22+=transported.a22-corrected.a22;
      result.a23+=transported.a23-corrected.a23;

      result.a31+=transported.a31-corrected.a31;
      result.a32+=transported.a32-corrected.a32;
      result.a33+=transported.a33-corrected.a33;

      return result;
    }

    case 4:
      if(pfs>=data.Beta.size() || !std::isfinite(data.Beta[pfs]) ||
         data.Beta[pfs]<=0.f)
        Run_Exceptioon("Mode 4: invalid geometric energy weight.");

      return ScaleFlexMatrix(original,data.Beta[pfs]);

    default:
      Run_Exceptioon("Invalid boundary correction mode.");
  }

  return original;
}
