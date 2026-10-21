#ifndef USERFLEXINLINE_H
#define USERFLEXINLINE_H

#include "JSphCpu.h"
#include "FunctionsMath.h"

namespace {

  tmatrix3f ScaleFlexMatrix(tmatrix3f m,float s){
    m.a11*=s; m.a12*=s; m.a13*=s;
    m.a21*=s; m.a22*=s; m.a23*=s;
    m.a31*=s; m.a32*=s; m.a33*=s;
    return m;
  }

}

//==============================================================================
/// Calculates the PK1 stress matrix for each flexible structure particle.
/// Calcula la matriz de tensión PK1 para cada partícula de estructura flexible.
//==============================================================================
inline tmatrix3f JSphCpu::CalcFlexStrucPK1Stress(const tmatrix3f& defgrad,const tmatrix6f& cmat)const{
  //-Calculate Green-Lagrange strain from deformation gradient.
  tmatrix3f gl=fmath::MulMatrix3x3(fmath::TrasMatrix3x3(defgrad),defgrad);
  gl.a11-=1.0f; gl.a22-=1.0f; gl.a33-=1.0f;
  gl.a11*=0.5f; gl.a12*=0.5f; gl.a13*=0.5f;
  gl.a21*=0.5f; gl.a22*=0.5f; gl.a23*=0.5f;
  gl.a31*=0.5f; gl.a32*=0.5f; gl.a33*=0.5f;
  //-Convert to Voigt notation.
  const float gl_1=gl.a11;
  const float gl_2=gl.a22;
  const float gl_3=gl.a33;
  const float gl_4=gl.a23+gl.a32;
  const float gl_5=gl.a31+gl.a13;
  const float gl_6=gl.a12+gl.a21;
  //-Multiply with stiffness tensor to get PK2 stress in Voigt form.
  const float pk2_1=cmat.a11*gl_1+cmat.a12*gl_2+cmat.a13*gl_3+cmat.a14*gl_4+cmat.a15*gl_5+cmat.a16*gl_6;
  const float pk2_2=cmat.a21*gl_1+cmat.a22*gl_2+cmat.a23*gl_3+cmat.a24*gl_4+cmat.a25*gl_5+cmat.a26*gl_6;
  const float pk2_3=cmat.a31*gl_1+cmat.a32*gl_2+cmat.a33*gl_3+cmat.a34*gl_4+cmat.a35*gl_5+cmat.a36*gl_6;
  const float pk2_4=cmat.a41*gl_1+cmat.a42*gl_2+cmat.a43*gl_3+cmat.a44*gl_4+cmat.a45*gl_5+cmat.a46*gl_6;
  const float pk2_5=cmat.a51*gl_1+cmat.a52*gl_2+cmat.a53*gl_3+cmat.a54*gl_4+cmat.a55*gl_5+cmat.a56*gl_6;
  const float pk2_6=cmat.a61*gl_1+cmat.a62*gl_2+cmat.a63*gl_3+cmat.a64*gl_4+cmat.a65*gl_5+cmat.a66*gl_6;
  //-Convert PK2 stress back to normal notation.
  const tmatrix3f pk2={pk2_1,pk2_6,pk2_5,pk2_6,pk2_2,pk2_4,pk2_5,pk2_4,pk2_3};
  //-Convert PK2 to PK1 and return.
  return fmath::MulMatrix3x3(defgrad,pk2);
}

#endif