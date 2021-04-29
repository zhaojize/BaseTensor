#ifndef CPPBLASEX_H
#define CPPBLASEX_H

#include <math.h>

#include "globalconfig.h"
#include "Complex.h"

inline double conj(const double &s) {return s;}

inline double dabs(const double &s) {return fabs(s);}

inline double dabs(const Complex &z) {return abs(z);}

inline double dabs2(const double &s) {return s*s;}

inline double dabs2(const Complex &z) {return z.real()*z.real()+z.imag()*z.imag();}

inline void conj(const BLASINT &dim, double* vec){};

inline void conj(const BLASINT &dim, Complex* vec) {for (BLASINT i = 0; i < dim; i++) vec[i] = conj(vec[i]);}

//void directmultiply(const DATATYPE* A, const  BLASINT &m, const BLASINT &n, const DATATYPE* B, const BLASINT &p, const BLASINT &q, DATATYPE* C);
#endif
