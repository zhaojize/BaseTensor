#ifndef CPPLAPACK_H
#define CPPLAPACK_H

/*version v0.4.0*/

#include "globalconfig.h"
#include "Complex.h"

/*-------exact diagonalization-hermite matrix------*/

bool heev(const BLASINT &dim, double* matrix, const BLASINT &lda, double* eigenvalue, const char &order = 'D');

bool heev(const BLASINT &dim, Complex* matrix, const BLASINT &lda, double* eigenvalue, const char &order = 'D');

bool heevd(const BLASINT &dim, double* matrix, const BLASINT &lda, double* eigenvalue, const char &order = 'D');

bool heevd(const BLASINT &dim, Complex* matrix, const BLASINT &lda, double* eigenvalue, const char &order = 'D');

bool heevr(const BLASINT &dim, double* matrix, const BLASINT &nv, const double &abstol, double* eigenvalue, double* U);

bool heevr(const BLASINT &dim, Complex* matrix, const BLASINT &lda, double* eigenvalue, const char &order = 'D');

bool heevx(const BLASINT &dim, double* matrix, const BLASINT &nv, const double &abstol, double* eigenvalue, double* U);	

/*------------general matrix diagonalization---------------*/

bool geev(const char &jobvl, const char &jobvr, const BLASINT &n, Complex* a, const BLASINT &lda, Complex* w, Complex* vl, 
	  const BLASINT &ldvl, Complex* vr, const BLASINT &ldvr, Complex* work, const BLASINT &lwork, double* rwork, BLASINT &info);

bool geevx(const char &balanc, const char &jobvl, const char &jobvr, const char &sense, const BLASINT &n, Complex* a, 
	   const BLASINT &lda, Complex* w, Complex* vl, const BLASINT &ldvl, Complex* vr, const BLASINT &ldvr, BLASINT &ilo, 
	   BLASINT &ihi, double* scale, double &abnrm, double* rconde, double* rcondv, Complex* work, const BLASINT &lwork, double* rwork, 
	   BLASINT &info);


/*---------------------------*/

BLASINT gelqf(const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* tau, double* work, const BLASINT &lwork, BLASINT &info);

BLASINT gelqf(const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* tau, Complex* work, const BLASINT &lwork, BLASINT &info);

BLASINT geqrf(const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* tau, double* work, const BLASINT &lwork, BLASINT &info);

BLASINT geqrf(const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* tau, Complex* work, const BLASINT &lwork, BLASINT &info);

BLASINT orglq(const BLASINT &m, const BLASINT &n, const BLASINT &k, double* matrix, const BLASINT &lda, double* tau, double* work, const BLASINT &lwork, BLASINT &info);

BLASINT orglq(const BLASINT &m, const BLASINT &n, const BLASINT &k, Complex* matrix, const BLASINT &lda, double* tau, Complex* work, const BLASINT &lwork, BLASINT &info);   // zunglq

BLASINT orgqr(const BLASINT &m, const BLASINT &n, const BLASINT &k, double* matrix, const BLASINT &lda, double* tau, double *work, const BLASINT &lwork, BLASINT &info);

BLASINT orgqr(const BLASINT &m, const BLASINT &n, const BLASINT &k, Complex* matrix, const BLASINT &lda, double* tau, Complex *work, const BLASINT &lwork, BLASINT &info);  // zungqr

/*-----------svd---------*/

/*for "double", "rwork" is useless, I use it to overload functions; one can set it 0*/

BLASINT gesvd(const char &jobu, const char &jobvt, const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* s, double* u, const BLASINT &ldu, double* vt, const BLASINT &ldvt, double* work, const BLASINT &lwork, double* rwork, BLASINT &info);

BLASINT gesvd(const char &jobu, const char &jobvt, const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* s, Complex* u, const BLASINT &ldu, Complex* vt, const BLASINT &ldvt, Complex* work, const BLASINT &lwork, double* rwork, BLASINT &info);

BLASINT gesvdx(const char &jobu, const char &jobvt, const char &range, const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, const double &vl, const double &vu, const BLASINT &il, const BLASINT &iu, BLASINT &ns, double* s, double* u, const BLASINT &ldu, double* vt, const BLASINT &ldvt, double* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info);

BLASINT gesvdx(const char &jobu, const char &jobvt, const char &range, const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, const double &vl, const double &vu, const BLASINT &il, const BLASINT &iu, BLASINT &ns, double* s, Complex* u, const BLASINT &ldu, Complex* vt, const BLASINT &ldvt, Complex* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info);

BLASINT gesdd(const char &jobz, const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* s, double* u, const BLASINT &ldu, double* vt, const BLASINT &ldvt, double* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info);

BLASINT gesdd(const char &jobz, const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* s, Complex* u, const BLASINT &ldu, Complex* vt, const BLASINT &ldvt, Complex* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info);

#endif

