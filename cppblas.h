#ifndef CPPBLAS_H
#define CPPBLAS_H

#include "globalconfig.h"

#include "Complex.h" 

extern "C"
{
/*double*/

    double dasum_(const BLASINT* n, const double* dx, const BLASINT* incx);

    void daxpy_(const BLASINT* n, const double* a, const double* x, const BLASINT* incx, double* y, const BLASINT* incy);

    void dcopy_(const BLASINT* n, const double* x, const BLASINT* incx, double* y, const BLASINT* incy);

    double ddot_(const BLASINT* n, const double* x, const BLASINT* incx, const double* y, const BLASINT* incy);

    void dgemm_(const char* transa, const char* transb, const BLASINT* m, const BLASINT* n, const BLASINT* k, const double* alpha, 
                const double* a, const BLASINT* lda, const double* b, const BLASINT* ldb, const double* beta, double* c, 
                const BLASINT* ldc);

    void dgemv_(const char* trans, const BLASINT* m, const BLASINT* n, const double* alpha, const double* a, const BLASINT* lda, 
                const double* x, const BLASINT* incx, const double* beta, double* y, const BLASINT* incy); 

    double dnrm2_(const BLASINT* n, const double* x, const BLASINT* incx);

    void dscal_(const BLASINT* n, const double* a, double* x, const BLASINT* incx);

/*Complex*/

    double dzasum_(const BLASINT* n, const Complex* x, const BLASINT* incx);

    void zaxpy_(const BLASINT* n, const Complex* a, const Complex* x, const BLASINT* incx, Complex* y, const BLASINT* incy);

    void zcopy_(const BLASINT* n, const Complex* x, const BLASINT* incx, Complex* y, const BLASINT* incy); 

    Complex zdotc_(const BLASINT* n, const Complex* x, const BLASINT* incx, const Complex* y, const BLASINT* incy);

    Complex zdotu_(const BLASINT* n, const Complex* zx, const BLASINT* incx, const Complex* zy, const BLASINT* incy);

    void zgemm_(const char* transa, const char* transb, const BLASINT* m, const BLASINT* n, const BLASINT* k, const Complex* alpha, 
                const Complex* a, const BLASINT* lda, const Complex* b, const BLASINT* ldb, const Complex* beta, Complex* c, 
                const BLASINT* ldc);
   
    void zgemv_(const char* trans, const BLASINT* m, const BLASINT* n, const Complex* alpha, const Complex* a, const BLASINT* lda, 
                const Complex* x, const BLASINT* incx, const Complex* beta, Complex* y, const BLASINT* incy);

    double dznrm2_(const BLASINT* n, const Complex* x, const BLASINT* incx);
 
    void zscal_(const BLASINT* n, const Complex* a, Complex* x, const BLASINT* incx);

    void zdscal_(const BLASINT* n, const double* a, Complex* x, const BLASINT* incx);
}

/*c++ interface to blas*/

/*asum*/

inline double asum(const BLASINT &n, const double* dx, const BLASINT &incx)
{
    return dasum_(&n, dx, &incx);
}

inline double asum(const BLASINT &n, const Complex* zx, const BLASINT &incx)
{
    return dzasum_(&n, zx, &incx);
}


/*axpy*/

inline void axpy(const BLASINT &n, const double &da, const double* dx, const BLASINT &incx, double* dy, const BLASINT &incy)
{
    daxpy_(&n, &da, dx, &incx, dy, &incy);
}

inline void axpy(const BLASINT &n, const Complex &za, const Complex* zx, const BLASINT &incx, Complex* zy, const BLASINT &incy)
{
    zaxpy_(&n, &za, zx, &incx, zy, &incy);
}

/*copy*/

inline void copy(const BLASINT &n, const double* dx, const BLASINT &incx, double* dy, const BLASINT &incy)
{
    dcopy_(&n, dx, &incx, dy, &incy);
}

inline void copy(const BLASINT &n, const Complex* zx, const BLASINT &incx, Complex* zy, const BLASINT &incy)
{
    zcopy_(&n, zx, &incx, zy, &incy);
}

/*dotu*/

inline double dotu(const BLASINT &n, const double* dx, const BLASINT &incx, const double* dy,const BLASINT &incy)
{
    return ddot_(&n, dx, &incx, dy, &incy);
}

inline Complex dotu(const BLASINT &n, const Complex* zx, const BLASINT &incx, const Complex* zy, const BLASINT &incy)
{
//    return zdotu_(&n, zx, &incx, zy, &incy);

    Complex a = 0;

    for (BLASINT i = 0; i < n; i++) a += zx[i]*zy[i];

    return a;
}

/*dotc*/

inline double dotc(const BLASINT &n, const double* dx, const BLASINT &incx, const double* dy, const BLASINT &incy)
{
    return ddot_(&n, dx, &incx, dy, &incy);
}

inline Complex dotc(const BLASINT &n, const Complex* zx, const BLASINT &incx, const Complex* zy, const BLASINT &incy)
{
//    return zdotc_(&n, zx, &incx, zy, &incy);

    Complex a = 0;

    for (BLASINT i = 0; i < n; i++) a+=conj(zx[i])*zy[i];

    return a;
}

/*gemm*/

inline void gemm(const char &transa, const char &transb, const BLASINT &m, const BLASINT &n, const BLASINT &k, const double &alpha,
          const double* a, const BLASINT &lda, const double* b, const BLASINT &ldb, const double &beta, double* c, const BLASINT &ldc)
{
    dgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}

inline void gemm(const char &transa, const char &transb, const BLASINT &m, const BLASINT &n, const BLASINT &k, const Complex &alpha,
          const Complex* a, const BLASINT &lda, const Complex* b, const BLASINT &ldb, const Complex &beta, Complex* c, const BLASINT &ldc)
{
    zgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda, &beta, c, &ldc);
}


/*gemv*/

void gemv(const char &trans, const BLASINT &m, const BLASINT &n, const double &alpha, const double* a, const BLASINT &lda, 
          const double* x, const BLASINT &incx, const double &beta, double* y, const BLASINT &incy);

void gemv(const char &trans, const BLASINT &m, const BLASINT &n, const Complex &alpha, const Complex* a, const BLASINT &lda, 
          const Complex* x, const BLASINT &incx, const Complex &beta, Complex* y, const BLASINT &incy);

/*nrm2*/

inline double nrm2(const BLASINT &n, const double* x, const BLASINT &incx)
{
    return dnrm2_(&n, x, &incx);
}

inline double nrm2(const BLASINT &n, const Complex* x, const BLASINT &incx)
{
    return dznrm2_(&n, x, &incx);
}

/*scal*/

inline void scal(const BLASINT &n, const double &da, double* x, const BLASINT &incx)
{
    dscal_(&n, &da, x, &incx);
}

inline void scal(const BLASINT &n, const Complex &za, Complex* x, const BLASINT &incx)
{
    zscal_(&n, &za, x, &incx);
}

inline void scal(const BLASINT &n, const double &da, Complex* x, const BLASINT &incx)
{
    zdscal_(&n, &da, x, &incx);
}
#endif
