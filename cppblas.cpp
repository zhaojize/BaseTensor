#include <assert.h>

#include "cppblas.h"

/*gemv*/

void gemv(const char &trans, const BLASINT &m, const BLASINT &n, const double &alpha, const double* a, const BLASINT &lda,
          const double* x, const BLASINT &incx, const double &beta, double* y, const BLASINT &incy)
{
    if(trans=='N'||trans=='n')
    {
        const char transf='T';
        dgemv_(&transf, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
    else  // 'T', 't', or 'C', 'c' 
    {
        const char transf='N';
        dgemv_(&transf, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
}

void gemv(const char &trans, const BLASINT &m, const BLASINT &n, const Complex &alpha, const Complex* a, const BLASINT &lda,
          const Complex* x, const BLASINT &incx, const Complex &beta, Complex* y, const BLASINT &incy)
{
    if(trans=='N'||trans=='n')
    {
        const char transf='T';
        zgemv_(&transf, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
    else if(trans=='T'||trans=='t')
    {
        const char transf='N';
        zgemv_(&transf, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
    else  // 'C', or 'c', not implemented yet
    {
        assert(0);
    }
}
