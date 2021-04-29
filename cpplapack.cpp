#include <iostream>
#include <math.h>
#include <assert.h>

#ifdef DEBUG_HEEV_TIME
#include <unistd.h>
#include <time.h>
#include <sys/times.h>
#endif

#include "cpplapack.h"

using namespace std;

extern "C"
{
/*---------*/

    void dsyev_(const char* jobz, const char* uplo, const BLASINT* n, double* a, const BLASINT* lda, double* w, double* work,
                BLASINT* lwork, BLASINT* info);

    void zheev_(const char* jobz, const char* uplo, const BLASINT* n, Complex* a, const BLASINT* lda, double* w, Complex* work, 
                BLASINT* lwork, double* rwork, BLASINT* info);

/*----d----*/

    void dsyevd_(const char* jobz, const char* uplo, const BLASINT* n, double* a, const BLASINT* lda, double* w, double* work,
                 BLASINT* lwork, BLASINT* iwork, BLASINT* liwork, BLASINT* info);

    void zheevd_(const char* jobz, const char* uplo, const BLASINT* n, Complex* a, const BLASINT* lda, double* w, Complex* work, 
                 BLASINT* lwork, double* rwork, BLASINT* lrwork, BLASINT* iwork, BLASINT* liwork, BLASINT* info);

/*----r----*/

    void dsyevr_(const char* jobz, const char* range, const char* uplo, const BLASINT* n, double* a, const BLASINT* lda, const double* vl, const double* vu,
                 const BLASINT* il, const BLASINT* iu, const double* abstol, BLASINT* m, double* w, double* z, const BLASINT* ldz, BLASINT* isuppz, double* work,
                 const BLASINT* lwork, BLASINT* iwork, const BLASINT* liwork, BLASINT* info);

    void zheevr_(const char* jobz, const char* range, const char* uplo, const BLASINT* n, Complex* a, const BLASINT* lda, const double* vl, const double* vu, 
                 const BLASINT* il, const BLASINT* iu, const double* abstol, BLASINT* m, double* w, Complex* z, const BLASINT* ldz, BLASINT* isuppz, Complex* work, 
                 const BLASINT* lwork, double* rwork, const BLASINT* lrwork, BLASINT* iwork, const BLASINT* liwork, BLASINT* info); 

/*----x----*/

    void dsyevx_(const char* jobz, const char* range, const char* uplo, const BLASINT* n, double* a, const BLASINT* lda, const double* vl, const double* vu, 
		 const BLASINT* il, const BLASINT* iu, const double* abstol, BLASINT* m, double* w, double* z, const BLASINT* ldz, double* work, 
		 const BLASINT* lwork, BLASINT* iwork, BLASINT* ifail, BLASINT* info);

/*---------------------------------------------------------------------------------------------------------------------------------------*/

    void zgeev_(const char* jobvl, const char* jobvr, const BLASINT* n, Complex* a, const BLASINT* lda, Complex* w, Complex* vl, 
   	        const BLASINT* ldvl, Complex* vr, const BLASINT* ldvr, Complex* work, const BLASINT* lwork, double* rwork, BLASINT* info);

    void zgeevx_(const char* balanc, const char* jobvl, const char* jobvr, const char* sense, const BLASINT* n, Complex* a, 
	         const BLASINT* lda, Complex* w, Complex* vl, const BLASINT* ldvl, Complex* vr, const BLASINT* ldvr, BLASINT* ilo, 
	         BLASINT* ihi, double* scale, double* abnrm, double* rconde, double* rcondv, Complex* work, const BLASINT* lwork, double* rwork, 
	         BLASINT* info);

/*---------------------------------------------------------------------------------------------------------------------------------------*/

    void dgelqf_(const BLASINT* m, const BLASINT* n, double* matrix, const BLASINT* lda, double* tau, double* work, const BLASINT* lwork, BLASINT* info);

    void zgelqf_(const BLASINT* m, const BLASINT* n, Complex* matrix, const BLASINT* lda, double* tau, Complex* work, const BLASINT* lwork, BLASINT* info);

    void dgeqrf_(const BLASINT* m, const BLASINT* n, double* matrix, const BLASINT* lda, double* tau, double* work, const BLASINT* lwork, BLASINT* info);

    void zgeqrf_(const BLASINT* m, const BLASINT* n, Complex* matrix, const BLASINT* lda, double* tau, Complex* work, const BLASINT* lwork, BLASINT* info);

    void dorglq_(const BLASINT* m, const BLASINT* n, const BLASINT* k, double* matrix, const BLASINT* lda, const double* tau, double* work, const BLASINT* lwork, BLASINT* info);

    void zunglq_(const BLASINT* m, const BLASINT* n, const BLASINT* k, Complex* matrix, const BLASINT* lda, const double* tau, Complex* work, const BLASINT* lwork, BLASINT* info);   // zunglq

    void dorgqr_(const BLASINT* m, const BLASINT* n, const BLASINT* k, double* matrix, const BLASINT* lda, const double* tau, double* work, const BLASINT* lwork, BLASINT* info);

    void zungqr_(const BLASINT* m, const BLASINT* n, const BLASINT* k, Complex* matrix, const BLASINT* lda, const double* tau, Complex* work, const BLASINT* lwork, BLASINT* info);  // zungqr

/*-------------------svd------------------*/

    void dgesvd_(const char* jobu, const char* jobvt, const BLASINT* m, const BLASINT* n, double* matrix, const BLASINT* lda, double* s, double* u, const BLASINT* ldu, double* vt, const BLASINT* ldvt, double* work, const BLASINT* lwork, BLASINT* info);

    void zgesvd_(const char* jobu, const char* jobvt, const BLASINT* m, const BLASINT* n, Complex* matrix, const BLASINT* lda, double* s, Complex* u, const BLASINT* ldu, Complex* vt, const BLASINT* ldvt, Complex* work, const BLASINT* lwork, double* rwork, BLASINT* info);


    void dgesvdx_(const char* jobu, const char* jobvt, const char* range, const BLASINT* m, const BLASINT* n, double* matrix, const BLASINT* lda, const double* vl, const double* vu, const BLASINT* il, const BLASINT* iu, BLASINT* ns, double* s, double* u, const BLASINT* ldu, double* vt, const BLASINT* ldvt, double* work, const BLASINT* lwork, BLASINT* iwork, BLASINT* info);

    void zgesvdx_(const char* jobu, const char* jobvt, const char* range, const BLASINT* m, const BLASINT* n, Complex* matrix, const BLASINT* lda, const double* vl, const double* vu, const BLASINT* il, const BLASINT* iu, BLASINT* ns, double* s, Complex* u, const BLASINT* ldu, Complex* vt, const BLASINT* ldvt, Complex* work, const BLASINT* lwork, double* rwork, BLASINT* iwork, BLASINT* info);

    void dgesdd_(const char* jobz, const BLASINT* m, const BLASINT* n, double* matrix, const BLASINT* lda, double* s, double* u, const BLASINT* ldu, double* vt, const BLASINT* ldvt, double* work, const BLASINT* lwork, BLASINT* iwork, BLASINT* info);

    void zgesdd_(const char* jobz, const BLASINT* m, const BLASINT* n, Complex* matrix, const BLASINT* lda, double* s, Complex* u, const BLASINT* ldu, Complex* vt, const BLASINT* ldvt, Complex* work, const BLASINT* lwork, double* rwork, BLASINT* iwork, BLASINT* info);
}

bool heev(const BLASINT &dim, double* matrix, const BLASINT &lda, double* eigenvalue, const char &order)
{
#ifdef DEBUG_HEEV_TIME
    long click = sysconf(_SC_CLK_TCK);
    static struct tms tms_begin_heev;
    int time_begin_heev = times(&tms_begin_heev);
#endif

    char jobz = 'V';
    char uplo = 'U';
    BLASINT info;

    BLASINT lwork = 3*dim;

    double* work = new double[lwork];
    assert(work);

    if (order == 'D') for(BLASINT i = 0; i < dim*dim; i++) matrix[i] = -matrix[i];

    dsyev_(&jobz, &uplo, &dim, matrix, &lda, eigenvalue, work, &lwork, &info);

    delete []work;

    if (info == 0)
    {
        if (order == 'D') for (BLASINT i = 0; i < dim; i++) eigenvalue[i] = -eigenvalue[i];

#ifdef DEBUG_HEEV_TIME
        static struct tms tms_end_heev;
        long time_end_heev = times(&tms_end_heev);
        cout.precision(10);
        cout<<"Time for ExactDiagonalization matrix with dimension " << n << " is :" << (time_end_heev-time_begin_heev)/(1.0*click) << endl;
#endif

        return true;
    }
    else 
    {
        if (info > 0) cout << "The algorithm failed to converge!" << endl;
        else cout << "The " << -info << " parameter is illegal!" << endl;

        return false;
    }
}

//Matrix must arranged by col;

bool heev(const BLASINT &dim, Complex* matrix, const BLASINT &lda, double* eigenvalue, const char &order)
{
#ifdef DEBUG_HEEV_TIME
    long click=sysconf(_SC_CLK_TCK);
    static struct tms tms_begin_heev;
    int time_begin_heev = times(&tms_begin_heev);
#endif

    char jobz = 'V';
    char uplo = 'U';
    BLASINT info;
    BLASINT lwork = 2*dim;

    Complex* work = new Complex[lwork];
    assert(work);

    double* rwork = new double[3*dim];
    assert(rwork);

/*Exchange row and column of Matrix,
*for fortran and c in the order of matrix are different
*/

    for (BLASINT i = 0; i < dim*dim; i++) matrix[i] = conj(matrix[i]);

    if (order == 'D') for (BLASINT i = 0; i < dim*dim; i++) matrix[i] = -matrix[i];

    zheev_(&jobz, &uplo, &dim, matrix, &lda, eigenvalue, work, &lwork, rwork, &info);

    delete []rwork;

    delete []work;

    if (info == 0)
    {
        if (order == 'D') for (BLASINT i = 0; i < dim; i++) eigenvalue[i] = -eigenvalue[i];

#ifdef DEBUG_HEEV_TIME
        static struct tms tms_end_heev;
        int time_end_heev = times(&tms_end_heev);
        cout.precision(10);
        cout << "Time for ExactDiagonalization matrix with dim " << n << " is :" << (time_end_heev-time_begin_heev)/(1.0*click) << endl;
#endif

        return true;
    }
    else 
    {
        if (info > 0) cout << "The algorithm failed to converge!" << endl;
        else cout << "The " << -info << " parameter is illegal!" << endl;

        return false;
    }
}

bool heevd(const BLASINT &dim, double* matrix, const BLASINT &lda, double* eigenvalue, const char &order)
{
#ifdef DEBUG_HEEV_TIME
    long click = sysconf(_SC_CLK_TCK);
    static struct tms tms_begin_heev;
    int time_begin_heev = times(&tms_begin_heev);
#endif

    char jobz = 'V';
    char uplo = 'U';
    BLASINT info;

    BLASINT lwork = 1+6*dim+2*dim*dim;

    double* work = new double[lwork];
    assert(work);

    BLASINT liwork = 3+5*dim;

    BLASINT* iwork = new BLASINT[liwork];
    assert(iwork);

    if (order == 'D') for(BLASINT i = 0; i < dim*dim; i++) matrix[i] = -matrix[i];

    dsyevd_(&jobz, &uplo, &dim, matrix, &lda, eigenvalue, work, &lwork, iwork, &liwork, &info);

    delete []iwork;

    delete []work;

    if (info == 0)
    {
        if (order == 'D') for (BLASINT i = 0; i < dim; i++) eigenvalue[i] = -eigenvalue[i];

#ifdef DEBUG_HEEV_TIME
        static struct tms tms_end_heev;
        long time_end_heev = times(&tms_end_heev);
        cout.precision(10);
        cout<<"Time for ExactDiagonalization matrix with dimension " << n << " is :" << (time_end_heev-time_begin_heev)/(1.0*click) << endl;
#endif

        return true;
    }
    else
    {
        if (info > 0) cout << "The algorithm failed to converge!" << endl;
        else cout << "The " << -info << " parameter is illegal!" << endl;

        return false;
    }
}

bool heevd(const BLASINT &dim, Complex* matrix, const BLASINT &lda, double* eigenvalue, const char &order)
{
#ifdef DEBUG_HEEV_TIME
    long click=sysconf(_SC_CLK_TCK);
    static struct tms tms_begin_heev;
    int time_begin_heev = times(&tms_begin_heev);
#endif

    char jobz = 'V';
    char uplo = 'U';
    BLASINT info;

    BLASINT lwork = 2*dim+dim*dim;

    Complex* work = new Complex[lwork];
    assert(work);

    BLASINT lrwork = 1+5*dim+2*dim*dim;

    double* rwork = new double[lrwork];
    assert(rwork);

    BLASINT liwork = 3+5*dim;

    BLASINT* iwork = new BLASINT[liwork];
    assert(iwork);

/*Exchange row and column of Matrix,
*for fortran and c in the order of matrix are different
*/

    for (BLASINT i = 0; i < dim*dim; i++) matrix[i] = conj(matrix[i]);

    if (order == 'D') for (BLASINT i = 0; i < dim*dim; i++) matrix[i] = -matrix[i];

    zheevd_(&jobz, &uplo, &dim, matrix, &lda, eigenvalue, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);

    delete []iwork;

    delete []rwork;

    delete []work;

    if (info == 0)
    {
        if (order == 'D') for (BLASINT i = 0; i < dim; i++) eigenvalue[i] = -eigenvalue[i];

#ifdef DEBUG_HEEV_TIME
        static struct tms tms_end_heev;
        int time_end_heev = times(&tms_end_heev);
        cout.precision(10);
        cout << "Time for ExactDiagonalization matrix with dim " << n << " is :" << (time_end_heev-time_begin_heev)/(1.0*click) << endl;
#endif

        return true;
    }
    else
    {
        if (info > 0) cout << "The algorithm failed to converge!" << endl;
        else cout << "The " << -info << " parameter is illegal!" << endl;

        return false;
    }
}

/*heevr returns from the biggest to the smallest, but lapack function is from the smallest to the biggest*/

bool heevr(const BLASINT &dim, double* matrix, const BLASINT &nv, const double &abstol, double* eigenvalue, double* U)	
{
#ifdef DEBUG_HEEV_TIME
    long click = sysconf(_SC_CLK_TCK);
    static struct tms tms_begin_heevr;
    int time_begin_heevr = times(&tms_begin_heevr);
#endif

    const char jobz = 'V';
    const char range = 'I';
    const char uplo = 'U';

    const double vl = 0.0;
    const double vu = 0.0;
    
    const BLASINT il = 1;
    const BLASINT iu = nv;

    BLASINT m;

    BLASINT* isuppz = new BLASINT[2*(1+nv)];
    assert(isuppz);

    const BLASINT lwork = 132*dim;

    double* work = new double[lwork];
    assert(work);

    const BLASINT liwork = 100*dim;

    BLASINT* iwork = new BLASINT[liwork];
    assert(iwork);

    BLASINT info;

    for(BLASINT i = 0; i < dim*dim; i++) matrix[i] = -matrix[i];

    dsyevr_(&jobz, &range, &uplo, &dim, matrix, &dim,  &vl, &vu, &il, &iu, &abstol, &m, eigenvalue, U, &dim, isuppz, work, &lwork, iwork, &liwork, &info);
    assert(m == nv);

    delete []iwork;

    delete []work;

    delete []isuppz;

    if (info == 0)
    {
        for (BLASINT i = 0; i < nv; i++) eigenvalue[i] = -eigenvalue[i];

#ifdef DEBUG_HEEV_TIME
        static struct tms tms_end_heevr;
        long time_end_heevr = times(&tms_end_heevr);
        cout.precision(10);
        cout<<"Time for ExactDiagonalization matrix with dimension " << dim << " is :" << (time_end_heevr-time_begin_heevr)/(1.0*click) << endl;
#endif

        return true;
    }
    else
    {
        if (info > 0) cout << "The algorithm in heevr failed to converge! info = " << info << endl;
        else cout << "The " << -info << " parameter is illegal!" << endl;

        return false;
    }
}

bool heevr(const BLASINT &dim, Complex* matrix, const BLASINT &lda, double* eigenvalue, const char &order)
{
#ifdef DEBUG_HEEV_TIME
    long click=sysconf(_SC_CLK_TCK);
    static struct tms tms_begin_heev;
    int time_begin_heev = times(&tms_begin_heev);
#endif

    assert(0);

    char jobz = 'V';
    char uplo = 'U';
    BLASINT info;
    BLASINT lwork = 2*dim;

    Complex* work = new Complex[lwork];
    assert(work);

    double* rwork = new double[3*dim];
    assert(rwork);

/*Exchange row and column of Matrix,
*for fortran and c in the order of matrix are different
*/

    for (BLASINT i = 0; i < dim*dim; i++) matrix[i] = conj(matrix[i]);

    if (order == 'D') for (BLASINT i = 0; i < dim*dim; i++) matrix[i] = -matrix[i];

    zheev_(&jobz, &uplo, &dim, matrix, &lda, eigenvalue, work, &lwork, rwork, &info);

    delete []rwork;

    delete []work;

    if (info == 0)
    {
        if (order == 'D') for (BLASINT i = 0; i < dim; i++) eigenvalue[i] = -eigenvalue[i];

#ifdef DEBUG_HEEV_TIME
        static struct tms tms_end_heev;
        int time_end_heev = times(&tms_end_heev);
        cout.precision(10);
        cout << "Time for ExactDiagonalization matrix with dim " << n << " is :" << (time_end_heev-time_begin_heev)/(1.0*click) << endl;
#endif

        return true;
    }
    else
    {
        if (info > 0) cout << "The algorithm failed to converge! info = " << info <<endl;
        else cout << "The " << -info << " parameter is illegal!" << endl;

        return false;
    }
}

bool heevx(const BLASINT &dim, double* matrix, const BLASINT &nv, const double &abstol, double* eigenvalue, double* U)	
{
#ifdef DEBUG_HEEV_TIME
    long click = sysconf(_SC_CLK_TCK);
    static struct tms tms_begin_heevr;
    int time_begin_heevr = times(&tms_begin_heevr);
#endif

    const char jobz = 'V';
    const char range = 'I';
    const char uplo = 'U';

    const double vl = 0.0;
    const double vu = 0.0;
    
    const BLASINT il = 1;
    const BLASINT iu = nv;

    BLASINT m;

    const BLASINT lwork = 132*dim;

    double* work = new double[lwork];
    assert(work);

    BLASINT* iwork = new BLASINT[5*dim];
    assert(iwork);

    BLASINT* ifail = new BLASINT[dim];
    assert(ifail);

    BLASINT info;

    for(BLASINT i = 0; i < dim*dim; i++) matrix[i] = -matrix[i];

    dsyevx_(&jobz, &range, &uplo, &dim, matrix, &dim,  &vl, &vu, &il, &iu, &abstol, &m, eigenvalue, U, &dim, work, &lwork, iwork, ifail, &info);
    assert(m == nv);

    delete []ifail;

    delete []iwork;

    delete []work;

    if (info == 0)
    {
        for (BLASINT i = 0; i < nv; i++) eigenvalue[i] = -eigenvalue[i];

#ifdef DEBUG_HEEV_TIME
        static struct tms tms_end_heevr;
        long time_end_heevr = times(&tms_end_heevr);
        cout.precision(10);
        cout<<"Time for ExactDiagonalization matrix with dimension " << dim << " is :" << (time_end_heevr-time_begin_heevr)/(1.0*click) << endl;
#endif

        return true;
    }
    else
    {
        if (info > 0) cout << "The algorithm in heevx failed to converge! info = " << info <<endl;
        else cout << "The " << -info << " parameter is illegal!" << endl;

        return false;
    }
}

bool geev(const char &jobvl, const char &jobvr, const BLASINT &n, Complex* a, const BLASINT &lda, Complex* w, Complex* vl, const BLASINT &ldvl, Complex* vr, const BLASINT &ldvr, Complex* work, const BLASINT &lwork, double* rwork, BLASINT &info)
{
    zgeev_(&jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work, &lwork, rwork, &info);

    return true;
}

bool geevx(const char &balanc, const char &jobvl, const char &jobvr, const char &sense, const BLASINT &n, Complex* a, const BLASINT &lda, Complex* w, Complex* vl, const BLASINT &ldvl, Complex* vr, const BLASINT &ldvr, BLASINT &ilo, BLASINT &ihi, double* scale, double &abnrm, double* rconde, double* rcondv, Complex* work, const BLASINT &lwork, double* rwork, BLASINT &info)
{
    zgeevx_(&balanc, &jobvl, &jobvr, &sense, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv, work, &lwork, rwork, &info);

    return true;
}

BLASINT gelqf(const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* tau, double* work, const BLASINT &lwork, BLASINT &info)
{
/*matrix transfered when from "C" to "FORTRAN"*/

    dgeqrf_(&n, &m, matrix, &lda, tau, work, &lwork, &info);

    return info; 
}

BLASINT gelqf(const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* tau, Complex* work, const BLASINT &lwork, BLASINT &info)
{
    zgeqrf_(&n, &m, matrix, &lda, tau, work, &lwork, &info);

    return info;
}

BLASINT geqrf(const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* tau, double* work, const BLASINT &lwork, BLASINT &info)
{
    dgelqf_(&n, &m, matrix, &lda, tau, work, &lwork, &info);

    return info;
}

BLASINT geqrf(const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* tau, Complex* work, const BLASINT &lwork, BLASINT &info)
{
    zgelqf_(&n, &m, matrix, &lda, tau, work, &lwork, &info);

    return info;
}

BLASINT orglq(const BLASINT &m, const BLASINT &n, const BLASINT &k, double* matrix, const BLASINT &lda, double* tau, double* work, const BLASINT &lwork, BLASINT &info)
{
    dorgqr_(&n, &m, &k, matrix, &lda, tau, work, &lwork, &info);

    return info;
}

BLASINT orglq(const BLASINT &m, const BLASINT &n, const BLASINT &k, Complex* matrix, const BLASINT &lda, double* tau, Complex* work, const BLASINT &lwork, BLASINT &info)
{
    zungqr_(&n, &m, &k, matrix, &lda, tau, work, &lwork, &info);

    return info;
}

BLASINT orgqr(const BLASINT &m, const BLASINT &n, const BLASINT &k, double* matrix, const BLASINT &lda, double* tau, double *work, const BLASINT &lwork, BLASINT &info)
{
    dorglq_(&n, &m, &k, matrix, &lda, tau, work, &lwork, &info);

    return info;
}

BLASINT orgqr(const BLASINT &m, const BLASINT &n, const BLASINT &k, Complex* matrix, const BLASINT &lda, double* tau, Complex *work, const BLASINT &lwork, BLASINT &info)
{
    zunglq_(&n, &m, &k, matrix, &lda, tau, work, &lwork, &info);

    return info;
}

BLASINT gesvd(const char &jobu, const char &jobvt, const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* s, double* u, const BLASINT &ldu, double* vt, const BLASINT &ldvt, double* work, const BLASINT &lwork, double* rwork, BLASINT &info)
{
    dgesvd_(&jobvt, &jobu, &n, &m, matrix, &lda, s, vt, &ldvt, u, &ldu, work, &lwork, &info);

    return info;
}

BLASINT gesvd(const char &jobu, const char &jobvt, const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* s, Complex* u, const BLASINT &ldu, Complex* vt, const BLASINT &ldvt, Complex* work, const BLASINT &lwork, double* rwork, BLASINT &info)
{
    zgesvd_(&jobvt, &jobu, &n, &m, matrix, &lda, s, vt, &ldvt, u, &ldu, work, &lwork, rwork, &info);

    return info;
}

BLASINT gesvdx(const char &jobu, const char &jobvt, const char &range, const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, const double &vl, const double &vu, const BLASINT &il, const BLASINT &iu, BLASINT &ns, double* s, double* u, const BLASINT &ldu, double* vt, const BLASINT &ldvt, double* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info)
{
    dgesvdx_(&jobvt, &jobu, &range, &n, &m, matrix, &lda, &vl, &vu, &il, &iu, &ns, s, vt, &ldvt, u, &ldu, work, &lwork, iwork, &info);

    return info;
}

BLASINT gesvdx(const char &jobu, const char &jobvt, const char &range, const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, const double &vl, const double &vu, const BLASINT &il, const BLASINT &iu, BLASINT &ns, double* s, Complex* u, const BLASINT &ldu, Complex* vt, const BLASINT &ldvt, Complex* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info)
{
    zgesvdx_(&jobvt, &jobu, &range, &n, &m, matrix, &lda, &vl, &vu, &il, &iu, &ns, s, vt, &ldvt, u, &ldu, work, &lwork, rwork, iwork, &info);
	
    return info;	
}

BLASINT gesdd(const char &jobz, const BLASINT &m, const BLASINT &n, double* matrix, const BLASINT &lda, double* s, double* u, const BLASINT &ldu, double* vt, const BLASINT &ldvt, double* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info)
{
    dgesdd_(&jobz, &n, &m, matrix, &lda, s, vt, &ldvt, u, &ldu, work, &lwork, iwork, &info); 	

    return info;
}

BLASINT gesdd(const char &jobz, const BLASINT &m, const BLASINT &n, Complex* matrix, const BLASINT &lda, double* s, Complex* u, const BLASINT &ldu, Complex* vt, const BLASINT &ldvt, Complex* work, const BLASINT &lwork, double* rwork, BLASINT* iwork, BLASINT &info)
{
    zgesdd_(&jobz, &n, &m, matrix, &lda, s, vt, &ldvt, u, &ldu, work, &lwork, rwork, iwork, &info);   

    return info; 
}

