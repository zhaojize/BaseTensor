#ifndef __MIO_LAPACK_H
#define __MIO_LAPACK_H

#include "cpplapack.h"
#include "TypeCompatible.h"

namespace miolpk
{
/*begin of namespace*/

template<typename DTTYPE>
BLASINT miogesvd(const char &jobu, const char &jobvt, const BLASINT &m, const BLASINT &n, DTTYPE* matrix, const BLASINT &lda, typename istc::SVDReal<DTTYPE>::type* s, DTTYPE* u, const BLASINT &ldu, DTTYPE* vt, const BLASINT &ldvt, BLASINT &info, void* pool, const BLASINT &poolsize)	
{
    using RDTTYPE = typename istc::SVDReal<DTTYPE>::type;

    BLASINT rwksize = 0;
    if (typeid(DTTYPE) != typeid(RDTTYPE)) rwksize = 5 * (m < n ? m : n); 
    RDTTYPE* rworksvd = reinterpret_cast<RDTTYPE*> (pool);
            
    DTTYPE* worksvd = reinterpret_cast<DTTYPE*> (rworksvd+rwksize); 

    gesvd(jobu, jobvt, m, n, matrix, lda, s, u, ldu, vt, ldvt, worksvd, -1, rworksvd, info);

    BLASINT lwork = static_cast<BLASINT>(real(worksvd[0]))+8;

    assert(poolsize > rwksize*sizeof(RDTTYPE)+lwork*sizeof(DTTYPE));

    gesvd(jobu, jobvt, m, n, matrix, lda, s, u, ldu, vt, ldvt, worksvd, lwork, rworksvd, info);

    return info;
}	

template<typename DTTYPE>
BLASINT miogesvdx(const char &jobu, const char &jobvt, const char &range, const BLASINT &m, const BLASINT &n, DTTYPE* matrix, const BLASINT &lda, const typename istc::SVDReal<DTTYPE>::type &vl, const typename istc::SVDReal<DTTYPE>::type &vu, const BLASINT &il, const BLASINT &iu, BLASINT &ns, typename istc::SVDReal<DTTYPE>::type* s, DTTYPE* u, const BLASINT &ldu, DTTYPE* vt, const BLASINT &ldvt, BLASINT &info, void* pool, const BLASINT &poolsize)
{
    using RDTTYPE = typename istc::SVDReal<DTTYPE>::type;

    BLASINT mindim = m < n ? m : n;

    BLASINT rwksize = 0;  //real
    if (typeid(DTTYPE) != typeid(RDTTYPE)) rwksize = mindim*((mindim+12));
    RDTTYPE* rwork = reinterpret_cast<RDTTYPE*>(pool);

    BLASINT iwksize = 12*mindim;
    BLASINT* iwork = reinterpret_cast<BLASINT*>(rwork+rwksize);

    DTTYPE* work = reinterpret_cast<DTTYPE*>(iwork+iwksize); 

    gesvdx(jobu, jobvt, range, m, n, matrix, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, work, -1, rwork, iwork, info);

    BLASINT lwork = (BLASINT)real(work[0])+8;

    assert(poolsize >= rwksize*sizeof(RDTTYPE)+iwksize*sizeof(BLASINT)+lwork*sizeof(DTTYPE));

    gesvdx(jobu, jobvt, range, m, n, matrix, lda, vl, vu, il, iu, ns, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);

    return info;
}

template<typename DTTYPE>
BLASINT miogesdd(const char &jobz, const BLASINT &m, const BLASINT &n, DTTYPE* matrix, const BLASINT &lda, typename istc::SVDReal<DTTYPE>::type* s, DTTYPE* u, const BLASINT &ldu, DTTYPE* vt, const BLASINT &ldvt, BLASINT &info, void* pool, const BLASINT &poolsize)
{
    using RDTTYPE = typename istc::SVDReal<DTTYPE>::type;

    const BLASINT mx = m > n ? m : n;
    const BLASINT mn = m > n ? n : m;
    const BLASINT mlgn = 5 * mn * (mn+1);
    const BLASINT meqn = 2*mx*mn + 2*mn*mn + mn;

    BLASINT iworksize = 9*mn;
    BLASINT* iwork = reinterpret_cast<BLASINT*>(pool);

    BLASINT rwksize = 0;
    if (typeid(DTTYPE) != typeid(RDTTYPE)) rwksize = mlgn > meqn ? mlgn : meqn;
    RDTTYPE* rworksvd = reinterpret_cast<RDTTYPE*> (iwork+iworksize);

    DTTYPE* worksvd = reinterpret_cast<DTTYPE*> (rworksvd+rwksize);

    gesdd(jobz, m, n, matrix, lda, s, u, ldu, vt, ldvt, worksvd, -1, rworksvd, iwork, info);
    assert(info == 0);

    BLASINT lwork = static_cast<BLASINT>(real(worksvd[0]))+8;

    assert(poolsize > rwksize*sizeof(RDTTYPE)+lwork*sizeof(DTTYPE)+iworksize*sizeof(BLASINT));

    gesdd(jobz, m, n, matrix, lda, s, u, ldu, vt, ldvt, worksvd, lwork, rworksvd, iwork, info);

    return info;
}

/*end of namespace*/
}
#endif
