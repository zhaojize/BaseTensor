/* SquareMatrix  
 * This library is distributed WITHOUT ANY WARRANTY.
 * Author:  Zhao Jize <zhaojize@outlook.com>
 * All copyright reserved by the author.
*/

/* 1, 5, 9, 13, 17, 21*/

/*v0.1*/

#if !defined _OPTMATRIX_H
#define _OPTMATRIX_H

#include <iostream>

#include "globalconfig.h"
#include "cppblas.h"
#include "cppblasex.h"
#include "cpplapack.h"
#include "TypeCompatible.h"
#include "BaseTensor.h"

template<typename DTTYPE> 
class SquareMatrix : public BaseTensor<DTTYPE> 
{
    public:
/********************constructor and destructor*********************/

/*read from file*/

        SquareMatrix(const char* filename, const bool &ifremoved, const BLASINT &readtype = 0);

/*read from stream*/

        SquareMatrix(std::ifstream &ifs);

/*initialize tensor from raw data*/

        template<typename T = DTTYPE> SquareMatrix(const BLASINT &dim, const T* data = nullptr, const BLASINT &dtsize = 0);

/*from rank-2 tensor with equal rank-0 and rank-1 to squarematrix*/

	SquareMatrix(const SquareMatrix &smt, const DTTYPE &scale = 1.0);

	SquareMatrix(SquareMatrix &&smt, const DTTYPE &scale = 1.0);

	SquareMatrix(const BaseTensor<DTTYPE> &bt, const DTTYPE &scale = 1.0);

	SquareMatrix(BaseTensor<DTTYPE> &&bt, const DTTYPE &scale = 1.0);

/*change the size*/	

	inline bool resize(const BLASINT &dim) {return BaseTensor<DTTYPE>::resize({dim, dim});}

	SquareMatrix &operator *= (const DTTYPE &s);
        SquareMatrix &operator /= (const DTTYPE &s);

        void expH(const typename istc::SVDReal<DTTYPE>::type &scale, void* tmpdata, const BLASINT &tdize);

    private:
        template<typename T = DTTYPE> SquareMatrix(std::initializer_list<BLASINT> rd, const T* data = nullptr, const BLASINT &dtsize = 0) = delete;

        bool resize(initializer_list<BLASINT> rd) = delete;

        bool reshape(void) = delete;

	void highOrderLSVD(void) = delete;

        void highOrderLSVDx(void) = delete;	
};

template <typename DTTYPE> 
SquareMatrix<DTTYPE>::SquareMatrix(const char* filename, const bool &ifremoved, const BLASINT &readtype) : BaseTensor<DTTYPE>(filename, ifremoved, readtype)
{
    assert(this->getTensorRank() == 2);	
    assert(this->getRankDim(0) == this->getRankDim(1));    
}

/*read from stream*/

template <typename DTTYPE> 
SquareMatrix<DTTYPE>::SquareMatrix(ifstream &ifs) : BaseTensor<DTTYPE> (ifs)
{
    assert(this->getTensorRank() == 2);	
    assert(this->getRankDim(0) == this->getRankDim(1));
}

template<typename DTTYPE> template<typename T>
SquareMatrix<DTTYPE>::SquareMatrix(const BLASINT &dim, const T* data, const BLASINT &dtsize) : BaseTensor<DTTYPE>({dim, dim}, data, dtsize)
{
}

/*copy and move: from rank-2 tensor with equal rank-0 and rank-1 to squarematrix*/

template <typename DTTYPE> 
SquareMatrix<DTTYPE>::SquareMatrix(const SquareMatrix &smt, const DTTYPE &scale) : BaseTensor<DTTYPE>(smt, scale)
{
    assert(this->getTensorRank() == 2);	
    assert(this->getRankDim(0) == this->getRankDim(1)); 	
}

template <typename DTTYPE> 
SquareMatrix<DTTYPE>::SquareMatrix(SquareMatrix &&smt, const DTTYPE &scale) : BaseTensor<DTTYPE>(std::move(smt), scale)
{
    assert(this->getTensorRank() == 2);	
    assert(this->getRankDim(0) == this->getRankDim(1)); 
}

template<typename DTTYPE>
SquareMatrix<DTTYPE>::SquareMatrix(const BaseTensor<DTTYPE> &bt, const DTTYPE &scale) : BaseTensor<DTTYPE>(bt, scale)
{
    assert(this->getTensorRank() == 2);	
    assert(this->getRankDim(0) == this->getRankDim(1)); 
}

template<typename DTTYPE>
SquareMatrix<DTTYPE>::SquareMatrix(BaseTensor<DTTYPE> &&bt, const DTTYPE &scale) : BaseTensor<DTTYPE>(std::move(bt), scale)
{
    assert(this->getTensorRank() == 2);	
    assert(bt.getRankDim(0) == bt.getRankDim(1));    
}

template<typename DTTYPE> 
SquareMatrix<DTTYPE> &SquareMatrix<DTTYPE>::operator *= (const DTTYPE &s)
{
    if (s == 1.0) return *this;

    scal(this->dataSize, s, this->tensorData, 1);

    return *this;
}

template<typename DTTYPE> 
SquareMatrix<DTTYPE> &SquareMatrix<DTTYPE>::operator /= (const DTTYPE &s)
{
    if (s == 1.0) return *this;

    scal(this->dataSize, 1.0/s, this->tensorData, 1);
  
    return *this;
}

template<typename DTTYPE>
void SquareMatrix<DTTYPE>::expH(const typename istc::SVDReal<DTTYPE>::type &scale, void* tmpdata, const BLASINT &tdsize)
{
    BLASINT dim = this->getRankDim(0);

    using RDTTYPE = typename istc::SVDReal<DTTYPE>::type; 

    assert(tdsize >= dim*sizeof(RDTTYPE)+dim*dim*sizeof(DTTYPE));

    RDTTYPE* egv = reinterpret_cast<RDTTYPE*> (tmpdata);

    heev(dim, this->tensorData, dim, egv, 'D');

    scal(dim, scale, egv, 1);

    for (BLASINT i = 0; i < dim; i++) egv[i] = exp(egv[i]);

    DTTYPE* lmbdU = reinterpret_cast<DTTYPE*> (egv+dim);

    memcpy(lmbdU, this->tensorData, dim*dim*sizeof(DTTYPE));

    for (BLASINT i = 0; i < dim; i++)
    {
	for (BLASINT j = 0; j < dim; j++)
	{
	    DATATYPE sum = 0.0;
            for (BLASINT k = 0; k < dim; k++) sum += lmbdU[k*dim+i]*egv[k]*conj(lmbdU[k*dim+j]); 	    
	    this->tensorData[i*dim+j] = sum; 
	}	
    }
}
#endif
