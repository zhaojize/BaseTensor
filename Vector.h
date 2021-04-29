/* Vector  
 * This library is distributed WITHOUT ANY WARRANTY.
 * Author:  Zhao Jize <zhaojize@outlook.com>
 * All copyright reserved by the author.
*/

/* 1, 5, 9, 13, 17, 21*/

/*v0.1*/

#if !defined _VECTOR_H
#define _VECTOR_H

#include <assert.h>

#include "globalconfig.h"
#include "cppblas.h"
#include "cppblasex.h"
#include "cpplapack.h"
#include "TypeCompatible.h"
#include "BaseTensor.h"

template<typename DTTYPE> 
class Vector : public BaseTensor<DTTYPE> 
{
    public:
/********************constructor and destructor*********************/

/*read from file*/

        Vector(const char* filename, const bool &ifremoved, const BLASINT &readtype = 0);

/*read from stream*/

        Vector(std::ifstream &ifs);

/*initialize tensor from raw data*/

        template<typename T = DTTYPE> Vector(const BLASINT &dim, const T* data = nullptr, const BLASINT &dtsize = 0);

/*copy or move : from rank-1 tensor to Vector*/

        Vector(const Vector &vt, const DTTYPE &scale = 1.0);

        Vector(Vector &&vt, const DTTYPE &scale = 1.0);

	Vector(const BaseTensor<DTTYPE> &bt, const DTTYPE &scale = 1.0);

	Vector(BaseTensor<DTTYPE> &&bt, const DTTYPE &scale = 1.0);

/*change the size*/	

	inline bool resize(initializer_list<BLASINT> rd)
	{
	    if (rd.size() != 1)
	    {
		assert(0);	    
		std::cout << "Error: parameter error in vector resize" << std::endl;    
		return false;
	    }	
	    else return BaseTensor<DTTYPE>::resize(rd);
        }

        inline bool truncateRankDim(const BLASINT &resdim) {return BaseTensor<DTTYPE>::truncateRankDim(0, resdim);}
        
        Vector &operator = (const Vector &vt); 
        Vector &operator = (const BaseTensor<DTTYPE> &bt); 
        Vector &operator *= (const DTTYPE &s);
        Vector &operator /= (const DTTYPE &s);

    private:
        bool reshape(initializer_list<BLASINT> rd) = delete;

        void leftSVD(void) = delete;

	void rightSVD(void) = delete;

	void highOrderLSVD(void) = delete;

        void highOrderLSVDx(void) = delete;	
};

template <typename DTTYPE> 
Vector<DTTYPE>::Vector(const char* filename, const bool &ifremoved, const BLASINT &readtype) : BaseTensor<DTTYPE>(filename, ifremoved, readtype)
{
    assert(this->getTensorRank() == 1);	
}

/*read from stream*/

template <typename DTTYPE> 
Vector<DTTYPE>::Vector(ifstream &ifs) : BaseTensor<DTTYPE> (ifs)
{
    assert(this->getTensorRank() == 1);	
}

template<typename DTTYPE> template<typename T> 
Vector<DTTYPE>::Vector(const BLASINT &dim, const T* data, const BLASINT &dtsize) : BaseTensor<DTTYPE>({dim}, data, dtsize)
{
}

template <typename DTTYPE> 
Vector<DTTYPE>::Vector(const Vector &vt, const DTTYPE &scale) : BaseTensor<DTTYPE>(vt, scale)
{
    assert(this->getTensorRank() == 1);	
}

template <typename DTTYPE> 
Vector<DTTYPE>::Vector(Vector &&vt, const DTTYPE &scale) : BaseTensor<DTTYPE>(std::forward<Vector<DTTYPE>>(vt), scale)
{
    assert(this->getTensorRank() == 1);		
}

template<typename DTTYPE>
Vector<DTTYPE>::Vector(const BaseTensor<DTTYPE> &bt, const DTTYPE &scale) : BaseTensor<DTTYPE>(bt, scale)
{
    assert(bt.getTensorRank() == 1); 
}

template<typename DTTYPE>
Vector<DTTYPE>::Vector(BaseTensor<DTTYPE> &&bt, const DTTYPE &scale) : BaseTensor<DTTYPE>(std::forward<BaseTensor<DTTYPE>>(bt), scale)
{
    assert(bt.getTensorRank() == 1);	
}

/*************end of constructor*************/


template<typename DTTYPE>
Vector<DTTYPE> &Vector<DTTYPE>::operator = (const Vector &vt)
{
    assert(this->resize({vt.getDataSize()}));

    memcpy(this->tensorData, vt.tensorData, this->getDataSize()*sizeof(DTTYPE));
    
    return *this; 
}

template<typename DTTYPE>
Vector<DTTYPE> &Vector<DTTYPE>::operator = (const BaseTensor<DTTYPE> &bt)
{
    assert(bt.getTensorRank() == 1);

    assert(this->resize({bt.getDataSize()}));

    memcpy(this->tensorData, bt.tensorData, this->getDataSize()*sizeof(DTTYPE));
    
    return *this; 
}

template<typename DTTYPE> 
Vector<DTTYPE> &Vector<DTTYPE>::operator *= (const DTTYPE &s)
{
    if (s == 1.0) return *this;

    scal(this->dataSize, s, this->tensorData, 1);

    return *this;
}

template<typename DTTYPE> 
Vector<DTTYPE> &Vector<DTTYPE>::operator /= (const DTTYPE &s)
{
    if (s == 1.0) return *this;

    scal(this->dataSize, 1.0/s, this->tensorData, 1);
  
    return *this;
}
#endif
