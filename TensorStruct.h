#ifndef __TENSOR_STRUCT_H
#define __TENSOR_STRUCT_H

#include <fstream>
#include <initializer_list>
#include <cassert>

#include "globalconfig.h"

class TensorStruct
{
    protected:	
        uint32_t tensorRank;

        BLASINT rankDim[MAXTRK];   

        BLASINT dataSize;

    public:
	TensorStruct(){};

	TensorStruct(const TensorStruct &ts);

	TensorStruct(const uint32_t &tr, const BLASINT* rd);

	TensorStruct(std::initializer_list<BLASINT> rd);

        TensorStruct(const TensorStruct &ts, const uint32_t &idx);

        TensorStruct(const TensorStruct &tsl, const uint32_t &il, const TensorStruct &tsr, const uint32_t &ir, const char &cttype);

	TensorStruct(const TensorStruct &ts, const uint32_t &il, const uint32_t &ir);

    public:	
/*information*/

        inline uint32_t getTensorRank() const {return tensorRank;}

	inline BLASINT getRankDim(const uint32_t &idx) const {return rankDim[idx];}

	inline BLASINT getDataSize() const {return dataSize;}
   
/*change the shape, others are kept*/

	bool reshape(std::initializer_list<BLASINT> rd);

        bool rankCombination(const uint32_t &rstart, const uint32_t &rnum);

        bool rankDecomposition(const uint32_t &rindex, std::initializer_list<BLASINT>rd); 

    public:  // will be protected
	inline void setTensorRank(const uint32_t &tr) {tensorRank = tr;}

	inline void setRankDim(const uint32_t &idx, const BLASINT &dim) {rankDim[idx] = dim;}

        void setDataSize();

    protected:
        bool setTensorStruct(const TensorStruct &ts);	

        bool setTensorStruct(const uint32_t &tr, const BLASINT* rd);

        bool setTensorStruct(std::initializer_list<BLASINT> rd);	

	bool setTensorStruct(const TensorStruct &ts, const uint32_t &idx);

        bool setTensorStruct(const TensorStruct &tsl, const uint32_t &il, const TensorStruct &tsr, const uint32_t &ir, const char &cttype);

        bool setTensorStruct(const TensorStruct &ts, const uint32_t &il, const uint32_t &ir);
};
#endif
