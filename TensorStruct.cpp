#include <assert.h>

#include "TensorStruct.h"

TensorStruct::TensorStruct(const TensorStruct &ts)
{
    assert(setTensorStruct(ts));
}

TensorStruct::TensorStruct(const uint32_t &tr, const BLASINT* rd)
{
    assert(setTensorStruct(tr, rd));
}

TensorStruct::TensorStruct(std::initializer_list<BLASINT> rd)
{
    assert(setTensorStruct(rd));
}

TensorStruct::TensorStruct(const TensorStruct &ts, const uint32_t &idx)
{
    assert(setTensorStruct(ts, idx));
}

TensorStruct::TensorStruct(const TensorStruct &tsl, const uint32_t &il, const TensorStruct &tsr, const uint32_t &ir, const char &cttype)
{
    assert(setTensorStruct(tsl, il, tsr, ir, cttype));
}

TensorStruct::TensorStruct(const TensorStruct &ts, const uint32_t &il, const uint32_t &ir)
{
    assert(setTensorStruct(ts, il, ir));
}

bool TensorStruct::reshape(std::initializer_list<BLASINT> rd)
{
    BLASINT trialsize = 1;	
    for (auto p = rd.begin(); p < rd.end(); p++) 
    {
	if (*p <= 0) return false;   

	trialsize *= *p;
    }

    if (trialsize == dataSize)
    {
        tensorRank = rd.size();
	uint32_t i = 0;
        for (auto p = rd.begin(); p < rd.end(); p++, i++) rankDim[i] = *p;
        
        return true;	
    }
    else return false;
}

/*several sequentional index are combined into one, or one index is decomposited into several sequential index*/

bool TensorStruct::rankCombination(const uint32_t &rstart, const uint32_t &rnum)
{
    if (rnum == 0 || rstart + rnum > tensorRank || rstart >= tensorRank) return false;    

    if (rnum == 1) return true;

    BLASINT dcom = 1;
    
    for (uint32_t i = rstart; i < rstart+rnum; i++) dcom *= rankDim[i]; 

    rankDim[rstart] = dcom;

    for (uint32_t i = rstart+rnum; i < tensorRank; i++) rankDim[i-rnum+1] = rankDim[i];

    tensorRank -= (rnum-1);

    return true;
}

bool TensorStruct::rankDecomposition(const uint32_t &rindex, std::initializer_list<BLASINT> rd)
{
    if (rd.size() == 0 || rindex >= tensorRank) return false;

    BLASINT dcom = 1;
    for (auto p = rd.begin(); p < rd.end(); p++) 
    {
	if (*p <= 0) return false;    
	dcom *= *p;
    }

    if (dcom != rankDim[rindex]) return false; 

    for (BLASINT i = tensorRank+rd.size()-2; i >= rindex+rd.size(); i--) rankDim[i] = rankDim[i+1-rd.size()]; 

    uint32_t i = 0;
    for (auto p = rd.begin(); p < rd.end(); p++, i++) rankDim[rindex+i] = *p; 

    tensorRank = tensorRank + rd.size() - 1;

    return true;
}

void TensorStruct::setDataSize()
{
    dataSize = 1;
    for (uint32_t i = 0; i < tensorRank; i++) dataSize *= rankDim[i];    
}

/*protected*/

bool TensorStruct::setTensorStruct(const TensorStruct &ts)
{
    tensorRank = ts.getTensorRank();
    for (uint32_t i = 0; i < tensorRank; i++) rankDim[i] = ts.getRankDim(i);
    dataSize = ts.getDataSize();
    return true;    	
}

bool TensorStruct::setTensorStruct(const uint32_t &tr, const BLASINT* rd)
{
    for (uint32_t i = 0; i < tr; i++)
    {
        if (rd[i] <= 0) return false;
    }

    tensorRank = 0;
    dataSize = 1;
    for (uint32_t i = 0; i < tr; i++)
    {
        rankDim[i] = rd[i];
        dataSize *= rankDim[i];
        tensorRank++;
    }

    return true;
}

bool TensorStruct::setTensorStruct(std::initializer_list<BLASINT> rd)
{
    for (auto p = rd.begin(); p < rd.end(); p++)
    {
        if (*p <= 0) return false;
    }

    tensorRank = 0;
    dataSize = 1;

    for (auto p = rd.begin(); p < rd.end(); p++)
    {
        rankDim[tensorRank] = *p;
        dataSize *= *p;
        tensorRank++;
    }

    return true;
}

bool TensorStruct::setTensorStruct(const TensorStruct &ts, const uint32_t &idx)
{
    assert(idx < ts.getTensorRank());

    tensorRank = 0;
    dataSize = 1;

    for (uint32_t i = 0; i < ts.getTensorRank(); i++)
    {
        if (i != idx)
        {
            rankDim[tensorRank] = ts.getRankDim(i);
            dataSize *= rankDim[tensorRank];
            tensorRank++;
        }
    }

    assert(tensorRank+1 == ts.getTensorRank());

    for (uint32_t i = 0; i < tensorRank; i++) rankDim[tensorRank+i] = rankDim[i];

    tensorRank *= 2;
    dataSize = dataSize*dataSize;

    return true;
}

bool TensorStruct::setTensorStruct(const TensorStruct &tsl, const uint32_t &il, const TensorStruct &tsr, const uint32_t &ir, const char &cttype)
{
    assert(cttype == 'S' || cttype == 'D' || cttype == 'I');

    if (cttype == 'S') //sequential contraction
    {	    
        assert(tsl.getTensorRank() >= 1 && tsr.getTensorRank() >= 1);

        tensorRank = tsl.getTensorRank() + tsr.getTensorRank() - 2;

        for (uint32_t i = 0; i < il; i++) rankDim[i] = tsl.getRankDim(i);
        for (uint32_t i = il; i+1 < tsl.getTensorRank(); i++) rankDim[i] = tsl.getRankDim(i+1);   

        for (uint32_t i = 0; i < ir; i++) rankDim[tsl.tensorRank-1+i] = tsr.getRankDim(i);
        for (uint32_t i = ir; i+1 < tsr.getTensorRank(); i++) rankDim[tsl.getTensorRank()-1+i] = tsr.getRankDim(i+1);

        dataSize = 1;
        for (uint32_t i = 0; i < tensorRank; i++) dataSize *= rankDim[i];
    }
    else if (cttype == 'D') // direct product contraction
    {
	assert(tsl.getTensorRank() >= 1 && tsr.getTensorRank() >= 1);

        tensorRank = tsl.getTensorRank() + tsr.getTensorRank() - 2;

        for (uint32_t i = 0; i < il; i++) rankDim[i] = tsl.getRankDim(i);
        for (uint32_t i = 0; i < ir; i++) rankDim[i+il] = tsr.getRankDim(i);

        for (uint32_t i = il+1; i < tsl.getTensorRank(); i++) rankDim[ir+i-1] = tsl.getRankDim(i);
        for (uint32_t i = ir+1; i < tsr.getTensorRank(); i++) rankDim[tsl.getTensorRank()-1+i-1] = tsr.getRankDim(i);

        dataSize = 1;
        for (uint32_t i = 0; i < tensorRank; i++) dataSize *= rankDim[i];
    }
    else if (cttype == 'I')
    {
	assert(tsl.getTensorRank() >= 1 && tsr.getTensorRank() >= 1);

        tensorRank = tsl.getTensorRank() + tsr.getTensorRank() - 2;

	for (uint32_t i = 0; i < il; i++) rankDim[i] = tsl.getRankDim(i);
        for (uint32_t i = 0; i < ir; i++) rankDim[i+il] = tsr.getRankDim(i);

	for (uint32_t i = ir+1; i < tsr.getTensorRank(); i++) rankDim[il+i-1] = tsr.getRankDim(i);
        for (uint32_t i = il+1; i < tsl.getTensorRank(); i++) rankDim[tsr.getTensorRank()-1+i-1] = tsl.getRankDim(i);

        dataSize = 1;
        for (uint32_t i = 0; i < tensorRank; i++) dataSize *= rankDim[i];
    }
    else
    {
	assert(0);    
    }

    return true;
}

bool TensorStruct::setTensorStruct(const TensorStruct &ts, const uint32_t &il, const uint32_t &ir)
{
    assert(ts.getTensorRank() >= 2);
    assert(il < ir);

    tensorRank = ts.getTensorRank() - 2;

    for (uint32_t i = 0; i < il; i++) rankDim[i] = ts.getRankDim(i);

    for (uint32_t i = il+1; i < ir; i++) rankDim[i-1] = ts.getRankDim(i);

    for (uint32_t i = ir+1; i < ts.getTensorRank(); i++) rankDim[i-2] = ts.getRankDim(i);

    dataSize = 1;
    for (uint32_t i = 0; i < tensorRank; i++) dataSize *= rankDim[i];

    return true;
}
