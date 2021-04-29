#ifndef __LLM_OPERATION_H
#define __LLM_OPERATION_H

#include <cstring>
#include <cassert>

#include "globalconfig.h"

namespace llmopr    
{
/*begin of namespace*/

template<typename DTTYPE>
inline void transposeTo(const BLASINT &dsrow, const BLASINT &dscol, const DTTYPE* sour, const BLASINT &ldsr, DTTYPE* dest, const BLASINT &lddt)
{
    for (BLASINT i = 0; i < dsrow; i++)
    {
        for (BLASINT j = 0; j < dscol; j++) dest[j*lddt+i] = sour[i*ldsr+j];
    }	    
}

template<typename DTTYPE>
void fastTransposeTo(const BLASINT &dsrow, const BLASINT &dscol, const DTTYPE* sour, const BLASINT &ldsr, DTTYPE* dest, const BLASINT &lddt)
{
    const BLASINT BSIZE = 16;

    const BLASINT bdr = (dsrow/BSIZE)*BSIZE;
    const BLASINT bdc = (dscol/BSIZE)*BSIZE;

    for (BLASINT i = 0; i < bdr; i += BSIZE) 
    {
        for(int j = 0; j < bdc; j += BSIZE) transposeTo(BSIZE, BSIZE, &sour[i*ldsr+j], ldsr, &dest[j*lddt+i], lddt);
    }

    for (BLASINT i = 0; i < bdr; i += BSIZE) transposeTo(BSIZE, dscol-bdc, &sour[i*ldsr+bdc], ldsr, &dest[bdc*lddt+i], lddt); 

    for (BLASINT j = 0; j < bdc; j += BSIZE) transposeTo(dsrow-bdr, BSIZE, &sour[bdr*ldsr+j], ldsr, &dest[j*lddt+bdr], lddt);

    transposeTo(dsrow-bdr, dscol-bdc, &sour[bdr*ldsr+bdc], ldsr, &dest[bdc*lddt+bdr], lddt);
}

template<typename DTTYPE>
void transposeOnsite(const BLASINT &drow, const BLASINT &dcol, DTTYPE* matrix, DTTYPE* tmpdata)
{
    memcpy(tmpdata, matrix, drow*dcol*sizeof(DTTYPE));

    fastTransposeTo(drow, dcol, tmpdata, dcol, matrix, drow);
}

template<typename DTTYPE>
void transposeOnsite(const BLASINT &drow, const BLASINT &dcol, DTTYPE* matrix)
{
    if (drow == dcol)
    {
	const BLASINT dim = dcol;   

	for (BLASINT i = 0; i < dim; i++)    
	{
	    for (BLASINT j = 0; j < dim; j++)	
	    {
	        DTTYPE tmp = matrix[i*dim+j];
                matrix[i*dim+j] = matrix[j*dim+i];
                matrix[j*dim+i] = tmp;		
	    }
	}
    }
    else
    {
	assert(0);    
    }
}

/*CO = cache oblivious*/

template<typename DTTYPE>
void transposeCO(const BLASINT &dsrow, const BLASINT &dscol, const DTTYPE* sour, const BLASINT &ldsr, DTTYPE* dest, const BLASINT &lddt)
{
    if (dscol < 32) transposeTo(dsrow, dscol, sour, ldsr, dest, lddt);
    else
    {
        if (dscol >= dsrow)	
	{
	    transposeCO(dsrow, dscol/2, sour, ldsr, dest, lddt);
            transposeCO(dsrow, dscol-dscol/2, sour+dscol/2, ldsr, dest+(dscol/2)*lddt, lddt);	    
	}	
	else
	{
	    transposeCO(dsrow/2, dscol, sour, ldsr, dest, lddt);
            transposeCO(dsrow-dsrow/2, dscol, sour+(dsrow/2)*ldsr, ldsr, dest+dsrow/2, lddt);	    
	}
    }    
}

/*ijk->jik*/

template<typename DTTYPE>
void transposeLMto(const BLASINT &diml, const BLASINT &dimm, const BLASINT &dimr, const DTTYPE* sour, DTTYPE* dest)
{
    for (BLASINT i = 0; i < diml; i++)
    {
        for (BLASINT j = 0; j < dimm; j++)
        {
            for (BLASINT k = 0; k < dimr; k++) dest[(j*diml+i)*dimr+k] = sour[(i*dimm+j)*dimr+k]; 
        }  
    }
}

/*ijk->ikj*/

template<typename DTTYPE>
void transposeMRto(const BLASINT &diml, const BLASINT &dimm, const BLASINT &dimr, const DTTYPE* sour, DTTYPE* dest)
{
    for (BLASINT i = 0; i < diml; i++) llmopr::transposeCO(dimm, dimr, sour+i*dimm*dimr, dimr, dest+i*dimr*dimm, dimm);	
}

/*end of namespace*/
}
#endif
