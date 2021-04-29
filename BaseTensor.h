/* BaseTensor  
 * This library is distributed WITHOUT ANY WARRANTY.
 * Author:  Zhao Jize <zhaojize@outlook.com>
 * All copyright reserved by the author.
*/

/* 1, 5, 9, 13, 17, 21*/

/*v2.0.1*/

#if !defined __BASE_TENSOR_H
#define __BASE_TENSOR_H

#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <string.h>
#include <tuple>
#include <assert.h>
#include <omp.h>

#include "globalconfig.h"
#include "cppblasex.h"
#include "cppblas.h"
#include "cpplapack.h"
#include "stringname.h"
#include "LLMOperation.h"
#include "TypeCompatible.h"
#include "TensorStruct.h"
#include "miolapack.h"

template<typename T> class Vector;

template<typename DTTYPE> 
class BaseTensor : public TensorStruct 
{
    using RDTTYPE = typename istc::SVDReal<DTTYPE>::type;

    public:
        long fileSize;              // the size of tensor in disk, in unit of byte

        DTTYPE* tensorData;

    protected:
        BLASINT maxDataSize;

    public:
/********************constructor and destructor*********************/

/*read from file*/

        BaseTensor(const char* filename, const bool &ifremoved, const BLASINT &readtype = 0);

/*read from stream*/

        BaseTensor(std::ifstream &ifs);

/*initialize tensor from matrix*/

        template<typename T = DTTYPE> BaseTensor(const uint32_t &tr, const BLASINT* rd, const T* data = nullptr, const BLASINT &dtsize = 0);

/*parameter changable constructor*/

	template<typename T = DTTYPE> BaseTensor(std::initializer_list<BLASINT> rd, const T* data = nullptr, const BLASINT &dtsize = 0);

/*tensor contraction 1: T_...i... T^*_...i...*/ 

        BaseTensor(const BaseTensor &bt, const uint32_t &idx, void* tmpdata, const BLASINT &tdsize); 

/*tensor contraction 2 (btl(A) != btr(B)) :
  (S) Sequential contraction : A_ijk B_mjn   = R_ikmn  
  (D) Direct product like : A_ijqkl  B_mnqst =  R_ijmnklst
  (I) Insert right index into left :  A_ijk B_mjn  = R_imnk
*/

        BaseTensor(const char &cttype, const BaseTensor &btl, const uint32_t &il, const BaseTensor &btr, const uint32_t &ir, void* tmpdata, const BLASINT &tdsize); 

/*internal index contraction*/

        BaseTensor(const BaseTensor &bt, const uint32_t &il, const uint32_t &ir);

/*direct product of two tensors with the same rank :  T_{iq;jw;ke;lr;mt;...} = A_{ijklm...} B_{qwert...}, iq(jw, ke, lr, mt, ...) is denoted by one index in T*/

        BaseTensor(const BaseTensor &btl, const BaseTensor &btr, void* tmpdata, const BLASINT &tdsize);

/*copy and move constructor*/

        BaseTensor(const BaseTensor &bt, const DTTYPE &scale = 1.0);

	BaseTensor(BaseTensor<DTTYPE> &&bt, const DTTYPE &scale = 1.0);

        virtual ~BaseTensor();

/********************end of constructor and destructor********************/

/**************************test the status of tensor*************************/

	inline BLASINT getMaxDataSize() const {return maxDataSize;}

/**************************reset maxDataSize and reallocate tensorData*****************************/
 
 	void reSetMaxDataSize(const BLASINT &maxsize);

/***********************change the status of a tensor*******************/	
/**************tensorData can only be changed by the following functions if a tensor is contructed************/

        template<typename T = DTTYPE> typename std::enable_if<istc::isComplex<T>::value && std::is_same<T, DTTYPE>::value, bool>::type fill(const double &min, const double &max, const uint32_t &rdtype = 0);

        template<typename T = DTTYPE> typename std::enable_if<!istc::isComplex<T>::value && std::is_same<T, DTTYPE>::value, bool>::type fill(const double &min, const double &max, const uint32_t &rdtype = 0);

	void fill(const DTTYPE &el);

	template<typename T> bool fill(const T* data, const BLASINT &dtsize);

/*resize the tensor, dataSize and maxDataSize can be changed. 
 * data may be destroyed
 * */

	bool resize(std::initializer_list<BLASINT> rd);

        bool resize(const uint32_t &tr, const BLASINT* rd);

        bool truncateRankDim(const uint32_t &idx, const BLASINT &resdim);  // resdim > 0

/*********************end of the initializer****************************/
 
        inline DTTYPE &operator[](const BLASINT &i){return tensorData[i];}

	BaseTensor &operator = (const BaseTensor &bt);

        BaseTensor &operator *= (const DTTYPE &s);
        BaseTensor &operator /= (const DTTYPE &s);
       
        bool operator == (const BaseTensor &bt) noexcept;

/*Operations on one leg. These operations should retain the structure of the tensor*/

/*1. T_...i... * lambda[i], no summation over i */

        template<typename TLP> bool legProduct(const uint32_t &il, const TLP* lambda, const BLASINT &lmbdsize);

/*lambda should be rank-1 tensor*/

	template<typename TLP> bool legProduct(const uint32_t &il, const Vector<TLP> &lambda);

/*2. T_...i... / lambda[i], no summation over i*/

        template<typename TLP> bool legDivision(const uint32_t &il, const TLP* lambda, const BLASINT &lmbdsize);

/*lambda should be rank-1 tensor*/

        template<typename TLP> bool legDivision(const uint32_t &il, const Vector<TLP> &lambda);

/*change the position of the index*/

/*shift index "il" before "iprev"*/

        bool shiftBefore(const uint32_t &il, const uint32_t &iprev, void* tmpdata, const BLASINT &tdsize);

        bool shiftBefore(const uint32_t &il, const uint32_t &iprev, BaseTensor<DTTYPE> &bt) const;

/*shift index "il" after "iback"*/

        bool shiftAfter(const uint32_t &il, const uint32_t &iback, void* tmpdata, const BLASINT &tdsize);

        bool shiftAfter(const uint32_t &il, const uint32_t &iback, BaseTensor<DTTYPE> &bt) const;

/*permute the index of the tensor*/

        bool permute(std::initializer_list<uint32_t> ri, void* tmpdata, const BLASINT &tdsize);

/*move the leg to the leftmost and perform singular value decomposition, T = U\lambda V^T,
* thistype => see rightSVD(...);
* type = 'S' or 'A'
*/

        template<typename TSR, typename = typename std::enable_if<std::is_base_of<BaseTensor<DTTYPE>, TSR>::value>::type> BLASINT leftSVD(const char &thistype, const uint32_t &index, const char &Utype, Vector<RDTTYPE> &lambda, TSR &U, void* pool, const BLASINT &poolsize);

        auto leftSVD(const char &thistype, const uint32_t &index, const char &Utype, void* pool, const BLASINT &poolsize);

/*move the leg to the rightmost and perform singular value decomposition, T = U\lambda V^T  
 * thistype = 'K', 'O' or 'D' :
 *  'K' -> the data of tensor is kept as before;
 *  'O' -> the tensor is overlapped by U with the column min(M, N)
 *  'D' -> the data of tensor is destroyed.
 *
* type = 'S' or 'A', see the manual of lapack 
*/

        template<typename TSR, typename = typename std::enable_if<std::is_base_of<BaseTensor<DTTYPE>, TSR>::value>::type> BLASINT rightSVD(const char &thistype, const uint32_t &index, const char &VDtype, Vector<RDTTYPE> &lambda, TSR &VD, void* pool, const BLASINT &poolsize);

        auto rightSVD(const char &thistype, const uint32_t &index, const char &VDtype, void* pool, const BLASINT &poolsize);

/*leftSVDD : SVD using gesdd. For more information, see the manual of lapack*/

template<typename TSR, typename = typename std::enable_if<std::is_base_of<BaseTensor<DTTYPE>, TSR>::value>::type> BLASINT leftSVDD(const char &thistype, const uint32_t &index, const char &SVDtype, TSR &U, Vector<RDTTYPE> &lambda, TSR &VD, void* pool, const BLASINT &poolsize);

/*rightSDD : SVD using gesdd*/

template<typename TSR, typename = typename std::enable_if<std::is_base_of<BaseTensor<DTTYPE>, TSR>::value>::type> BLASINT rightSVDD(const char &thistype, const uint32_t &index, const char &SVDtype, TSR &U, Vector<RDTTYPE> &lambda, TSR &VD, void* pool, const BLASINT &poolsize);

/*high order SVD, T_...i...j...k...l... = U_i\alpha U_j\beta U_k\gamma U_l\keppa S_...\alpha...\beta...\gamma...\keppa...
* here all U are unitary matrix. After this decomposition, the core tensor is stored locally.
*/
/*class 1: no truncation, Us are unitary matrix,  core tensor is the same size as original tensor*/

        void highOrderLSVD(const uint32_t &num, const uint32_t* index, DTTYPE**U, const BLASINT* Usize, void* lambdamax, const BLASINT &lmbdsize, void* pool, const BLASINT &poolsize);

        void highOrderLSVD(const uint32_t &num, const uint32_t* index, BaseTensor**U, Vector<RDTTYPE>** lambda, void* pool, const BLASINT &poolsize);

/*class 2: no trunction in U and lambda,  core tensor are truncated according to dimtru*/

        void highOrderLSVD(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, DTTYPE** U, const BLASINT* Usize, void* lambdamax, const BLASINT &lmbdsize, void* pool, const BLASINT &poolsize);

        void highOrderLSVD(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, BaseTensor** U, Vector<RDTTYPE>** lambda, void* pool, const BLASINT &poolsize);

/*class 3: use gesvdx for partial SVD, U is truncated and core tensor are truncated according to dimtru*/	

        void highOrderLSVDx(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, DTTYPE** U, const BLASINT* Usize, void* lambdamax, const BLASINT &lmbdsize, const BLASINT* dimtrlbd, void* pool, const BLASINT &poolsize);

        void highOrderLSVDx(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, BaseTensor** U, Vector<RDTTYPE>** lambda, void* pool, const BLASINT &poolsize);

/*normalize
 * ntype = 0 : v*v^\dagger = 1.0 
 * ntype = 1 : v/abs(max(v))
 * ntype = ...
*/
        template<typename RDT, typename = typename std::enable_if<std::is_floating_point<RDT>::value>::type> auto normalize(const BLASINT &ntype, const RDT &nr = 1.0);

        template<typename RDT, typename = typename std::enable_if<std::is_floating_point<RDT>::value>::type> auto norm(const BLASINT &ntype);

/*complex conjugate*/

        void cconj(void) noexcept;

/*save Tensor to disk*/

        virtual bool saveTensorToFile(const char* filename) const;     // binary

        virtual bool saveTensorToStream(std::ofstream &ofs) const;

        bool readTensorFromStream(std::ifstream &ifs);

	bool readTensorFromMemory(char* buffer);

/*print tensorData to file*/

        bool printTensorToFile(const char* suffixname, const streamsize &width, const char &type = 'D');   // ascii, type ='A' or 'D'

/*check the tensor*/	

        void checkBaseTensor() const;

    private:
        BaseTensor() = delete;  // default constructor is forbidden!        

	void setFileSize() noexcept;

	bool shiftBefore_(const uint32_t &il, const uint32_t &iprev, void* tmpdata, const BLASINT &tdsize);

	bool shiftBefore_(const uint32_t &il, const uint32_t &iprev, BaseTensor<DTTYPE> &bt) const;

	bool shiftAfter_(const uint32_t &il, const uint32_t &iback, void* tmpdata, const BLASINT &tdsize);

	bool shiftAfter_(const uint32_t &il, const uint32_t &iback, BaseTensor<DTTYPE> &bt) const;

    public:
        template<typename T> friend void tensorContraction(const char &cttype, const BaseTensor<T> &btl, const uint32_t &il, const BaseTensor<T> &btr, const uint32_t &ir, BaseTensor<T> &bc, void* tmpdata, const BLASINT &tdsize);
};

template <typename DTTYPE> 
BaseTensor<DTTYPE>::BaseTensor(const char* filename, const bool &ifremoved, const BLASINT &readtype)
{
    if (readtype == 0)
    {
        ifstream file(filename, ios::in);
        if (!file) assert(0);

        assert(readTensorFromStream(file));

        file.close();  
    }
    else if (readtype == 1)
    {
        FILE* file = fopen(filename, "r");
        if (!file) assert(0);

        struct stat filestat;
        if (fstat(fileno(file), &filestat) < 0) assert(0);

        char* buffer = new(std::nothrow) char[filestat.st_size];
        assert(buffer);

        if (fread(buffer, sizeof(char), filestat.st_size, file) != filestat.st_size) assert(0);

        assert(readTensorFromMemory(buffer));

        delete []buffer;

        fclose(file); 
    }
    else 
    {
        int filedes = open(filename, O_RDONLY);
        if (filedes < 0) assert(0);

        struct stat filestat;
        if (fstat(filedes, &filestat) < 0) assert(0);

        char* buffer = (char*)mmap(0, filestat.st_size, PROT_READ, MAP_PRIVATE, filedes, 0);
        if (!buffer) assert(0);

        if (madvise(buffer, filestat.st_size, MADV_WILLNEED | MADV_SEQUENTIAL) != 0)
        {
            munmap(buffer, filestat.st_size);
            close(filedes);

            assert(0);
        }

        assert(readTensorFromMemory(buffer));

        close(filedes);

        munmap(buffer, filestat.st_size);
    }

    if (ifremoved) remove(filename);

    maxDataSize = dataSize;

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    
}

/*read from stream*/

template <typename DTTYPE> 
BaseTensor<DTTYPE>::BaseTensor(ifstream &ifs)
{
    assert(readTensorFromStream(ifs));

    maxDataSize = dataSize;

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    
}

/*initialize a tensor from a matrix*/

template<typename DTTYPE> template<typename T> 
BaseTensor<DTTYPE>::BaseTensor(const uint32_t &tr, const BLASINT* rd, const T* data, const BLASINT &dtsize) : TensorStruct(tr, rd)
{
    maxDataSize = dataSize;

    tensorData = new(std::nothrow) DTTYPE[maxDataSize];
    assert(tensorData);

    if (dtsize > 0) 
    {
        if (dtsize >= dataSize)
	{	
            for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = data[i];   // "copy" may be wrong due to possible different data type!
	}
	else std::cout << "Be careful that dtsize is smaller than dataSize!" << std::endl;	
    }

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    
}

template<typename DTTYPE> template<typename T> 
BaseTensor<DTTYPE>::BaseTensor(std::initializer_list<BLASINT> rd, const T* data, const BLASINT &dtsize) : TensorStruct(rd)
{
    maxDataSize = dataSize;

    tensorData = new(std::nothrow) DTTYPE[maxDataSize]; 
    assert(tensorData);

    if (dtsize > 0)
    {
        if (dtsize >= dataSize)
	{	
	    for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = data[i];
	}
	else std::cout << "Be careful that dtsize is smaller than dataSize!" << std::endl;	
    }

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    
}

/*tensor contraction 1: T_...i... T^*_...i...*/

template<typename DTTYPE> 
BaseTensor<DTTYPE>::BaseTensor(const BaseTensor &bt, const uint32_t &idx, void* tmpdata, const BLASINT &tdsize) : TensorStruct(bt, idx)
{
    maxDataSize = dataSize;

    BLASINT dm = bt.rankDim[idx];

    tensorData = new(std::nothrow) DTTYPE[maxDataSize];
    assert(tensorData);

    if (idx == 0) 
    {
        BLASINT dr = bt.dataSize/dm;

	gemm('C', 'N', dr, dr, dm, 1.0, bt.tensorData, dr, bt.tensorData, dr, 0.0, tensorData, dr);

	if (typeid(DTTYPE) != typeid(real(DTTYPE()))) // Complex
	{
            for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = conj(tensorData[i]);	    
	}
    }
    else if (idx+1 == bt.getTensorRank())	
    {
	BLASINT dl = bt.getDataSize()/dm;

	gemm('N', 'C', dl, dl, dm, 1.0, bt.tensorData, dm, bt.tensorData, dm, 0.0, tensorData, dl);	
    }
    else
    {
	BLASINT dl = 1;
        for (uint32_t i = 0; i < idx; i++) dl *= bt.getRankDim(i);

        BLASINT dr = 1;
        for (uint32_t i = idx+1; i < bt.getTensorRank(); i++) dr *= bt.getRankDim(i);

	assert(dl*dm*dr == bt.dataSize);

        DTTYPE* localtmpdata = reinterpret_cast<DTTYPE*> (tmpdata);

	BLASINT dmr = dm*dr;

	if (tdsize >= bt.getDataSize()*sizeof(DTTYPE))
	{
            for (BLASINT l = 0; l < dl; l++) llmopr::fastTransposeTo(dm, dr, bt.tensorData+l*dmr, dr, localtmpdata+l*dmr, dm);    

            gemm('N', 'C', dl*dr, dl*dr, dm, 1.0, localtmpdata, dm, localtmpdata, dm, 0.0, tensorData, dl*dr);
	}
	else if (tdsize >= dmr*sizeof(DTTYPE))
	{
	    for (BLASINT l = 0; l < dl; l++) llmopr::transposeOnsite(dm, dr, bt.tensorData+l*dmr, localtmpdata);	

            gemm('N', 'C', dl*dr, dl*dr, dm, 1.0, bt.tensorData, dm, bt.tensorData, dm, 0.0, tensorData, dl*dr);

	    /*restore the data in bt*/

            for (BLASINT l = 0; l < dl; l++) llmopr::transposeOnsite(dr, dm, bt.tensorData+l*dmr, localtmpdata);	
	}
	else
	{
	    std::cout << "tmpdata is too small, it is " << tdsize << ", but it should be at least " << dm*dr*sizeof(DTTYPE) << std::endl;
            
            for (BLASINT l = 0; l < dl; l++) llmopr::transposeOnsite(dm, dr, bt.tensorData+l*dmr);	

            gemm('N', 'C', dl*dr, dl*dr, dm, 1.0, bt.tensorData, dm, bt.tensorData, dm, 0.0, tensorData, dl*dr);

	    /*restore the data in bt*/

            for (BLASINT l = 0; l < dl; l++) llmopr::transposeOnsite(dr, dm, bt.tensorData+l*dmr);
	}
    }

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    
}

/* tensor contraction : cttype = 'S', 'D', or 'I'*/

template<typename DTTYPE> 
BaseTensor<DTTYPE>::BaseTensor(const char &cttype, const BaseTensor &btl, const uint32_t &il, const BaseTensor &btr, const uint32_t &ir, void* tmpdata, const BLASINT &tdsize) : TensorStruct(btl, il, btr, ir, cttype)
{
    assert(&btl != &btr);

#ifdef DEBUG_BASETENSOR_TIME_cbibitt   
    std::cout<<std::endl<<"***start cbibitt  :: " <<std::endl; 
    clock_t time_begin = clock();
#endif

    maxDataSize = dataSize;

/*left dim*/
		    
    BLASINT Dll = 1;
    for (uint32_t i = 0; i < il; i++) Dll *= btl.getRankDim(i);
    
    BLASINT Dlm = btl.getRankDim(il); 

    BLASINT Dlr = (btl.getDataSize()/Dll)/Dlm;

/*right dim*/

    BLASINT Drl = 1;
    for (uint32_t i = 0; i < ir; i++) Drl *= btr.getRankDim(i);

    BLASINT Drm = btr.getRankDim(ir);

    BLASINT Drr = (btr.getDataSize()/Drl)/Drm;

    assert(dataSize == Dll*Dlr*Drl*Drr);
    assert(Dlm == Drm);

    DTTYPE* localtmpdata = reinterpret_cast<DTTYPE*>(tmpdata);

    tensorData = new(std::nothrow) DTTYPE[maxDataSize];
    assert(tensorData);

/*contraction*/

    if (cttype == 'S')
    {
        char transl;
        char transr = 'T';

        BLASINT ldl;
        BLASINT ldr;

        if (il == 0)
        {
	    transl = 'T';
            ldl = Dll*Dlr;	    
        }
        else if (il+1 == btl.getTensorRank())
        {
	    transl = 'N';
            ldl = Dlm;	    
        }
        else 
        {
	    BLASINT dlmr = Dlm*Dlr;

            if (tdsize >= btl.getDataSize()*sizeof(DTTYPE))
            {
	        #pragma omp parallel for schedule(dynamic, 1)	
                for (BLASINT l = 0; l < Dll; l++) llmopr::fastTransposeTo(Dlm, Dlr, btl.tensorData+l*dlmr, Dlr, localtmpdata+l*dlmr, Dlm);

	        copy(btl.getDataSize(), localtmpdata, 1, btl.tensorData, 1);
	    }
            else if (tdsize >= dlmr*sizeof(DTTYPE))
            {
                for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlm, Dlr, btl.tensorData+l*dlmr, localtmpdata); 
	    }
	    else
	    {
	        std::cout << "warning : tmpdata is too small, tdsize is " << tdsize << ", it should be at leat " << dlmr*sizeof(DTTYPE) <<std::endl;
	
                for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlm, Dlr, btl.tensorData+l*dlmr);		
	    }

	    transl = 'N';
	    ldl = Dlm;
        }

        const DTTYPE* psour = btr.tensorData;

        if (ir == 0)
        {
	    transr = 'N';
            ldr = Drl*Drr;	    
        }
        else if (ir+1 == btr.getTensorRank())
        {
	    transr = 'T';
            ldr = Drm;	    
        }
        else 
        {
	    BLASINT drmr = Drm*Drr;

            if (tdsize >= btr.getDataSize()*sizeof(DTTYPE))
            {
	        #pragma omp parallel for schedule(dynamic, 1)	
                for (BLASINT l = 0; l < Drl; l++) llmopr::fastTransposeTo(Drm, Drr, btr.tensorData+l*drmr, Drr, localtmpdata+l*drmr, Drm);

                psour = localtmpdata;	    
	    }
            else if (tdsize >= drmr*sizeof(DTTYPE))
	    {
	        for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drm, Drr, btr.tensorData+l*drmr, localtmpdata);
	    }
            else
	    {
	        std::cout << "warning : tmpdata is too small, tdsize is " << tdsize << ", it should be at least "<<drmr*sizeof(DTTYPE)<<std::endl;	
            
                for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drm, Drr, btr.tensorData+l*drmr);	
	    }

	    transr = 'T';
	    ldr = Drm;
        }  

        gemm(transl, transr, Dll*Dlr, Drl*Drr, Dlm, 1.0, btl.tensorData, ldl, psour, ldr, 0.0, tensorData, Drl*Drr);

/*restore the data in btr*/

        if (ir != 0 && ir+1 != btr.getTensorRank() && tdsize < btr.getDataSize()*sizeof(DTTYPE)) 
        {
            BLASINT drmr = Drm*Drr;
        
            if (tdsize >= drmr*sizeof(DTTYPE)) 
	    {
	        for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drr, Drm, btr.tensorData+l*drmr, localtmpdata);	
	    }	
	    else 
	    {
                for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drr, Drm, btr.tensorData+l*drmr);		
	    }
        }

/*restore the data in btl*/

        if (il != 0 && il+1 != btl.getTensorRank())
        {
	    BLASINT dlmr = Dlm*Dlr;

    	    if (tdsize >= btl.getDataSize()*sizeof(DTTYPE))
	    {
	        #pragma omp parallel for schedule(dynamic, 1)	
                for (BLASINT l = 0; l < Dll; l++) llmopr::fastTransposeTo(Dlr, Dlm, btl.tensorData+l*dlmr, Dlm, localtmpdata+l*dlmr, Dlr);

	        copy(btl.getDataSize(), localtmpdata, 1, btl.tensorData, 1);	    
	    }
	    else if (tdsize >= dlmr*sizeof(DTTYPE))
	    {
	        for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlr, Dlm, btl.tensorData+l*dlmr, localtmpdata);
	    }
	    else
	    {
	        for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlr, Dlm, btl.tensorData+l*dlmr);
	    }
        }
    }
    else if (cttype == 'D')
    {
        if (Drl == 1 && Dll == 1) 
	{
	    gemm('T', 'N', Dlr, Drr, Dlm, 1.0, btl.tensorData, Dlr, btr.tensorData, Drr, 0.0, tensorData, Drr);
        }
    	else if (Drl == 1 && Dlr == 1) 
	{
	    gemm('N', 'N', Dll, Drr, Dlm, 1.0, btl.tensorData, Dlm, btr.tensorData, Drr, 0.0, tensorData, Drr);	    
        }
    	else if (Drl == 1 && tdsize >= btl.getDataSize()*sizeof(DTTYPE))
	{
	    llmopr::transposeMRto(Dll, Dlm, Dlr, btl.tensorData, localtmpdata);
	    gemm('N', 'N', Dll*Dlr, Drr, Dlm, 1.0, localtmpdata, Dlm, btr.tensorData, Drr, 0.0, tensorData, Drr);	    
        }
	else if (Dlr == 1 && Drr == 1) 
	{
	    gemm('N', 'T', Dll, Drl, Dlm, 1.0, btl.tensorData, Dlm, btr.tensorData, Drm, 0.0, tensorData, Drl);
	}
	else if (Dlr == 1 && tdsize >= btr.getDataSize()*sizeof(DTTYPE))
	{
	    llmopr::transposeLMto(Drl, Drm, Drr, btr.tensorData, localtmpdata);
	    gemm('N', 'N', Dll, Drl*Drr, Dlm, 1.0, btl.tensorData, Dlm, localtmpdata, Drl*Drr, 0.0, tensorData, Drl*Drr);	    
	}
	else if (Dll == 1 && Drr == 1) 
	{
	    gemm('N', 'N', Drl, Dlr, Dlm, 1.0, btl.tensorData, Dlm, btr.tensorData, Drm, 0.0, tensorData, Dlr);
	}
        else if (Drr == 1)
	{
	    #pragma omp parallel for  schedule(dynamic, 1) 	
	    for (BLASINT l = 0; l < Dll; l++)
	    {
		const DTTYPE* ltdata = btl.tensorData+l*Dlm*Dlr;
		DTTYPE* localdata = tensorData+l*Drl*Drr;
	        gemm('N', 'N', Drl, Drr, Dlm, 1.0, ltdata, Dlr, btr.tensorData, Drm, 0.0, localdata, Drr);	
	    }	    
	}
	else
	{
	    #pragma omp parallel for  schedule(dynamic, 1)  
            for (BLASINT lr = 0; lr < Dll*Drl; lr++)
            {
                const BLASINT l = lr/Drl;           
                const BLASINT r = lr%Drl;           
                const DTTYPE* ltdata = btl.tensorData+l*Dlm*Dlr;             
                const DTTYPE* rtdata = btr.tensorData+r*Drm*Drr;             
                DTTYPE* localdata = tensorData+(l*Drl+r)*Dlr*Drr;            
                gemm('T', 'N', Dlr, Drr, Dlm, 1.0, ltdata, Dlr, rtdata, Drr, 0.0, localdata, Drr);
            }
	}
    }
    else // (cttype == 'I')
    {
        if (Dll*Drl > 10)
        {
            #pragma omp parallel for  schedule(dynamic, 1)  
            for (BLASINT lr = 0; lr < Dll*Drl; lr++)
            {
                const BLASINT l = lr/Drl;
                const BLASINT r = lr%Drl;           
                const DTTYPE* ltdata = btl.tensorData+l*Dlm*Dlr;
                const DTTYPE* rtdata = btr.tensorData+r*Drm*Drr; 
                DTTYPE* localdata = tensorData+(l*Drl+r)*Drr*Dlr;            
                gemm('T', 'N', Drr, Dlr, Dlm, 1.0, rtdata, Drr, ltdata, Dlr, 0.0, localdata, Dlr);
            }
        }
        else
        {  
            for (BLASINT lr = 0; lr < Dll*Drl; lr++)
            {
                const BLASINT l = lr/Drl;
                const BLASINT r = lr%Drl;
                const DTTYPE* ltdata = btl.tensorData+l*Dlm*Dlr;
                const DTTYPE* rtdata = btr.tensorData+r*Drm*Drr;
                DTTYPE* localdata = tensorData+(l*Drl+r)*Drr*Dlr;
                gemm('T', 'N', Drr, Dlr, Dlm, 1.0, rtdata, Drr, ltdata, Dlr, 0.0, localdata, Dlr);
            }
        }	    
    }

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif   

#ifdef DEBUG_BASETENSOR_TIME_cbibitt    
    std::cout<<"***Time on BaseTensor(cbibitt) is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl<<std::endl;
#endif    	
}

/*internal index contraction*/

template<typename DTTYPE> 
BaseTensor<DTTYPE>::BaseTensor(const BaseTensor &bt, const uint32_t &il, const uint32_t &ir) : TensorStruct(bt, il, ir)
{
    assert(il < ir);

    BLASINT Dl = 1;
    for (uint32_t i = 0; i < il; i++) Dl *= bt.getRankDim(i);

    BLASINT Dm = 1;
    for (uint32_t i = il+1; i < ir; i++) Dm *= bt.getRankDim(i);
    
    BLASINT Dr = 1;
    for (uint32_t i = ir+1; i < bt.getTensorRank(); i++) Dr *= bt.getRankDim(i);

    BLASINT Dil = bt.getRankDim(il); 
    assert(Dil == bt.getRankDim(ir));
    
    maxDataSize = dataSize;

    tensorData = new(std::nothrow) DTTYPE[maxDataSize];
    assert(tensorData);

    for (BLASINT l = 0; l < Dl; l++)
    {
        for (BLASINT m = 0; m < Dm; m++)
        {
            for (BLASINT r = 0; r < Dr; r++) 
            {
                DTTYPE sum = 0.0;
                for (BLASINT k = 0; k < Dil; k++) sum += bt.tensorData[(((l*Dil+k)*Dm+m)*Dil+k)*Dr+r];
                tensorData[(l*Dm+m)*Dr+r] = sum; 
            }
        }
    }

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    
}

template<typename DTTYPE> 
BaseTensor<DTTYPE>::BaseTensor(const BaseTensor &btl, const BaseTensor &btr, void* tmpdata, const BLASINT &tdsize)
{
    assert(btl.tensorRank == btr.tensorRank);    

    tensorRank = btl.tensorRank + btr.tensorRank;    
    assert(tensorRank <= MAXTRK);
    
    for (unsigned i = 0; i < btl.tensorRank; i++) rankDim[i] = btl.rankDim[i];

    for (unsigned i = 0; i < btr.tensorRank; i++) rankDim[btl.tensorRank+i] = btr.rankDim[i];
    
    dataSize = btl.dataSize * btr.dataSize;

    maxDataSize = dataSize;

/*tensorData should not be nullptr*/

    tensorData = new(std::nothrow) DTTYPE[maxDataSize]; 
    assert(tensorData);

    for (BLASINT i = 0; i < btl.dataSize; i++)
    {
        for (BLASINT j = 0; j < btr.dataSize; j++) tensorData[i*btr.dataSize+j] = btl.tensorData[i] * btr.tensorData[j];    
    } 

    for (unsigned i = 0; i+1 < btr.tensorRank; i++) shiftBefore(btl.tensorRank+i, 2*i+1, tmpdata, tdsize);

    for (unsigned i = 0; i < btl.tensorRank; i++) rankCombination(i, 2);

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif
}

template <typename DTTYPE> 
BaseTensor<DTTYPE>::BaseTensor(const BaseTensor &bt, const DTTYPE &scale) : TensorStruct(bt)
{
    fileSize = bt.fileSize;

    maxDataSize = dataSize;

    tensorData = new(std::nothrow) DTTYPE[maxDataSize];
    assert(tensorData);

    memcpy(tensorData, bt.tensorData, dataSize*sizeof(DTTYPE));

    scal(dataSize, scale, tensorData, 1);

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    
}

template<typename DTTYPE>
BaseTensor<DTTYPE>::BaseTensor(BaseTensor<DTTYPE> &&bt, const DTTYPE &scale) : TensorStruct(bt)
{
    std::cout<<"move base to base"<<endl;

    fileSize = bt.fileSize;
    maxDataSize = bt.getMaxDataSize();

    tensorData = bt.tensorData;
    if (scale != 1.0) scal(this->dataSize, scale, tensorData, 1);

    bt.tensorData = nullptr;
    bt.dataSize = 0;
    bt.maxDataSize = 0;   

#ifdef CHECK_BASETENSOR
    this->checkBaseTensor();
#endif    
}

template<typename DTTYPE> 
BaseTensor<DTTYPE>::~BaseTensor()
{
    if (tensorData != nullptr) 
    { 
        delete []tensorData;

	tensorData = nullptr;
    }
}

template<typename DTTYPE>
void BaseTensor<DTTYPE>::reSetMaxDataSize(const BLASINT &maxsize)
{
    maxDataSize = maxsize;

    delete []tensorData;

    tensorData = new(std::nothrow) DTTYPE[maxDataSize];
    assert(tensorData);
}

/*********************initialize tensorData of a tensor*****************/	

template<typename DTTYPE> template<typename T> 
typename std::enable_if<istc::isComplex<T>::value && std::is_same<T, DTTYPE>::value, bool>::type BaseTensor<DTTYPE>::fill(const double &min, const double &max, const uint32_t &rdtype)
{
    if (max <= min) 
    {
	std::cout << " Error: max should be larger than min!" << std::endl;   
	return false;	
    }

    for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = DTTYPE(drand48()*(max-min)+min, drand48()*(max-min)+min);	

    return true;
}

template<typename DTTYPE> template<typename T> 
typename std::enable_if<(!istc::isComplex<T>::value) && std::is_same<T, DTTYPE>::value, bool>::type BaseTensor<DTTYPE>::fill(const double &min, const double &max, const uint32_t &rdtype)
{
    if (max <= min)
    { 
	std::cout << " Error: max should be larger than min!" << std::endl;    
	return false;	
    }

    for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = drand48()*(max-min)+min;	

    return true;
}

template<typename DTTYPE>
void BaseTensor<DTTYPE>::fill(const DTTYPE &el)
{
    for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = el;
}

template<typename DTTYPE> template<typename T>
bool BaseTensor<DTTYPE>::fill(const T* data, const BLASINT &dtsize)
{
    if (dataSize > dtsize) 
    {
	std::cout << " Error: number of input data is not enough!" << std::endl;    
	return false;	
    }

    if (typeid(DTTYPE) == typeid(T))
    {
        memcpy(tensorData, data, dataSize*sizeof(DTTYPE));	    
    }
    else
    {
        for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = data[i];
    }

    return true;
}

template<typename DTTYPE>
bool BaseTensor<DTTYPE>::resize(std::initializer_list<BLASINT> rd)
{
    if (rd.size() == 0)	// rd = {}, empty
    {
	tensorRank = 0;
        dataSize = 1;
    }
    else
    {
        for (auto p = rd.begin(); p < rd.end(); p++)
	{    
	    if (*p <= 0) 
	    {
		std::cout << " negative dimension!" << std::endl;    
		return false;
	    }
        }

	tensorRank = 0;
        dataSize = 1;
            
        for (auto p = rd.begin(); p < rd.end(); p++)	
	{
	    rankDim[tensorRank] = *p;
	    dataSize *= *p;
	    tensorRank++; 	
	}  
    }

    if (dataSize > maxDataSize)
    {
	maxDataSize = 32*(1+dataSize/32);

	assert (tensorData != nullptr);
	
	delete []tensorData;

        tensorData = new(std::nothrow) DTTYPE[maxDataSize];
        assert(tensorData);	
    }

    setFileSize();

    return true;
}

template<typename DTTYPE>
bool BaseTensor<DTTYPE>::resize(const uint32_t &tr, const BLASINT* rd)
{
    if (tr == 0)
    {
        tensorRank = 0;
        dataSize = 1;
    }
    else
    {
        for (uint32_t i = 0; i < tr; i++)
	{    
	    if (rd[i] <= 0) 
	    {
		std::cout << " Error: wrong dimension!" << std::endl;    
		return false;
	    }
        }

	tensorRank = 0;
        dataSize = 1;
            
        for (uint32_t i = 0; i < tr; i++)	
	{
	    rankDim[tensorRank] = rd[i];
	    dataSize *= rd[i];
	    tensorRank++; 	
	}  
    }

    if (dataSize > maxDataSize)
    {
	maxDataSize = 32*(1+dataSize/32);

	assert (tensorData != nullptr);
	
	delete []tensorData;

        tensorData = new(std::nothrow) DTTYPE[maxDataSize];
        assert(tensorData);	
    }

    setFileSize();

    return true;
 }

template<typename DTTYPE>
bool BaseTensor<DTTYPE>::truncateRankDim(const uint32_t &idx, const BLASINT &resdim)
{
    if (idx >= tensorRank || resdim <= 0 || resdim > rankDim[idx]) 
    {
        std::cout << "	Error: parameter is wrong!" << std::endl;    
	return false;
    }

    if (resdim == rankDim[idx]) return true; 

    if (idx == 0)  // first one
    {
        dataSize /= rankDim[idx];	
        rankDim[idx] = resdim;
        dataSize *= rankDim[idx];	
    }  
    else if (idx+1 == tensorRank)  // last one
    {
        BLASINT dcol = rankDim[idx]; 
        BLASINT drow = dataSize/dcol;	

	for (BLASINT i = 1; i < drow; i++)
	{
            DTTYPE* dest = tensorData + i*resdim;
            DTTYPE* sour = tensorData + i*dcol; 	   
	    memmove(dest, sour, resdim*sizeof(DTTYPE)); 
	}

	dataSize /= rankDim[idx];
	rankDim[idx] = resdim;
	dataSize *= rankDim[idx];
    }
    else   // middle
    {
	assert(0);    
    }

    setFileSize();

    return true;
}

/*********************end of the initializer****************************/

template<typename DTTYPE>
BaseTensor<DTTYPE> &BaseTensor<DTTYPE>::operator = (const BaseTensor &bt)
{
    assert(this->resize(bt.tensorRank, bt.rankDim));	

    memcpy(tensorData, bt.tensorData, dataSize*sizeof(DTTYPE));
    
    return *this; 
}

template<typename DTTYPE> 
BaseTensor<DTTYPE> &BaseTensor<DTTYPE>::operator *= (const DTTYPE &s)
{
    if (s == 1.0) return *this;

    scal(dataSize, s, tensorData, 1);

    return *this;
}

template<typename DTTYPE> 
BaseTensor<DTTYPE> &BaseTensor<DTTYPE>::operator /= (const DTTYPE &s)
{
    if (s == 1.0) return *this;

    scal(dataSize, 1.0/s, tensorData, 1);
  
    return *this;
}

template<typename DTTYPE>
bool BaseTensor<DTTYPE>::operator == (const BaseTensor &bt) noexcept
{
    if (tensorRank != bt.tensorRank) return false;

    for (uint32_t i = 0; i < tensorRank; i++)
    {
	if (rankDim[i] != bt.rankDim[i]) return false;    
    }	    

    for (BLASINT i = 0; i < dataSize; i++) 
    {
	if (dabs(tensorData[i]-bt.tensorData[i]) > 1.0e-12) return false;    
    }

    return true;
}

/*1. T_ijk = T_ijk * lambda[j] 
* no summation over "j"
*/

template<typename DTTYPE>  template <typename TLP> 
bool BaseTensor<DTTYPE>::legProduct(const uint32_t &il, const TLP* lambda, const BLASINT &lmbdsize)
{
    BLASINT Dl = 1;
    for (uint32_t i = 0; i < il; i++) Dl *= rankDim[i];

    BLASINT Dm = rankDim[il];

    BLASINT Dr = (dataSize/Dl)/Dm;

#ifdef CHECK_VECTOR_SIZE
    assert(lmbdsize >= Dm);
#endif

    if (Dr == 1)
    {
	for (BLASINT l = 0; l < Dl; l++)
        {
            for (BLASINT m = 0; m < Dm; m++) tensorData[l*Dm+m] *= lambda[m];
        }
    }
    else
    {
        for (BLASINT l = 0; l < Dl; l++)
        {
            for (BLASINT m = 0; m < Dm; m++)
            {
                for (BLASINT r = 0; r < Dr; r++) tensorData[(l*Dm+m)*Dr+r] *= lambda[m];
            }
        }
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif  

    return true;  
}

template<typename DTTYPE> template<typename TLP>
bool BaseTensor<DTTYPE>::legProduct(const uint32_t &il, const Vector<TLP> &lambda)
{
    if (this->getRankDim(il) != lambda.getDataSize())
    {
	std::cout << " Error: size does not match!" << std::endl;
        return false;	
    }

    BLASINT Dl = 1;
    for (uint32_t i = 0; i < il; i++) Dl *= rankDim[i];

    BLASINT Dm = rankDim[il];

    BLASINT Dr = (dataSize/Dl)/Dm;

    if (Dr == 1)
    {
        for (BLASINT l = 0; l < Dl; l++)
        {
            for (BLASINT m = 0; m < Dm; m++) tensorData[l*Dm+m] *= lambda.tensorData[m];
	}
    }
    else
    {
        for (BLASINT l = 0; l < Dl; l++)
        {
            for (BLASINT m = 0; m < Dm; m++)
            {
                for (BLASINT r = 0; r < Dr; r++) tensorData[(l*Dm+m)*Dr+r] *= lambda.tensorData[m];
            }
	}
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    

    return true;
}

/*2. T_ijk = T_ijk / lambda[j] 
* no summation over "j"
*/

template<typename DTTYPE> template<typename TLP> 
bool BaseTensor<DTTYPE>::legDivision(const uint32_t &il, const TLP* lambda, const BLASINT &lmbdsize)
{
    BLASINT Dl = 1;
    for (uint32_t i = 0; i < il; i++) Dl *= rankDim[i];

    BLASINT Dm = rankDim[il];

    BLASINT Dr = (dataSize/Dl)/Dm;

#ifdef CHECK_VECTOR_SIZE
    assert(lmbdsize >= Dm);
#endif    

    for (BLASINT l = 0; l < Dl; l++)
    {
        for (BLASINT m = 0; m < Dm; m++)
        {
            for (BLASINT r = 0; r < Dr; r++) tensorData[(l*Dm+m)*Dr+r] /= lambda[m];
        }
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif

    return true;    
}

template<typename DTTYPE> template<typename TLP> 
bool BaseTensor<DTTYPE>::legDivision(const uint32_t &il, const Vector<TLP> &lambda)
{
    if (this->getRankDim(il) != lambda.getDataSize())
    {
	std::cout << " Error: size does not match!" << std::endl;
        return false;	
    }

    BLASINT Dl = 1;
    for (uint32_t i = 0; i < il; i++) Dl *= rankDim[i];

    BLASINT Dm = rankDim[il];

    BLASINT Dr = (dataSize/Dl)/Dm;

    for (BLASINT l = 0; l < Dl; l++)
    {
        for (BLASINT m = 0; m < Dm; m++)
        {
            for (BLASINT r = 0; r < Dr; r++) tensorData[(l*Dm+m)*Dr+r] /= lambda.tensorData[m];
        }
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    

    return true;
}

/*shift index "idx" before "iprev":   head  0, 1, 2, 3, 4, 5, ... tail*/
/*将idx插到iprev前面*/

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::shiftBefore(const uint32_t &idx, const uint32_t &iprev, void* tmpdata, const BLASINT &tdsize)
{
    assert(idx < tensorRank && iprev < tensorRank);    

    if (iprev == idx || iprev == idx+1) return true;
    else if (idx > iprev) return shiftBefore_(idx, iprev, tmpdata, tdsize);    
    else return shiftAfter_(idx, iprev-1, tmpdata, tdsize);    
}

template<typename DTTYPE>
bool BaseTensor<DTTYPE>::shiftBefore(const uint32_t &idx, const uint32_t &iprev, BaseTensor<DTTYPE> &bt) const
{
    assert(idx < tensorRank && iprev < tensorRank);

    if (iprev == idx || iprev == idx+1)
    { 
	bt = *this;    
	return true;
    }
    else if (idx > iprev) return shiftBefore_(idx, iprev, bt);
    else return shiftAfter_(idx, iprev-1, bt);
}

/*shift index "idx" after "iback":  head  0, 1, 2, 3, 4, ... tail*/
/*将idx插到iback后面*/

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::shiftAfter(const uint32_t &idx, const uint32_t &iback, void* tmpdata, const BLASINT &tdsize)
{
    assert (idx < tensorRank && iback < tensorRank);

    if (iback == idx || iback+1 == idx) return true;
    else if (idx < iback) return shiftAfter_(idx, iback, tmpdata, tdsize);
    else return shiftBefore_(idx, iback+1, tmpdata, tdsize);
}

template<typename DTTYPE>
bool BaseTensor<DTTYPE>::shiftAfter(const uint32_t &idx, const uint32_t &iback, BaseTensor<DTTYPE> &bt) const
{
    assert (idx < tensorRank && iback < tensorRank);

    if (iback == idx || iback+1 == idx)
    {
	bt = *this;    
	return true;
    }
    else if (idx < iback) return shiftAfter_(idx, iback, bt);
    else return shiftBefore_(idx, iback+1, bt);
}

/*permute the index of the tensor : it is very slow*/

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::permute(std::initializer_list<uint32_t> ri, void* tmpdata, const BLASINT &tdsize)
{
    std::cout << "in prepare" << std::endl;	
    assert(0);

    uint32_t targetindex[MAXTRK]; 	

    uint32_t localrank = 0;

    for (auto p = ri.begin(); p < ri.end(); p++) 
    {
        targetindex[localrank] = *p;
        localrank++;	
    }

/*check*/

    assert(localrank == tensorRank);

    for (uint32_t i = 0; i < tensorRank; i++)
    {
	bool matched = false;

        for (uint32_t j = 0; j < tensorRank; j++) if (i == targetindex[j]) matched = true;	

	if (!matched) assert(0); 
    }

/*check finished*/

    uint32_t presentindex[MAXTRK]; 

    for (uint32_t i = 0; i < tensorRank; i++) presentindex[i] = i;

    uint32_t index = 0;

    while (index+1 < tensorRank)
    {
	uint32_t tindex = targetindex[index];

        uint32_t pindex = 0;
        for (uint32_t i = index; i < tensorRank; i++)
	{
            if (tindex == presentindex[i]) 
	    {
		pindex = i;
		break;
	    }
	}

	if (pindex > index)
	{
            for (uint32_t i = pindex; i > index; i--) presentindex[i] = presentindex[i-1];
	    presentindex[index] = tindex;

            shiftBefore(pindex, index, tmpdata, tdsize);
        }

	index++;
    }

    for (uint32_t i = 0; i < tensorRank; i++) assert(targetindex[i] == presentindex[i]);

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    

    return true;
}

/*move the leg to the leftmost and perform singular value decomposition, T = U\lambda V^T,
* Utype = 'S' or 'A', 
* thistype = 'O', 'K', or 'D'
* V^T is stored in this tensor if "thistype = 'O'"  => overwritten
* this tensor is kept when thistype = 'K'           => kept
* this tensor is destroyed when thistype = 'D'      => destroyed
*/

template<typename DTTYPE> template<typename TSR, typename> BLASINT BaseTensor<DTTYPE>::leftSVD(const char &thistype, const uint32_t &index, const char &Utype, Vector<RDTTYPE> &lambda, TSR &U, void* pool, const BLASINT &poolsize)
{
#ifdef DEBUG_LEFTSVD_TIME
    clock_t time_begin = clock();
#endif
	
    assert(index < tensorRank);
    assert(thistype == 'O' || thistype == 'K' || thistype == 'D');
    assert(Utype == 'A' || Utype == 'S'); 

    BLASINT diml = 1;
    for (BLASINT i = 0; i < index; i++) diml *= rankDim[i];

    BLASINT dimm = rankDim[index];

    BLASINT dimr = 1;
    for (BLASINT i = index+1; i < tensorRank; i++) dimr *= rankDim[i];

    const BLASINT dimlr = diml*dimr;

    assert(dimlr*dimm == dataSize);

    const BLASINT mindim = dimm<dimlr?dimm:dimlr;

    BLASINT dcolU;  // column of U

    if (Utype == 'A') dcolU = dimm;
    else dcolU = mindim;

    DTTYPE* psour;

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    assert(dataSize < poolsize);

    BLASINT extdtsize = 0;

    if (thistype == 'K') 
    {
	llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);
        psour = tmpdata;

        extdtsize = dataSize;	
    }
    else
    {
        shiftBefore(index, 0, tmpdata, dataSize*sizeof(DTTYPE));
        psour = tensorData;
    }

    char VDtype = 'N';
    if (thistype == 'O') VDtype = 'O';

    BLASINT info;

    DTTYPE VT[16];// just a variable, not referenced, 

    assert(lambda.resize({dcolU}));
    lambda.fill(0);

    assert(U.resize({dimm, dcolU}));

    miolpk::miogesvd(Utype, VDtype, dimm, dimlr, psour, dimlr, lambda.tensorData, U.tensorData, dcolU, VT, 2, info, tmpdata+extdtsize, poolsize-extdtsize*sizeof(DTTYPE));

    if (thistype == 'O') 
    {
	dataSize /= rankDim[0];

	rankDim[0] = mindim;    // min(dimm, dimlr) 

	dataSize *= rankDim[0];

	setFileSize();  //correct
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    

#ifdef DEBUG_LEFTSVD_TIME 
    std::cout<<"Time on leftSVD is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return info;
}

template<typename DTTYPE> 
auto BaseTensor<DTTYPE>::leftSVD(const char &thistype, const uint32_t &index, const char &Utype, void* pool, const BLASINT &poolsize)
{
    assert(index < tensorRank);
    assert(thistype == 'O' || thistype == 'K' || thistype == 'D');
    assert(Utype == 'A' || Utype == 'S'); 

    BLASINT diml = 1;
    for (BLASINT i = 0; i < index; i++) diml *= rankDim[i];

    BLASINT dimm = rankDim[index];

    BLASINT dimr = 1;
    for (BLASINT i = index+1; i < tensorRank; i++) dimr *= rankDim[i];

    const BLASINT dimlr = diml*dimr;

    assert(dimlr*dimm == dataSize);

    const BLASINT mindim = dimm<dimlr?dimm:dimlr;

    BLASINT dcolU;  // column of U

    if (Utype == 'A') dcolU = dimm;
    else dcolU = mindim;

    DTTYPE* psour;

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    assert(dataSize < poolsize);

    if (thistype == 'K')
    {
	llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);
        psour = tmpdata;
    }
    else
    {
        shiftBefore(index, 0, tmpdata, dataSize*sizeof(DTTYPE));
        psour = tensorData;
    }

    char VDtype = 'N';
    if (thistype == 'O') VDtype = 'O';

    BLASINT info;

    DTTYPE VT[16];// just a variable, not referenced, 

    Vector<RDTTYPE> lambda(dcolU);
    lambda.fill(0);

    BaseTensor<DTTYPE> U({dimm, dcolU});

    miolpk::miogesvd(Utype, VDtype, dimm, dimlr, psour, dimlr, lambda.tensorData, U.tensorData, dcolU, VT, 2, info, tmpdata+dataSize, poolsize-dataSize*sizeof(DTTYPE));

    if (thistype == 'O') 
    {
	dataSize /= rankDim[0];

	rankDim[0] = mindim;    // min(dimm, dimlr) 

	dataSize *= rankDim[0];

	setFileSize();  //correct
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    

    assert(info == 0);

    return std::tuple<BaseTensor<DTTYPE>, Vector<RDTTYPE>>(U, lambda);
}

/*move the leg to the rightmost and perform singular value decomposition, T = U\lambda V^T  
* VDtype = 'S' or 'A', U is stored locally. 
*/

template<typename DTTYPE> template<typename TSR, typename> 
BLASINT BaseTensor<DTTYPE>::rightSVD(const char &thistype, const uint32_t &index, const char &VDtype, Vector<RDTTYPE> &lambda, TSR &VD, void* pool, const BLASINT &poolsize)
{
#ifdef DEBUG_RIGHTSVD_TIME
    clock_t time_begin = clock();
#endif	

    assert(index < tensorRank && tensorRank >= 2);
    assert(thistype == 'O' || thistype == 'K' || thistype == 'D');
    assert(VDtype == 'A' || VDtype == 'S');

    BLASINT diml = 1;
    for (BLASINT i = 0; i < index; i++) diml *= rankDim[i];

    BLASINT dimm = rankDim[index];

    BLASINT dimr = 1;
    for (BLASINT i = index+1; i < tensorRank; i++) dimr *= rankDim[i];

    const BLASINT dimlr = diml*dimr;

    assert(dimlr*dimm == dataSize);

    const BLASINT mindim = dimm<dimlr?dimm:dimlr;

    BLASINT drowVD;

    if (VDtype == 'A') drowVD = dimm;
    else drowVD = mindim;

    DTTYPE* psour;

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    assert(poolsize >= dataSize*sizeof(DTTYPE));

    BLASINT extdtsize = 0;

    if (thistype == 'K') 
    {
	llmopr::transposeMRto(diml, dimm, dimr, tensorData, tmpdata);
        psour = tmpdata;

	extdtsize = dataSize;
    }
    else
    {
        shiftAfter(index, tensorRank-1, tmpdata, dataSize*sizeof(DTTYPE));
        psour = tensorData;
    }

    char Utype = 'N';
    if (thistype == 'O') Utype = 'O';

    BLASINT info;

    DTTYPE U[16]; // not referenced

    assert(lambda.resize({drowVD}));
    lambda.fill(0);

    VD.resize({drowVD, dimm});

    miolpk::miogesvd(Utype, VDtype, dimlr, dimm, psour, dimm, lambda.tensorData, U, 2, VD.tensorData, dimm, info, tmpdata+extdtsize, poolsize-extdtsize*sizeof(DTTYPE));

    if (thistype == 'O')
    {
	dataSize /= rankDim[tensorRank-1]; 

        rankDim[tensorRank-1] = mindim;    // min(dimm, dimlr)

	dataSize *= rankDim[tensorRank-1];

	setFileSize();   //correct
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif        

#ifdef DEBUG_RIGHTSVD_TIME 
    std::cout<<"Time on rightSVD is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return info;
}

template<typename DTTYPE> 
auto BaseTensor<DTTYPE>::rightSVD(const char &thistype, const uint32_t &index, const char &VDtype, void* pool, const BLASINT &poolsize)
{
    assert(index < tensorRank && tensorRank >= 2);
    assert(thistype == 'O' || thistype == 'K' || thistype == 'D');
    assert(VDtype == 'A' || VDtype == 'S');

    BLASINT diml = 1;
    for (BLASINT i = 0; i < index; i++) diml *= rankDim[i];

    BLASINT dimm = rankDim[index];

    BLASINT dimr = 1;
    for (BLASINT i = index+1; i < tensorRank; i++) dimr *= rankDim[i];

    const BLASINT dimlr = diml*dimr;

    assert(dimlr*dimm == dataSize);

    const BLASINT mindim = dimm<dimlr?dimm:dimlr;

    BLASINT drowVD;

    if (VDtype == 'A') drowVD = dimm;
    else drowVD = mindim;

    DTTYPE* psour;

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    assert(poolsize >= dataSize*sizeof(DTTYPE));

    if (thistype == 'K') 
    {
	llmopr::transposeMRto(diml, dimm, dimr, tensorData, tmpdata);
        psour = tmpdata;
    }
    else
    {
        shiftAfter(index, tensorRank-1, tmpdata, dataSize*sizeof(DTTYPE));
        psour = tensorData;
    }

    char Utype = 'N';
    if (thistype == 'O') Utype = 'O';

    BLASINT info;

    DTTYPE U[16]; // not referenced

    Vector<RDTTYPE> lambda(drowVD);
    lambda.fill(0);

    BaseTensor<DTTYPE> VD({drowVD, dimm});

    miolpk::miogesvd(Utype, VDtype, dimlr, dimm, psour, dimm, lambda.tensorData, U, 2, VD.tensorData, dimm, info, tmpdata+dataSize, poolsize-dataSize*sizeof(DTTYPE));

    if (thistype == 'O')
    {
	dataSize /= rankDim[tensorRank-1]; 

        rankDim[tensorRank-1] = mindim;    // min(dimm, dimlr)

	dataSize *= rankDim[tensorRank-1];

	setFileSize();   //correct
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif        

    assert(info == 0);

    return std::tuple<Vector<RDTTYPE>, BaseTensor<DTTYPE> >(lambda, VD);
}

/*leftSVDD : SVD using gesdd. For more information, see the manual of lapack*/

template<typename DTTYPE> template<typename TSR, typename> 
BLASINT BaseTensor<DTTYPE>::leftSVDD(const char &thistype, const uint32_t &index, const char &SVDtype, TSR &U, Vector<RDTTYPE> &lambda, TSR &VD, void* pool, const BLASINT &poolsize)
{
#ifdef DEBUG_LEFTSVDD_TIME
    clock_t time_begin = clock();
#endif
	
    assert(index < tensorRank);
    assert(thistype == 'K' || thistype == 'D');
    assert(SVDtype == 'A' || SVDtype == 'S'); 

    BLASINT diml = 1;
    for (BLASINT i = 0; i < index; i++) diml *= rankDim[i];

    BLASINT dimm = rankDim[index];

    BLASINT dimr = 1;
    for (BLASINT i = index+1; i < tensorRank; i++) dimr *= rankDim[i];

    const BLASINT dimlr = diml*dimr;

    assert(dimlr*dimm == dataSize);

    const BLASINT mindim = dimm<dimlr?dimm:dimlr;

    DTTYPE* psour;

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    assert(dataSize < poolsize);

    BLASINT extdtsize = 0;

    if (thistype == 'K') 
    {
	llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);
        psour = tmpdata;

        extdtsize = dataSize;	
    }
    else
    {
        shiftBefore(index, 0, tmpdata, dataSize*sizeof(DTTYPE));
        psour = tensorData;
    }

    BLASINT info;

    assert(lambda.resize({mindim}));
    lambda.fill(0.0);

    if (SVDtype == 'A') 
    {
        assert(U.resize({dimm, dimm}));
	assert(VD.resize({dimlr, dimlr}));
    }
    else
    {
        assert(U.resize({dimm, mindim}));	    
	assert(VD.resize({mindim, dimlr}));
    }

    miolpk::miogesdd(SVDtype, dimm, dimlr, psour, dimlr, lambda.tensorData, U.tensorData, U.getRankDim(U.getTensorRank()-1), VD.tensorData, dimlr, info, tmpdata+extdtsize, poolsize-extdtsize*sizeof(DTTYPE));

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif    

#ifdef DEBUG_LEFTSVDD_TIME
    std::cout<<"Time on leftSVDD is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return info;
}

/*rightSDD : SVD using gesdd*/

template<typename DTTYPE> template<typename TSR, typename>
BLASINT BaseTensor<DTTYPE>::rightSVDD(const char &thistype, const uint32_t &index, const char &SVDtype, TSR &U, Vector<RDTTYPE> &lambda, TSR &VD, void* pool, const BLASINT &poolsize)
{
#ifdef DEBUG_RIGHTSVDD_TIME
    clock_t time_begin = clock();
#endif

    assert(index < tensorRank && tensorRank >= 2);
    assert(thistype == 'K' || thistype == 'D');
    assert(SVDtype == 'A' || SVDtype == 'S');

    BLASINT diml = 1;
    for (BLASINT i = 0; i < index; i++) diml *= rankDim[i];

    BLASINT dimm = rankDim[index];

    BLASINT dimr = 1;
    for (BLASINT i = index+1; i < tensorRank; i++) dimr *= rankDim[i];

    const BLASINT dimlr = diml*dimr;

    assert(dimlr*dimm == dataSize);

    const BLASINT mindim = dimm<dimlr?dimm:dimlr;

    DTTYPE* psour;

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    assert(poolsize >= dataSize*sizeof(DTTYPE));

    BLASINT extdtsize = 0;

    if (thistype == 'K')
    {
        llmopr::transposeMRto(diml, dimm, dimr, tensorData, tmpdata);
        psour = tmpdata;

        extdtsize = dataSize;
    }
    else
    {
        shiftAfter(index, tensorRank-1, tmpdata, dataSize*sizeof(DTTYPE));
        psour = tensorData;
    }

    BLASINT info;

    assert(lambda.resize({mindim}));
    lambda.fill(0.0);

    if (SVDtype == 'A')
    {
	assert(U.resize({dimlr, dimlr}));    
        assert(VD.resize({dimm, dimm}));
    }
    else
    {
	assert(U.resize({dimlr, mindim}));
        assert(VD.resize({mindim, dimm}));	
    }

    miolpk::miogesdd(SVDtype, dimlr, dimm, psour, dimm, lambda.tensorData, U.tensorData, U.getRankDim(U.getTensorRank()-1), VD.tensorData, dimm, info, tmpdata+extdtsize, poolsize-extdtsize*sizeof(DTTYPE));

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif

#ifdef DEBUG_RIGHTSVDD_TIME
    std::cout<<"Time on rightSVDD is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return info;
}

/*high order SVD, all U are unitary matrix, core tensor is stored locally and no truncation*/

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::highOrderLSVD(const uint32_t &num, const uint32_t* index, DTTYPE**U, const BLASINT* Usize, void* lambdamax, const BLASINT &lmbdsize, void* pool, const BLASINT &poolsize)
{
    RDTTYPE* startlambda = reinterpret_cast<RDTTYPE*>(lambdamax);

#ifdef CHECK_VECTOR_SIZE
    BLASINT lmbdsizesum = 0;
#endif

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    for (BLASINT n = 0; n < num; n++)    
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        const BLASINT dimlr = diml*dimr;

        assert(dimlr*dimm == dataSize);

        BLASINT mindim = dimm<dimlr?dimm:dimlr;

/*B_jik <= A_ijk*/

        llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);

        BLASINT info;

        DTTYPE VD[16];// just a variable, not referenced, 

#ifdef CHECK_VECTOR_SIZE
	assert(lmbdsizesum+dimm <= lmbdsize);
	assert(Usize[n] >= dimm*dimm);

        lmbdsizesum += dimm;  // sum here
#endif	

        memset(startlambda, 0, dimm*sizeof(RDTTYPE));  // dimtrlbd[n] singular values are kept, including those zero ones, it may be larger than mindim.

        miolpk::miogesvd('A', 'N', dimm, dimlr, tmpdata, dimlr, startlambda, U[n], dimm, VD, 2, info, tmpdata+dataSize, poolsize-dataSize*sizeof(DTTYPE));

	startlambda += dimm;
    }

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        for (BLASINT l = 0; l < diml; l++)
        {
            DTTYPE* psour = tensorData + l*dimm*dimr;
            gemm('C', 'N', dimm, dimr, dimm, 1.0, U[n], dimm, psour, dimr, 0.0, tmpdata, dimr);
            memcpy(psour, tmpdata, dimm*dimr*sizeof(DTTYPE));
        }
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif        
}

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::highOrderLSVD(const uint32_t &num, const uint32_t* index, BaseTensor**U, Vector<RDTTYPE>** lambda, void* pool, const BLASINT &poolsize)
{
    for (uint32_t i = 0; i < num; i++) assert(index[i] < tensorRank);    

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    for (BLASINT n = 0; n < num; n++)    
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        U[n]->resize({dimm, dimm});
       
        const BLASINT dimlr = diml*dimr;

        assert(dimlr*dimm == dataSize);

        BLASINT mindim = dimm<dimlr?dimm:dimlr;

/*B_jik <= A_ijk*/

        llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);

        BLASINT info;

        DTTYPE VD[16];// just a variable, not referenced, 

	RDTTYPE* lmbdn;
	BLASINT extlmbdnsize;

        RDTTYPE* rtmpdata = reinterpret_cast<RDTTYPE*> (tmpdata+dataSize); 

	if (lambda == nullptr)
	{
	    lmbdn = rtmpdata;
	    extlmbdnsize = dimm;
	}
	else
	{
	    lambda[n]->resize({dimm});	
	    lambda[n]->fill(0.0);

            lmbdn = lambda[n]->tensorData;		
	    extlmbdnsize = 0;
        }

	miolpk::miogesvd('A', 'N', dimm, dimlr, tmpdata, dimlr, lmbdn, U[n]->tensorData, dimm, VD, 2, info, rtmpdata+extlmbdnsize, poolsize-dataSize*sizeof(DTTYPE)-extlmbdnsize*sizeof(RDTTYPE));
    }

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        for (BLASINT l = 0; l < diml; l++)
        {
            DTTYPE* psour = tensorData + l*dimm*dimr;
            gemm('C', 'N', dimm, dimr, dimm, 1.0, U[n]->tensorData, dimm, psour, dimr, 0.0, tmpdata, dimr);
            memcpy(psour, tmpdata, dimm*dimr*sizeof(DTTYPE));
        }
    }

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif        
}

/*high order SVD, all U are unitary matrix, core tensor is stored locally and truncations are performed  according to dimtru[] 
 *           U   ==> no truncation
 * core tensor   ==> truncated    
*/

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::highOrderLSVD(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, DTTYPE**U, const BLASINT* Usize, void* lambdamax, const BLASINT &lmbdsize, void* pool, const BLASINT &poolsize)
{
    RDTTYPE* startlambda = reinterpret_cast<RDTTYPE*>(lambdamax);

#ifdef CHECK_VECTOR_SIZE
    BLASINT lmbdsizesum = 0;
#endif

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        const BLASINT dimlr = diml*dimr;

        assert(dimlr*dimm == dataSize);

        const BLASINT mindim = dimm<dimlr?dimm:dimlr;

/*B_jik <= A_ijk*/

        llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);

        BLASINT info;

        DTTYPE VD[16];// just a variable, not referenced, 

#ifdef CHECK_VECTOR_SIZE
	assert(lmbdsizesum+dimm <= lmbdsize);
	assert(Usize[n] >= dimm*dimm);  // no truncation in U, ==> U[dimm,dimm]

        lmbdsizesum += dimm;  // sum here
#endif	

        memset(startlambda, 0, dimm*sizeof(RDTTYPE));

        miolpk::miogesvd('A', 'N', dimm, dimlr, tmpdata, dimlr, startlambda, U[n], dimm, VD, 2, info, tmpdata+dataSize, poolsize-dataSize*sizeof(DTTYPE));

	startlambda += dimm;
    }

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        assert(dimm >= dimtru[n]);

        for (BLASINT l = 0; l < diml; l++)
        {
            DTTYPE* psour = tensorData + l*dimm*dimr;
            DTTYPE* pdest = tmpdata + l*dimtru[n]*dimr;
            gemm('C', 'N', dimtru[n], dimr, dimm, 1.0, U[n], dimm, psour, dimr, 0.0, pdest, dimr);
        }

        memcpy(tensorData, tmpdata, diml*dimtru[n]*dimr*sizeof(DTTYPE));

        rankDim[index[n]] = dimtru[n];
    }

    dataSize = 1;
    for (BLASINT i = 0; i < tensorRank; i++) dataSize *= rankDim[i];

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif        
}

template<typename DTTYPE>  
void BaseTensor<DTTYPE>::highOrderLSVD(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, BaseTensor** U, Vector<RDTTYPE>** lambda, void* pool, const BLASINT &poolsize)
{
    for (uint32_t i = 0; i < num; i++) assert(index[i] < tensorRank);    

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        U[n]->resize({dimm, dimm});

        const BLASINT dimlr = diml*dimr;

        assert(dimlr*dimm == dataSize);

        const BLASINT mindim = dimm<dimlr?dimm:dimlr;

/*B_jik <= A_ijk*/

        llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);

        BLASINT info;

        DTTYPE VD[16];// just a variable, not referenced, 

	RDTTYPE* lmbdn;
	BLASINT extlmbdnsize;

        RDTTYPE* rtmpdata = reinterpret_cast<RDTTYPE*> (tmpdata+dataSize);

	if (lambda == nullptr)
	{
	    lmbdn = rtmpdata;
	    extlmbdnsize = dimm;
	}
	else
	{
	    lambda[n]->resize({dimm});	
	    lambda[n]->fill(0.0);

            lmbdn = lambda[n]->tensorData;		
	    extlmbdnsize = 0;
        }

	miolpk::miogesvd('A', 'N', dimm, dimlr, tmpdata, dimlr, lmbdn, U[n]->tensorData, dimm, VD, 2, info, rtmpdata+extlmbdnsize, poolsize-dataSize*sizeof(DTTYPE)-extlmbdnsize*sizeof(RDTTYPE));
        assert(info == 0);
    }

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        assert(dimm >= dimtru[n]);

        for (BLASINT l = 0; l < diml; l++)
        {
            DTTYPE* psour = tensorData + l*dimm*dimr;
            DTTYPE* pdest = tmpdata + l*dimtru[n]*dimr;
            gemm('C', 'N', dimtru[n], dimr, dimm, 1.0, U[n]->tensorData, dimm, psour, dimr, 0.0, pdest, dimr);
        }

        memcpy(tensorData, tmpdata, diml*dimtru[n]*dimr*sizeof(DTTYPE));

        rankDim[index[n]] = dimtru[n];
    }

    dataSize = 1;
    for (BLASINT i = 0; i < tensorRank; i++) dataSize *= rankDim[i];

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif        
}

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::highOrderLSVDx(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, DTTYPE**U, const BLASINT* Usize, void* lambdamax, const BLASINT &lmbdsize, const BLASINT* dimtrlbd, void* pool, const BLASINT &poolsize)
{
    RDTTYPE* startlambda = reinterpret_cast<RDTTYPE*>(lambdamax);

#ifdef CHECK_VECTOR_SIZE
    BLASINT lmbdsizesum = 0;
#endif

    DTTYPE* tmpdata = reinterpret_cast<DTTYPE*>(pool);

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        const BLASINT dimm = rankDim[index[n]];

        const BLASINT dimlr = diml*dimr;

	assert(dimm*dimlr == dataSize);

        const BLASINT mindim = dimm<dimlr?dimm:dimlr;

/*B_jik <= A_ijk*/

        llmopr::transposeLMto(diml, dimm, dimr, tensorData, tmpdata);

        BLASINT info;

        DTTYPE VD[16];// just a variable, not referenced, 

	RDTTYPE vl, vu;

        BLASINT ns;	    

#ifdef CHECK_VECTOR_SIZE
	assert(lmbdsizesum+mindim <= lmbdsize && lmbdsizesum+dimtrlbd[n] <= lmbdsize);
	assert(Usize[n] >= dimm*dimtru[n]);  // no truncation in U, ==> U[dimm,dimm]

        lmbdsizesum += dimtrlbd[n];  // sum here
#endif	

        memset(startlambda, 0, dimtrlbd[n]*sizeof(RDTTYPE));   // only dimtrlbd[n] singular values are kept, it can be larger than dimtru[n]

	miolpk::miogesvdx('V', 'N', 'I', dimm, dimlr, tmpdata, dimlr, vl, vu, 1, dimtru[n], ns, startlambda, U[n], dimtru[n], VD, 2, info, tmpdata+dataSize, poolsize-dataSize*sizeof(DTTYPE));
        assert(info == 0 && ns == dimtru[n]);

	startlambda += dimtrlbd[n];
    }

    for (BLASINT n = 0; n < num; n++)
    {
        BLASINT diml = 1;
        for (BLASINT i = 0; i < index[n]; i++) diml *= rankDim[i];

        BLASINT dimr = 1;
        for (BLASINT i = index[n]+1; i < tensorRank; i++) dimr *= rankDim[i];

        BLASINT dimm = rankDim[index[n]];

        assert(dimm >= dimtru[n]);

        for (BLASINT l = 0; l < diml; l++)
        {
            DTTYPE* psour = tensorData + l*dimm*dimr;
            DTTYPE* pdest = tmpdata + l*dimtru[n]*dimr;
            gemm('C', 'N', dimtru[n], dimr, dimm, 1.0, U[n], dimtru[n], psour, dimr, 0.0, pdest, dimr);
        }

        memcpy(tensorData, tmpdata, diml*dimtru[n]*dimr*sizeof(DTTYPE));

        rankDim[index[n]] = dimtru[n];
    }

    dataSize = 1;
    for (BLASINT i = 0; i < tensorRank; i++) dataSize *= rankDim[i];

    setFileSize();

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif        
}

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::highOrderLSVDx(const uint32_t &num, const uint32_t* index, const BLASINT* dimtru, BaseTensor** U, Vector<RDTTYPE>** lambda, void* pool, const BLASINT &poolsize)
{
    assert(0);
}

template<typename DTTYPE> template<typename RDT, typename> auto BaseTensor<DTTYPE>::normalize(const BLASINT &ntype, const RDT &nr)
{
    if (ntype == 0)	
    {
        auto rnm = nrm2(dataSize, tensorData, 1); 
        DTTYPE invr = nr/rnm;

        scal(dataSize, invr, tensorData, 1);    

        return rnm;
    }
    else if (ntype == 1)  // +nrm or -nrm for the maximum absolute one
    {
	auto maxcs = dabs(tensorData[0]);

        for (BLASINT i = 1; i < dataSize; i++)
        {
            if (dabs(tensorData[i]) > maxcs) maxcs = dabs(tensorData[i]);
        }

	DTTYPE invmaxcs = nr/maxcs;

	scal(dataSize, invmaxcs, tensorData, 1);
   
        return maxcs;      	
    }
    else 
    {
	assert(0);   
        abort();	
    }
}

template<typename DTTYPE> template<typename RDT, typename> auto BaseTensor<DTTYPE>::norm(const BLASINT &ntype)
{
    if (ntype == 0)	
    {
        return nrm2(dataSize, tensorData, 1); 
    }
    else if (ntype == 1)  // +nrm or -nrm for the maximum absolute one
    {
	auto maxcs = dabs(tensorData[0]);

        for (BLASINT i = 1; i < dataSize; i++)
        {
            if (dabs(tensorData[i]) > maxcs) maxcs = dabs(tensorData[i]);
        }
   
        return maxcs;      	
    }
    else 
    {
	assert(0);   
        abort();	
    }
}

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::cconj() noexcept
{
    if (typeid(RDTTYPE) != typeid(DTTYPE))
    {	    
        for (BLASINT i = 0; i < dataSize; i++) tensorData[i] = conj(tensorData[i]);	
    }
}

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::saveTensorToFile(const char* filename) const
{
    std::ofstream file;
    file.open(filename, ios::trunc | ios::out);
    if (!file) return false;

    bool savebool = saveTensorToStream(file);

    file.close();

    return savebool;
}

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::saveTensorToStream(std::ofstream &ofs) const
{
    ofs.write((const char*)&fileSize, sizeof(long));

    ofs.write((const char*)&tensorRank, sizeof(uint32_t));

    for (BLASINT i = 0; i < tensorRank; i++) ofs.write((const char*)&rankDim[i], sizeof(BLASINT));

    ofs.write((const char*)&dataSize, sizeof(BLASINT));

    ofs.write((const char*)tensorData, dataSize*sizeof(DTTYPE));

    return true;
}

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::readTensorFromStream(ifstream &ifs)
{
    ifs.read((char*)&fileSize, sizeof(long));

    ifs.read((char*)&tensorRank, sizeof(uint32_t));

    for (uint32_t i = 0; i < tensorRank; i++) ifs.read((char*)&rankDim[i], sizeof(BLASINT));

    ifs.read((char*)&dataSize, sizeof(BLASINT));

    tensorData = new(std::nothrow) DTTYPE[dataSize];
    assert(tensorData);

    ifs.read((char*)tensorData, dataSize*sizeof(DTTYPE));

    return true;
}

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::readTensorFromMemory(char* buffer)
{
    char* nextbuffer = buffer;

    fileSize = *((long*)nextbuffer);
    nextbuffer += sizeof(long);

    tensorRank = *((uint32_t*)nextbuffer);
    nextbuffer += sizeof(uint32_t);

    for (BLASINT i = 0; i < tensorRank; i++) 
    {
        rankDim[i] = *((BLASINT*)nextbuffer);
        nextbuffer += sizeof(BLASINT);
    }

    dataSize = *((BLASINT*)nextbuffer);
    nextbuffer += sizeof(BLASINT);

    tensorData = new(std::nothrow) DTTYPE[dataSize];
    assert(tensorData);

    memcpy((char*)tensorData, nextbuffer, dataSize*sizeof(DTTYPE));

    nextbuffer += dataSize*sizeof(DTTYPE);

    return true;
}

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::printTensorToFile(const char* suffixname, const streamsize &width, const char &type)
{
/*
 * If type == 'A', print tensorRank, rankDim, tensorData;
 * if type == 'D', print tensorData only.
 * */

    assert(type == 'A' || type == 'D');

    char filename[120];

    stringname(110, filename, suffixname, "R.dat");

    std::ofstream file;
    file.open(filename, ios::trunc | ios::out);
    if (!file) return false;

    file.precision(width);
    file.width(width);

    if (type == 'A') 
    {
        file << tensorRank <<"  ";
        for (uint32_t i = 0; i < tensorRank; i++) file << rankDim[i] <<"  ";
        file << std::endl;	
    }

    for (BLASINT i = 0; i < dataSize; i++) file << real(tensorData[i]) << "  ";
    file.close();

    stringname(110, filename, suffixname, "I.dat");

    file.open(filename, ios::trunc | ios::out);
    if (!file) return false;

    file.precision(width);
    file.width(width);

    if (type == 'A') 
    {
        file << tensorRank <<"  ";
        for (uint32_t i = 0; i < tensorRank; i++) file << rankDim[i] << "  ";
        file << std::endl;	
    }

    for (BLASINT i = 0; i < dataSize; i++) file << imag(tensorData[i]) << "  ";

    file.close();

    return true;
}

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::checkBaseTensor() const
{
    assert(tensorRank <= MAXTRK && tensorRank > 0);

    for (uint32_t i = 0; i < tensorRank; i++) assert(rankDim[i] > 0);

    BLASINT td = 1;
    for (uint32_t i = 0; i < tensorRank; i++) td *= rankDim[i];
    assert(td == dataSize);
}

/*private functions*/

template<typename DTTYPE> 
void BaseTensor<DTTYPE>::setFileSize() noexcept
{
    fileSize = sizeof(long) + sizeof(uint32_t) + (tensorRank+1)*sizeof(BLASINT);

    fileSize += dataSize*sizeof(DTTYPE);    
}

/*shift index "il" before "iprev", require : il > iprev*/
/*将il插到iprev前面*/

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::shiftBefore_(const uint32_t &il, const uint32_t &iprev, void* tmpdata, const BLASINT &tdsize)
{
#ifdef DEBUG_SHIFT   
    std::cout<<"     ==start shiftBefore iitt  :: " << std::endl;
    clock_t time_begin = clock();
#endif

    assert(iprev < il);    

    BLASINT Dl = 1; 
    for (unsigned i = 0; i < iprev; i++) Dl *= rankDim[i];
    
    BLASINT Dc = 1;
    for (unsigned i = iprev; i < il; i++) Dc *= rankDim[i];

    BLASINT Dm = rankDim[il];

    BLASINT Dr = ((dataSize/Dl)/Dc)/Dm;

    BLASINT dcmr = Dc*Dm*Dr;

    DTTYPE* localtmpdata = reinterpret_cast<DTTYPE*> (tmpdata);

    if (tdsize >= dataSize*sizeof(DTTYPE))
    {
        if (Dr == 1)
        {
	    #pragma omp parallel for schedule(dynamic, 1)	
            for (BLASINT l = 0; l < Dl; l++)
            {
                DTTYPE* pdest = tensorData + l*dcmr;
                DTTYPE* psour = localtmpdata + l*dcmr;

		copy(dcmr, pdest, 1, psour, 1);

                llmopr::fastTransposeTo(Dc, Dm, psour, Dm, pdest, Dc);
            }
        }
        else
        {
	    #pragma omp parallel for schedule(dynamic, 1) 	
            for (BLASINT l = 0; l < Dl; l++)
            {
                DTTYPE* pdest = tensorData + l*dcmr;
                DTTYPE* psour = localtmpdata + l*dcmr;

		copy(dcmr, pdest, 1, psour, 1);

                llmopr::transposeLMto(Dc, Dm, Dr, psour, pdest);
            }
        }
    }
    else if (tdsize >= dcmr*sizeof(DTTYPE))
    {
	if (Dr == 1)
	{
	    for (BLASINT l = 0; l < Dl; l++)
            {
                DTTYPE* pdest = tensorData + l*dcmr;

                memcpy(localtmpdata, pdest, dcmr*sizeof(DTTYPE));

                llmopr::fastTransposeTo(Dc, Dm, localtmpdata, Dm, pdest, Dc);
            }
	}
	else
	{	
	    for (BLASINT l = 0; l < Dl; l++)
            {
                DTTYPE* pdest = tensorData + l*dcmr;
           
	        memcpy(localtmpdata, pdest, dcmr*sizeof(DTTYPE)); 

                llmopr::transposeLMto(Dc, Dm, Dr, localtmpdata, pdest);
	    }
        }
    }
    else
    {
        assert(0); // not finished yet!	   
        return false;	
    }

    for (uint32_t i = il; i > iprev; i--) rankDim[i] = rankDim[i-1]; 

    rankDim[iprev] = Dm;

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif 

#ifdef DEBUG_SHIFT
    std::cout<<"     ==Time on shiftBefore(iitt) is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return true;    
}

/*shift index "il" before "iprev", require : il > iprev*/
/*将il插到iprev前面*/

template<typename DTTYPE>
bool BaseTensor<DTTYPE>::shiftBefore_(const uint32_t &il, const uint32_t &iprev, BaseTensor<DTTYPE> &bt) const
{
#ifdef DEBUG_SHIFT   
    std::cout<<"     ==start shiftBefore iib  :: " << std::endl;
    clock_t time_begin = clock();
#endif

    assert(iprev < il);

    bt.resize(tensorRank, rankDim);

    BLASINT Dl = 1;
    for (unsigned i = 0; i < iprev; i++) Dl *= rankDim[i];

    BLASINT Dc = 1;
    for (unsigned i = iprev; i < il; i++) Dc *= rankDim[i];

    BLASINT Dm = rankDim[il];

    BLASINT Dr = ((dataSize/Dl)/Dc)/Dm;

    BLASINT dcmr = Dc*Dm*Dr;

    if (Dr == 1)
    {
        #pragma omp parallel for schedule(dynamic, 1)   
        for (BLASINT l = 0; l < Dl; l++)
        {
            DTTYPE* psour = tensorData + l*dcmr;
            DTTYPE* pdest = bt.tensorData + l*dcmr;

            llmopr::fastTransposeTo(Dc, Dm, psour, Dm, pdest, Dc);
        }
    }
    else
    {
        #pragma omp parallel for  schedule(dynamic, 1)  
        for (BLASINT l = 0; l < Dl; l++)
        {
            DTTYPE* psour = tensorData + l*dcmr;
            DTTYPE* pdest = bt.tensorData + l*dcmr;

            llmopr::transposeLMto(Dc, Dm, Dr, psour, pdest);
        }
    }

    for (uint32_t i = il; i > iprev; i--) bt.rankDim[i] = bt.rankDim[i-1];

    bt.rankDim[iprev] = Dm;

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif

#ifdef DEBUG_SHIFT
    std::cout<<"     ==Time on shiftBefore(iib) is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return true;
}

/*shift index "il" after "iback", require  il < iback*/
/*将il插到iback后面*/

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::shiftAfter_(const uint32_t &il, const uint32_t &iback, void* tmpdata, const BLASINT &tdsize)
{
#ifdef DEBUG_SHIFT   
    std::cout<<"     ==start shiftAfter iitt  :: " << std::endl;
    clock_t time_begin = clock();
#endif
	
    assert(il < iback);

    BLASINT Dl = 1;
    for (BLASINT l = 0; l < il; l++) Dl *= rankDim[l];

    BLASINT Dm = rankDim[il];

    BLASINT Dc = 1;
    for (BLASINT c = il+1; c <= iback; c++) Dc *= rankDim[c];

    BLASINT Dr = ((dataSize/Dl)/Dm)/Dc;

    DTTYPE* localtmpdata = reinterpret_cast<DTTYPE*> (tmpdata);

    BLASINT dmcr = Dm*Dc*Dr;

    if (tdsize >= Dl*dmcr*sizeof(DTTYPE))
    {
        if (Dr == 1)
        {
	    #pragma omp parallel for schedule(dynamic, 1)	
            for (BLASINT l = 0; l < Dl; l++) llmopr::fastTransposeTo(Dm, Dc, tensorData+l*dmcr, Dc, localtmpdata+l*dmcr, Dm);

            copy(getDataSize(), localtmpdata, 1, tensorData, 1);
        }
        else
        {
	    #pragma omp parallel for schedule(dynamic, 1)	
            for (BLASINT l = 0; l < Dl; l++)
            {
                DTTYPE* pdest = tensorData + l*dmcr;
                DTTYPE* psour = localtmpdata + l*dmcr;

                copy(dmcr, pdest, 1, psour, 1);

                llmopr::transposeLMto(Dm, Dc, Dr, psour, pdest);
            }
        }
    }
    else if (tdsize >= dmcr*sizeof(DTTYPE))
    {
	if (Dr == 1)
	{
	    for (BLASINT l = 0; l < Dl; l++)
	    {
		DTTYPE* pdest = tensorData + l*dmcr;

                memcpy(localtmpdata, pdest, dmcr*sizeof(DTTYPE));

                llmopr::fastTransposeTo(Dm, Dc, localtmpdata, Dc, pdest, Dm);
	    }	    
	}
	else
	{	
            for (BLASINT l = 0; l < Dl; l++)
            {
                DTTYPE* pdest = tensorData + l*dmcr;

	        memcpy(localtmpdata, pdest, dmcr*sizeof(DTTYPE));
            
	        llmopr::transposeLMto(Dm, Dc, Dr, localtmpdata, pdest);
	    }
        } 
    }
    else
    {
	assert(0);
        return false;	
    }

    for (uint32_t i = il; i < iback; i++) rankDim[i] = rankDim[i+1];

    rankDim[iback] = Dm; 

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif       

#ifdef DEBUG_SHIFT
    std::cout<<"     ==Time on shiftAfter(iitt) is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return true; 
}

/*shift index "il" after "iback", require  il < iback*/
/*将il插到iback后面*/

template<typename DTTYPE> 
bool BaseTensor<DTTYPE>::shiftAfter_(const uint32_t &il, const uint32_t &iback, BaseTensor<DTTYPE> &bt) const
{
#ifdef DEBUG_SHIFT   
    std::cout<<"     ==start shiftAfter iib  :: " << std::endl;
    clock_t time_begin = clock();
#endif

    assert(il < iback);

    bt.resize(tensorRank, rankDim);

    BLASINT Dl = 1;
    for (BLASINT l = 0; l < il; l++) Dl *= rankDim[l];

    BLASINT Dm = rankDim[il];

    BLASINT Dc = 1;
    for (BLASINT c = il+1; c <= iback; c++) Dc *= rankDim[c];

    BLASINT Dr = ((dataSize/Dl)/Dm)/Dc;

    BLASINT dmcr = Dm*Dc*Dr;

    if (Dr == 1)
    {
        #pragma omp parallel for schedule(dynamic, 1)	
        for (BLASINT l = 0; l < Dl; l++) llmopr::fastTransposeTo(Dm, Dc, tensorData+l*dmcr, Dc, bt.tensorData+l*dmcr, Dm);
    }
    else
    {
        #pragma omp parallel for schedule(dynamic, 1)	
        for (BLASINT l = 0; l < Dl; l++)
        {
            DTTYPE* psour = tensorData + l*dmcr;
            DTTYPE* pdest = bt.tensorData + l*dmcr;

            llmopr::transposeLMto(Dm, Dc, Dr, psour, pdest);
        }
    }

    for (uint32_t i = il; i < iback; i++) bt.rankDim[i] = bt.rankDim[i+1];

    bt.rankDim[iback] = Dm; 

#ifdef CHECK_BASETENSOR
    checkBaseTensor();
#endif       

#ifdef DEBUG_SHIFT
    std::cout<<"     ==Time on shiftAfter(iib) is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl;
#endif

    return true; 
}

template<typename DT> void tensorContraction(const char &cttype, const BaseTensor<DT> &btl, const uint32_t &il, const BaseTensor<DT> &btr, const uint32_t &ir, BaseTensor<DT> &bc, void* tmpdata, const BLASINT &tdsize)
{
    assert(&btl != &btr);

#ifdef DEBUG_BASETENSOR_TIME_cbibitt   
    std::cout<<std::endl<<"***start cbibitt  :: " <<std::endl; 
    clock_t time_begin = clock();
#endif

    bc.setTensorStruct(btl, il, btr, ir, cttype);

    if (bc.getMaxDataSize() < bc.getDataSize()) bc.reSetMaxDataSize(bc.getDataSize()); 

/*left dim*/
		    
    BLASINT Dll = 1;
    for (uint32_t i = 0; i < il; i++) Dll *= btl.getRankDim(i);
    
    BLASINT Dlm = btl.getRankDim(il); 

    BLASINT Dlr = (btl.getDataSize()/Dll)/Dlm;

/*right dim*/

    BLASINT Drl = 1;
    for (uint32_t i = 0; i < ir; i++) Drl *= btr.getRankDim(i);

    BLASINT Drm = btr.getRankDim(ir);

    BLASINT Drr = (btr.getDataSize()/Drl)/Drm;

    assert(bc.getDataSize() == Dll*Dlr*Drl*Drr);
    assert(Dlm == Drm);

    DT* localtmpdata = reinterpret_cast<DT*>(tmpdata);

/*contraction*/

    if (cttype == 'S')
    {
        char transl;
        char transr = 'T';

        BLASINT ldl;
        BLASINT ldr;

        if (il == 0)
        {
	    transl = 'T';
            ldl = Dll*Dlr;	    
        }
        else if (il+1 == btl.getTensorRank())
        {
	    transl = 'N';
            ldl = Dlm;	    
        }
        else 
        {
	    BLASINT dlmr = Dlm*Dlr;

            if (tdsize >= btl.getDataSize()*sizeof(DT))
            {
	        #pragma omp parallel for schedule(dynamic, 1)	
                for (BLASINT l = 0; l < Dll; l++) llmopr::fastTransposeTo(Dlm, Dlr, btl.tensorData+l*dlmr, Dlr, localtmpdata+l*dlmr, Dlm);

	        copy(btl.getDataSize(), localtmpdata, 1, btl.tensorData, 1);
	    }
            else if (tdsize >= dlmr*sizeof(DT))
            {
                for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlm, Dlr, btl.tensorData+l*dlmr, localtmpdata); 
	    }
	    else
	    {
	        std::cout << "warning : tmpdata is too small, tdsize is " << tdsize << ", it should be at leat " << dlmr*sizeof(DT) <<std::endl;
	
                for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlm, Dlr, btl.tensorData+l*dlmr);		
	    }

	    transl = 'N';
	    ldl = Dlm;
        }

        const DT* psour = btr.tensorData;

        if (ir == 0)
        {
	    transr = 'N';
            ldr = Drl*Drr;	    
        }
        else if (ir+1 == btr.getTensorRank())
        {
	    transr = 'T';
            ldr = Drm;	    
        }
        else 
        {
	    BLASINT drmr = Drm*Drr;

            if (tdsize >= btr.getDataSize()*sizeof(DT))
            {
	        #pragma omp parallel for schedule(dynamic, 1)	
                for (BLASINT l = 0; l < Drl; l++) llmopr::fastTransposeTo(Drm, Drr, btr.tensorData+l*drmr, Drr, localtmpdata+l*drmr, Drm);

                psour = localtmpdata;	    
	    }
            else if (tdsize >= drmr*sizeof(DT))
	    {
	        for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drm, Drr, btr.tensorData+l*drmr, localtmpdata);
	    }
            else
	    {
	        std::cout << "warning : tmpdata is too small, tdsize is " << tdsize << ", it should be at least "<<drmr*sizeof(DT)<<std::endl;	
            
                for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drm, Drr, btr.tensorData+l*drmr);	
	    }

	    transr = 'T';
	    ldr = Drm;
        }  

        gemm(transl, transr, Dll*Dlr, Drl*Drr, Dlm, 1.0, btl.tensorData, ldl, psour, ldr, 0.0, bc.tensorData, Drl*Drr);

/*restore the data in btr*/

        if (ir != 0 && ir+1 != btr.getTensorRank() && tdsize < btr.getDataSize()*sizeof(DT)) 
        {
            BLASINT drmr = Drm*Drr;
        
            if (tdsize >= drmr*sizeof(DT)) 
	    {
	        for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drr, Drm, btr.tensorData+l*drmr, localtmpdata);	
	    }	
	    else 
	    {
                for (BLASINT l = 0; l < Drl; l++) llmopr::transposeOnsite(Drr, Drm, btr.tensorData+l*drmr);		
	    }
        }

/*restore the data in btl*/

        if (il != 0 && il+1 != btl.getTensorRank())
        {
	    BLASINT dlmr = Dlm*Dlr;

    	    if (tdsize >= btl.getDataSize()*sizeof(DT))
	    {
	        #pragma omp parallel for schedule(dynamic, 1)	
                for (BLASINT l = 0; l < Dll; l++) llmopr::fastTransposeTo(Dlr, Dlm, btl.tensorData+l*dlmr, Dlm, localtmpdata+l*dlmr, Dlr);

	        copy(btl.getDataSize(), localtmpdata, 1, btl.tensorData, 1);	    
	    }
	    else if (tdsize >= dlmr*sizeof(DT))
	    {
	        for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlr, Dlm, btl.tensorData+l*dlmr, localtmpdata);
	    }
	    else
	    {
	        for (BLASINT l = 0; l < Dll; l++) llmopr::transposeOnsite(Dlr, Dlm, btl.tensorData+l*dlmr);
	    }
        }
    }
    else if (cttype == 'D')
    {
        if (Drl == 1 && Dll == 1) 
	{
	    gemm('T', 'N', Dlr, Drr, Dlm, 1.0, btl.tensorData, Dlr, btr.tensorData, Drr, 0.0, bc.tensorData, Drr);
        }
    	else if (Drl == 1 && Dlr == 1) 
	{
	    gemm('N', 'N', Dll, Drr, Dlm, 1.0, btl.tensorData, Dlm, btr.tensorData, Drr, 0.0, bc.tensorData, Drr);	    
        }
    	else if (Drl == 1 && tdsize >= btl.getDataSize()*sizeof(DT))
	{
	    llmopr::transposeMRto(Dll, Dlm, Dlr, btl.tensorData, localtmpdata);
	    gemm('N', 'N', Dll*Dlr, Drr, Dlm, 1.0, localtmpdata, Dlm, btr.tensorData, Drr, 0.0, bc.tensorData, Drr);	    
        }
	else if (Dlr == 1 && Drr == 1) 
	{
	    gemm('N', 'T', Dll, Drl, Dlm, 1.0, btl.tensorData, Dlm, btr.tensorData, Drm, 0.0, bc.tensorData, Drl);
	}
	else if (Dlr == 1 && tdsize >= btr.getDataSize()*sizeof(DT))
	{
	    llmopr::transposeLMto(Drl, Drm, Drr, btr.tensorData, localtmpdata);
	    gemm('N', 'N', Dll, Drl*Drr, Dlm, 1.0, btl.tensorData, Dlm, localtmpdata, Drl*Drr, 0.0, bc.tensorData, Drl*Drr);	    
	}
	else if (Dll == 1 && Drr == 1) 
	{
	    gemm('N', 'N', Drl, Dlr, Dlm, 1.0, btl.tensorData, Dlm, btr.tensorData, Drm, 0.0, bc.tensorData, Dlr);
	}
        else if (Drr == 1)
	{
	    #pragma omp parallel for  schedule(dynamic, 1) 	
	    for (BLASINT l = 0; l < Dll; l++)
	    {
		const DT* ltdata = btl.tensorData+l*Dlm*Dlr;
		DT* localdata = bc.tensorData+l*Drl*Drr;
	        gemm('N', 'N', Drl, Drr, Dlm, 1.0, ltdata, Dlr, btr.tensorData, Drm, 0.0, localdata, Drr);	
	    }	    
	}
	else
	{
	    #pragma omp parallel for  schedule(dynamic, 1)  
            for (BLASINT lr = 0; lr < Dll*Drl; lr++)
            {
                const BLASINT l = lr/Drl;           
                const BLASINT r = lr%Drl;           
                const DT* ltdata = btl.tensorData+l*Dlm*Dlr;             
                const DT* rtdata = btr.tensorData+r*Drm*Drr;             
                DT* localdata = bc.tensorData+(l*Drl+r)*Dlr*Drr;            
                gemm('T', 'N', Dlr, Drr, Dlm, 1.0, ltdata, Dlr, rtdata, Drr, 0.0, localdata, Drr);
            }
	}
    }
    else // (cttype == 'I')
    {
        if (Dll*Drl > 10)
        {
            #pragma omp parallel for  schedule(dynamic, 1)  
            for (BLASINT lr = 0; lr < Dll*Drl; lr++)
            {
                const BLASINT l = lr/Drl;
                const BLASINT r = lr%Drl;           
                const DT* ltdata = btl.tensorData+l*Dlm*Dlr;
                const DT* rtdata = btr.tensorData+r*Drm*Drr; 
                DT* localdata = bc.tensorData+(l*Drl+r)*Drr*Dlr;            
                gemm('T', 'N', Drr, Dlr, Dlm, 1.0, rtdata, Drr, ltdata, Dlr, 0.0, localdata, Dlr);
            }
        }
        else
        {  
            for (BLASINT lr = 0; lr < Dll*Drl; lr++)
            {
                const BLASINT l = lr/Drl;
                const BLASINT r = lr%Drl;
                const DT* ltdata = btl.tensorData+l*Dlm*Dlr;
                const DT* rtdata = btr.tensorData+r*Drm*Drr;
                DT* localdata = bc.tensorData+(l*Drl+r)*Drr*Dlr;
                gemm('T', 'N', Drr, Dlr, Dlm, 1.0, rtdata, Drr, ltdata, Dlr, 0.0, localdata, Dlr);
            }
        }	    
    }

    bc.setFileSize();

#ifdef CHECK_BASETENSOR
    bc.checkBaseTensor();
#endif   

#ifdef DEBUG_BASETENSOR_TIME_cbibitt    
    std::cout<<"***Time on BaseTensor(cbibitt) is " << double(clock()-time_begin)/CLOCKS_PER_SEC << std::endl<<std::endl;
#endif    	
}
#endif
