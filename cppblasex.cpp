#include "cppblasex.h"

void directmultiply(const DATATYPE* A, const  BLASINT &dax, const BLASINT &day, const DATATYPE* B, const BLASINT &dbx, const BLASINT &dby, DATATYPE* C)
{
    for (BLASINT i = 0; i < dax; i++)
    {
        for (BLASINT k = 0; k < dbx; k++)
        {
            BLASINT ik = i*dbx+k;
            for (BLASINT j = 0; j < day; j++)
            {
                 for (BLASINT l = 0; l < dby; l++)
                 {
                     BLASINT jl = j*dby+l;
                     C[ik*day*dby+jl] = A[i*day+j]*B[k*dby+l]; 
                 }    
            }
        }
    }
}
