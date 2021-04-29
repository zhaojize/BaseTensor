#include <stddef.h>

using BLASINT = int;

/*We define two data type here, be sure to choose 
* one of them. 
*/

/*
#ifndef DATATYPE
 #define COMPLEXDATATYPE
 #define DATATYPE  Complex
 #include "Complex.h"
#endif 
*/


#ifndef DATATYPE
#define DOUBLEDATATYPE
#define DATATYPE double
#endif


#define MAXTRK  16   //maximum tensor rank


#define MAXPRD  10 


#if !defined FERMION
#define FERMION   978013 
#endif


#if !defined BOSON 
#define BOSON     101762
#endif


#if !defined STATISTICTYPE      
//#define STATISTICTYPE  FERMION
#define STATISTICTYPE  BOSON
#endif
