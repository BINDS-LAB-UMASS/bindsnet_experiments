//--------------------------------------------------------------------------
/*! \file main.h

\brief Header file containing global variables and macros used in running the model.
*/
//--------------------------------------------------------------------------

using namespace std;
#include <cassert>
#include "hr_time.h"

#include "utils.h" // for CHECK_CUDA_ERRORS
#include "stringUtils.h"

#ifndef CPU_ONLY
#include <cuda_runtime.h>
#else
#define __host__
#define __device__
#endif


#ifndef RAND
#define RAND(Y,X) Y = Y * 1103515245 +12345;X= (unsigned int)(Y >> 16) & 32767
#endif

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 10000

// and some global variables
CStopWatch timer;

//----------------------------------------------------------------------
// other stuff:


