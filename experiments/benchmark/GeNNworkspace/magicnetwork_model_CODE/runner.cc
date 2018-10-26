

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model magicnetwork_model containing general control code.
*/
//-------------------------------------------------------------------------

#define RUNNER_CC_COMPILE

#include "definitions.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cassert>
#include <stdint.h>
#if defined(__GNUG__) && !defined(__clang__) && defined(__x86_64__)
    #include <gnu/libc-version.h>
#endif

// ------------------------------------------------------------------------
// global variables

unsigned long long iT;
double t;


// ------------------------------------------------------------------------
// remote neuron groups


// ------------------------------------------------------------------------
// neuron variables

__device__ volatile unsigned int d_done;

// ------------------------------------------------------------------------
// postsynaptic variables

// ------------------------------------------------------------------------
// synapse variables


//-------------------------------------------------------------------------
/*! \brief Function to convert a firing probability (per time step) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given probability.
*/
//-------------------------------------------------------------------------

void convertProbabilityToRandomNumberThreshold(double *p_pattern, uint64_t *pattern, int N) {
    double fac= pow(2.0, (double) sizeof(uint64_t)*8-16);
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (p_pattern[i]*fac);
    }
}

//-------------------------------------------------------------------------
/*! \brief Function to convert a firing rate (in kHz) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given rate.
*/
//-------------------------------------------------------------------------

void convertRateToRandomNumberThreshold(double *rateKHz_pattern, uint64_t *pattern, int N) {
    double fac= pow(2.0, (double) sizeof(uint64_t)*8-16)*DT;
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (rateKHz_pattern[i]*fac);
    }
}

#include "runnerGPU.cc"
#include "init.cc"
#include "neuronFnct.cc"

void allocateMem() {
    CHECK_CUDA_ERRORS(cudaSetDevice(0));
    // ------------------------------------------------------------------------
    // remote neuron groups
    
    
    // ------------------------------------------------------------------------
    // local neuron groups
}

void freeMem() {
}

void exitGeNN() {
    freeMem();
    cudaDeviceReset();
}

// ------------------------------------------------------------------------
// the actual time stepping procedure (using CPU)
void stepTimeCPU() {
    calcNeuronsCPU(t);
    iT++;
    t= iT*DT;
}
