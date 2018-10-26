

#ifndef _magicnetwork_model_neuronKrnl_cc
#define _magicnetwork_model_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model magicnetwork_model containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(double t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    
    if (id == 0) {
    }
    __threadfence();
    
    __syncthreads();
    
}
#endif
