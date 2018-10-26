

#ifndef _magicnetwork_model_synapseKrnl_cc
#define _magicnetwork_model_synapseKrnl_cc
#define BLOCKSZ_SYN 32

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model magicnetwork_model containing the synapse kernel and learning kernel functions.
*/
//-------------------------------------------------------------------------

extern "C" __global__ void calcSynapses(double t)
 {
    unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;
    unsigned int lmax, j, r;
    double addtoinSyn;
    unsigned int ipost;
    unsigned int prePos; 
    unsigned int npost; 
    
    // synapse group synapses
    if (id < 320) {
        
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 9) {
                dd_glbSpkCntneurongroup[0] = 0;
                dd_glbSpkCntpoissongroup[0] = 0;
                d_done = 0;
            }
        }
    }
    
}


#endif
