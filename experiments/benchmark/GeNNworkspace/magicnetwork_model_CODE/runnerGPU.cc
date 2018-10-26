
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model magicnetwork_model containing the host side code for a GPU simulator version.
*/
//-------------------------------------------------------------------------


// software version of atomic add for double precision
__device__ double atomicAddSW(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    }
    while (assumed != old);
    return __longlong_as_double(old);
}

template<typename RNG>
__device__ float exponentialDistFloat(RNG *rng) {
    while (true) {
        const float u = curand_uniform(rng);
        if (u != 0.0f) {
            return -logf(u);
        }
    }
}

template<typename RNG>
__device__ double exponentialDistDouble(RNG *rng) {
    while (true) {
        const double u = curand_uniform_double(rng);
        if (u != 0.0) {
            return -log(u);
        }
    }
}

#include "neuronKrnl.cc"

// ------------------------------------------------------------------------
// copying remote data to device

// ------------------------------------------------------------------------
// copying things to device

// ------------------------------------------------------------------------
// copying things from device

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice(bool hostInitialisedOnly) {
}

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice() {
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice() {
}
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice() {
}
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice() {
}
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice() {
}

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice() {
}

// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice() {
}

// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice() {
}

// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice() {
}

// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice() {
}

// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice() {
}

// ------------------------------------------------------------------------
// the time stepping procedure (using GPU)
void stepTimeGPU() {
    dim3 nThreads(32, 1);
    dim3 nGrid(0, 1);
    
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
}
