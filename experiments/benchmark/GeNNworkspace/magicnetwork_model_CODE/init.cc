

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize() {
    #if defined(__GNUG__) && !defined(__clang__) && defined(__x86_64__) && __GLIBC__ == 2 && (__GLIBC_MINOR__ == 23 || __GLIBC_MINOR__ == 24)
    if(std::getenv("LD_BIND_NOW") == NULL) {
        fprintf(stderr, "Warning: a bug has been found in glibc 2.23 or glibc 2.24 (https://bugs.launchpad.net/ubuntu/+source/glibc/+bug/1663280) "
                        "which results in poor CPU maths performance. We recommend setting the environment variable LD_BIND_NOW=1 to work around this issue.\n");
    }
    #endif
    srand((unsigned int) time(NULL));
    
    // remote neuron spike variables
    // neuron variables
    
    // synapse variables
    
    
    copyStateToDevice(true);
    
}

void initializeAllSparseArrays() {
}

void initmagicnetwork_model() {
    
    
}

