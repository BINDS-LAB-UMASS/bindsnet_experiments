// scalar can be any scalar type such as float, double

#include <stdint.h>

#ifndef COPY_STATIC_ARRAYS
#define COPY_STATIC_ARRAYS

template<class scalar>
void copy_brian_to_genn(scalar *bv, scalar *gv, int N)
{
    for (int i= 0; i < N; i++) {
	gv[i]= bv[i];
    }
}


template<class scalar>
void copy_genn_to_brian(scalar *gv, scalar *bv, int N)
{
    for (int i= 0; i < N; i++) {
	bv[i]= gv[i];
    }
}

#endif
