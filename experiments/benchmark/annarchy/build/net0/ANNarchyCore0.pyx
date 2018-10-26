# cython: embedsignature=True
from cpython.exc cimport PyErr_CheckSignals
from libcpp.vector cimport vector
from libcpp.map cimport map, pair
from libcpp cimport bool
import numpy as np
cimport numpy as np

import ANNarchy
from ANNarchy.core.cython_ext.Connector cimport LILConnectivity as LIL
from ANNarchy.core.cython_ext.Connector cimport CSRConnectivity, CSRConnectivityPre1st

cdef extern from "ANNarchy.h":

    # User-defined functions


    # User-defined constants


    # Data structures

    # Export Population 0 (Input)
    cdef struct PopStruct0 :
        int get_size()
        void set_size(int)
        bool is_active()
        void set_active(bool)
        void reset()


        # Local parameter rates
        vector[double] get_rates()
        double get_single_rates(int rk)
        void set_rates(vector[double])
        void set_single_rates(int, double)

        # Local variable p
        vector[double] get_p()
        double get_single_p(int rk)
        void set_p(vector[double])
        void set_single_p(int, double)

        # Local variable r
        vector[double] get_r()
        double get_single_r(int rk)
        void set_r(vector[double])
        void set_single_r(int, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()

    # Export Population 1 (Output)
    cdef struct PopStruct1 :
        int get_size()
        void set_size(int)
        bool is_active()
        void set_active(bool)
        void reset()


        # Local parameter tau_m
        vector[double] get_tau_m()
        double get_single_tau_m(int rk)
        void set_tau_m(vector[double])
        void set_single_tau_m(int, double)

        # Local parameter tau_e
        vector[double] get_tau_e()
        double get_single_tau_e(int rk)
        void set_tau_e(vector[double])
        void set_single_tau_e(int, double)

        # Local parameter vt
        vector[double] get_vt()
        double get_single_vt(int rk)
        void set_vt(vector[double])
        void set_single_vt(int, double)

        # Local parameter vr
        vector[double] get_vr()
        double get_single_vr(int rk)
        void set_vr(vector[double])
        void set_single_vr(int, double)

        # Local parameter El
        vector[double] get_El()
        double get_single_El(int rk)
        void set_El(vector[double])
        void set_single_El(int, double)

        # Local parameter Ee
        vector[double] get_Ee()
        double get_single_Ee(int rk)
        void set_Ee(vector[double])
        void set_single_Ee(int, double)

        # Local variable v
        vector[double] get_v()
        double get_single_v(int rk)
        void set_v(vector[double])
        void set_single_v(int, double)

        # Local variable g_exc
        vector[double] get_g_exc()
        double get_single_g_exc(int rk)
        void set_g_exc(vector[double])
        void set_single_g_exc(int, double)

        # Local variable r
        vector[double] get_r()
        double get_single_r(int rk)
        void set_r(vector[double])
        void set_single_r(int, double)




        # Compute firing rate
        void compute_firing_rate(double window)


        # memory management
        long int size_in_bytes()
        void clear()


    # Export Projection 0
    cdef struct ProjStruct0 :
        # Flags
        bool _transmission
        bool _plasticity
        bool _update
        int _update_period
        long _update_offset
        # Size
        int get_size()
        int nb_synapses(int)
        void set_size(int)


        # LIL Connectivity
        vector[int] get_post_rank()
        vector[vector[int]] get_pre_rank()
        void set_post_rank(vector[int])
        void set_pre_rank(vector[vector[int]])
        void inverse_connectivity_matrix()

        # Local variable w
        vector[vector[double]] get_w()
        vector[double] get_dendrite_w(int)
        double get_synapse_w(int, int)
        void set_w(vector[vector[double]])
        void set_dendrite_w(int, vector[double])
        void set_synapse_w(int, int, double)








        # memory management
        long int size_in_bytes()
        void clear()


    # Monitors
    cdef cppclass Monitor:
        vector[int] ranks
        int period_
        int period_offset_
        long offset_

    void addRecorder(Monitor*)
    void removeRecorder(Monitor*)

    # Population 0 (Input) : Monitor
    cdef cppclass PopRecorder0 (Monitor):
        PopRecorder0(vector[int], int, int, long) except +
        long int size_in_bytes()

        vector[vector[double]] rates
        bool record_rates

        vector[vector[double]] p
        bool record_p

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Population 1 (Output) : Monitor
    cdef cppclass PopRecorder1 (Monitor):
        PopRecorder1(vector[int], int, int, long) except +
        long int size_in_bytes()

        vector[vector[double]] tau_m
        bool record_tau_m

        vector[vector[double]] tau_e
        bool record_tau_e

        vector[vector[double]] vt
        bool record_vt

        vector[vector[double]] vr
        bool record_vr

        vector[vector[double]] El
        bool record_El

        vector[vector[double]] Ee
        bool record_Ee

        vector[vector[double]] v
        bool record_v

        vector[vector[double]] g_exc
        bool record_g_exc

        vector[vector[double]] r
        bool record_r

        map[int, vector[long]] spike
        bool record_spike
        void clear_spike()

    # Projection 0 : Monitor
    cdef cppclass ProjRecorder0 (Monitor):
        ProjRecorder0(vector[int], int, int, long) except +

        vector[vector[vector[double]]] w
        bool record_w


    # Instances

    PopStruct0 pop0
    PopStruct1 pop1

    ProjStruct0 proj0

    # Methods
    void initialize(double, long)
    void setSeed(long)
    void run(int nbSteps) nogil
    int run_until(int steps, vector[int] populations, bool or_and)
    void step()

    # Time
    long getTime()
    void setTime(long)

    # dt
    double getDt()
    void setDt(double dt_)


    # Number of threads
    void setNumberThreads(int)


# Population wrappers

# Wrapper for population 0 (Input)
cdef class pop0_wrapper :

    def __cinit__(self, size):

        pop0.set_size(size)

    property size:
        def __get__(self):
            return pop0.get_size()
    def reset(self):
        pop0.reset()
    def activate(self, bool val):
        pop0.set_active(val)


    # Local parameter rates
    cpdef np.ndarray get_rates(self):
        return np.array(pop0.get_rates())
    cpdef set_rates(self, np.ndarray value):
        pop0.set_rates( value )
    cpdef double get_single_rates(self, int rank):
        return pop0.get_single_rates(rank)
    cpdef set_single_rates(self, int rank, value):
        pop0.set_single_rates(rank, value)

    # Local variable p
    cpdef np.ndarray get_p(self):
        return np.array(pop0.get_p())
    cpdef set_p(self, np.ndarray value):
        pop0.set_p( value )
    cpdef double get_single_p(self, int rank):
        return pop0.get_single_p(rank)
    cpdef set_single_p(self, int rank, value):
        pop0.set_single_p(rank, value)

    # Local variable r
    cpdef np.ndarray get_r(self):
        return np.array(pop0.get_r())
    cpdef set_r(self, np.ndarray value):
        pop0.set_r( value )
    cpdef double get_single_r(self, int rank):
        return pop0.get_single_r(rank)
    cpdef set_single_r(self, int rank, value):
        pop0.set_single_r(rank, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop0.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop0.size_in_bytes()

    def clear(self):
        return pop0.clear()

# Wrapper for population 1 (Output)
cdef class pop1_wrapper :

    def __cinit__(self, size):

        pop1.set_size(size)

    property size:
        def __get__(self):
            return pop1.get_size()
    def reset(self):
        pop1.reset()
    def activate(self, bool val):
        pop1.set_active(val)


    # Local parameter tau_m
    cpdef np.ndarray get_tau_m(self):
        return np.array(pop1.get_tau_m())
    cpdef set_tau_m(self, np.ndarray value):
        pop1.set_tau_m( value )
    cpdef double get_single_tau_m(self, int rank):
        return pop1.get_single_tau_m(rank)
    cpdef set_single_tau_m(self, int rank, value):
        pop1.set_single_tau_m(rank, value)

    # Local parameter tau_e
    cpdef np.ndarray get_tau_e(self):
        return np.array(pop1.get_tau_e())
    cpdef set_tau_e(self, np.ndarray value):
        pop1.set_tau_e( value )
    cpdef double get_single_tau_e(self, int rank):
        return pop1.get_single_tau_e(rank)
    cpdef set_single_tau_e(self, int rank, value):
        pop1.set_single_tau_e(rank, value)

    # Local parameter vt
    cpdef np.ndarray get_vt(self):
        return np.array(pop1.get_vt())
    cpdef set_vt(self, np.ndarray value):
        pop1.set_vt( value )
    cpdef double get_single_vt(self, int rank):
        return pop1.get_single_vt(rank)
    cpdef set_single_vt(self, int rank, value):
        pop1.set_single_vt(rank, value)

    # Local parameter vr
    cpdef np.ndarray get_vr(self):
        return np.array(pop1.get_vr())
    cpdef set_vr(self, np.ndarray value):
        pop1.set_vr( value )
    cpdef double get_single_vr(self, int rank):
        return pop1.get_single_vr(rank)
    cpdef set_single_vr(self, int rank, value):
        pop1.set_single_vr(rank, value)

    # Local parameter El
    cpdef np.ndarray get_El(self):
        return np.array(pop1.get_El())
    cpdef set_El(self, np.ndarray value):
        pop1.set_El( value )
    cpdef double get_single_El(self, int rank):
        return pop1.get_single_El(rank)
    cpdef set_single_El(self, int rank, value):
        pop1.set_single_El(rank, value)

    # Local parameter Ee
    cpdef np.ndarray get_Ee(self):
        return np.array(pop1.get_Ee())
    cpdef set_Ee(self, np.ndarray value):
        pop1.set_Ee( value )
    cpdef double get_single_Ee(self, int rank):
        return pop1.get_single_Ee(rank)
    cpdef set_single_Ee(self, int rank, value):
        pop1.set_single_Ee(rank, value)

    # Local variable v
    cpdef np.ndarray get_v(self):
        return np.array(pop1.get_v())
    cpdef set_v(self, np.ndarray value):
        pop1.set_v( value )
    cpdef double get_single_v(self, int rank):
        return pop1.get_single_v(rank)
    cpdef set_single_v(self, int rank, value):
        pop1.set_single_v(rank, value)

    # Local variable g_exc
    cpdef np.ndarray get_g_exc(self):
        return np.array(pop1.get_g_exc())
    cpdef set_g_exc(self, np.ndarray value):
        pop1.set_g_exc( value )
    cpdef double get_single_g_exc(self, int rank):
        return pop1.get_single_g_exc(rank)
    cpdef set_single_g_exc(self, int rank, value):
        pop1.set_single_g_exc(rank, value)

    # Local variable r
    cpdef np.ndarray get_r(self):
        return np.array(pop1.get_r())
    cpdef set_r(self, np.ndarray value):
        pop1.set_r( value )
    cpdef double get_single_r(self, int rank):
        return pop1.get_single_r(rank)
    cpdef set_single_r(self, int rank, value):
        pop1.set_single_r(rank, value)





    # Compute firing rate
    cpdef compute_firing_rate(self, double window):
        pop1.compute_firing_rate(window)


    # memory management
    def size_in_bytes(self):
        return pop1.size_in_bytes()

    def clear(self):
        return pop1.clear()


# Projection wrappers

# Wrapper for projection 0
cdef class proj0_wrapper :

    def __cinit__(self, synapses):

        cdef LIL syn = synapses
        cdef int size = syn.size
        cdef int nb_post = syn.post_rank.size()
        proj0.set_size( size )
        proj0.set_post_rank( syn.post_rank )
        proj0.set_pre_rank( syn.pre_rank )

        proj0.set_w(syn.w)




    property size:
        def __get__(self):
            return proj0.get_size()

    def nb_synapses(self, int n):
        return proj0.nb_synapses(n)

    # Transmission flag
    def _get_transmission(self):
        return proj0._transmission
    def _set_transmission(self, bool l):
        proj0._transmission = l

    # Update flag
    def _get_update(self):
        return proj0._update
    def _set_update(self, bool l):
        proj0._update = l

    # Plasticity flag
    def _get_plasticity(self):
        return proj0._plasticity
    def _set_plasticity(self, bool l):
        proj0._plasticity = l

    # Update period
    def _get_update_period(self):
        return proj0._update_period
    def _set_update_period(self, int l):
        proj0._update_period = l

    # Update offset
    def _get_update_offset(self):
        return proj0._update_offset
    def _set_update_offset(self, long l):
        proj0._update_offset = l


    # Connectivity
    def post_rank(self):
        return proj0.get_post_rank()
    def set_post_rank(self, val):
        proj0.set_post_rank(val)
        proj0.inverse_connectivity_matrix()
    def pre_rank(self, int n):
        return proj0.get_pre_rank()[n]
    def pre_rank_all(self):
        return proj0.get_pre_rank()
    def set_pre_rank(self, val):
        proj0.set_pre_rank(val)
        proj0.inverse_connectivity_matrix()

    # Local variable w
    def get_w(self):
        return proj0.get_w()
    def set_w(self, value):
        proj0.set_w( value )
    def get_dendrite_w(self, int rank):
        return proj0.get_dendrite_w(rank)
    def set_dendrite_w(self, int rank, vector[double] value):
        proj0.set_dendrite_w(rank, value)
    def get_synapse_w(self, int rank_post, int rank_pre):
        return proj0.get_synapse_w(rank_post, rank_pre)
    def set_synapse_w(self, int rank_post, int rank_pre, double value):
        proj0.set_synapse_w(rank_post, rank_pre, value)







    # memory management
    def size_in_bytes(self):
        return proj0.size_in_bytes()

    def clear(self):
        return proj0.clear()


# Monitor wrappers
cdef class Monitor_wrapper:
    cdef Monitor *thisptr
    def __cinit__(self, list ranks, int period, int period_offset, long offset):
        pass
    property ranks:
        def __get__(self): return self.thisptr.ranks
        def __set__(self, val): self.thisptr.ranks = val
    property period:
        def __get__(self): return self.thisptr.period_
        def __set__(self, val): self.thisptr.period_ = val
    property offset:
        def __get__(self): return self.thisptr.offset_
        def __set__(self, val): self.thisptr.offset_ = val
    property period_offset:
        def __get__(self): return self.thisptr.period_offset_
        def __set__(self, val): self.thisptr.period_offset_ = val

def add_recorder(Monitor_wrapper recorder):
    addRecorder(recorder.thisptr)
def remove_recorder(Monitor_wrapper recorder):
    removeRecorder(recorder.thisptr)


# Population Monitor wrapper
cdef class PopRecorder0_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, period_offset, long offset):
        self.thisptr = new PopRecorder0(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (<PopRecorder0 *>self.thisptr).size_in_bytes()

    property rates:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).rates
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).rates = val
    property record_rates:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).record_rates
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).record_rates = val
    def clear_rates(self):
        (<PopRecorder0 *>self.thisptr).rates.clear()

    property p:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).p
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).p = val
    property record_p:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).record_p
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).record_p = val
    def clear_p(self):
        (<PopRecorder0 *>self.thisptr).p.clear()

    property r:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).r
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).r = val
    property record_r:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).record_r
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).record_r = val
    def clear_r(self):
        (<PopRecorder0 *>self.thisptr).r.clear()

    property spike:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).spike
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).spike = val
    property record_spike:
        def __get__(self): return (<PopRecorder0 *>self.thisptr).record_spike
        def __set__(self, val): (<PopRecorder0 *>self.thisptr).record_spike = val
    def clear_spike(self):
        (<PopRecorder0 *>self.thisptr).clear_spike()

# Population Monitor wrapper
cdef class PopRecorder1_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, period_offset, long offset):
        self.thisptr = new PopRecorder1(ranks, period, period_offset, offset)

    def size_in_bytes(self):
        return (<PopRecorder1 *>self.thisptr).size_in_bytes()

    property tau_m:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).tau_m
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).tau_m = val
    property record_tau_m:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_tau_m
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_tau_m = val
    def clear_tau_m(self):
        (<PopRecorder1 *>self.thisptr).tau_m.clear()

    property tau_e:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).tau_e
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).tau_e = val
    property record_tau_e:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_tau_e
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_tau_e = val
    def clear_tau_e(self):
        (<PopRecorder1 *>self.thisptr).tau_e.clear()

    property vt:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).vt
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).vt = val
    property record_vt:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_vt
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_vt = val
    def clear_vt(self):
        (<PopRecorder1 *>self.thisptr).vt.clear()

    property vr:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).vr
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).vr = val
    property record_vr:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_vr
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_vr = val
    def clear_vr(self):
        (<PopRecorder1 *>self.thisptr).vr.clear()

    property El:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).El
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).El = val
    property record_El:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_El
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_El = val
    def clear_El(self):
        (<PopRecorder1 *>self.thisptr).El.clear()

    property Ee:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).Ee
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).Ee = val
    property record_Ee:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_Ee
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_Ee = val
    def clear_Ee(self):
        (<PopRecorder1 *>self.thisptr).Ee.clear()

    property v:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).v
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).v = val
    property record_v:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_v
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_v = val
    def clear_v(self):
        (<PopRecorder1 *>self.thisptr).v.clear()

    property g_exc:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).g_exc
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).g_exc = val
    property record_g_exc:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_g_exc
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_g_exc = val
    def clear_g_exc(self):
        (<PopRecorder1 *>self.thisptr).g_exc.clear()

    property r:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).r
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).r = val
    property record_r:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_r
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_r = val
    def clear_r(self):
        (<PopRecorder1 *>self.thisptr).r.clear()

    property spike:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).spike
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).spike = val
    property record_spike:
        def __get__(self): return (<PopRecorder1 *>self.thisptr).record_spike
        def __set__(self, val): (<PopRecorder1 *>self.thisptr).record_spike = val
    def clear_spike(self):
        (<PopRecorder1 *>self.thisptr).clear_spike()

# Projection Monitor wrapper
cdef class ProjRecorder0_wrapper(Monitor_wrapper):
    def __cinit__(self, list ranks, int period, int period_offset, long offset):
        self.thisptr = new ProjRecorder0(ranks, period, period_offset, offset)

    property w:
        def __get__(self): return (<ProjRecorder0 *>self.thisptr).w
        def __set__(self, val): (<ProjRecorder0 *>self.thisptr).w = val
    property record_w:
        def __get__(self): return (<ProjRecorder0 *>self.thisptr).record_w
        def __set__(self, val): (<ProjRecorder0 *>self.thisptr).record_w = val
    def clear_w(self):
        (<ProjRecorder0 *>self.thisptr).w.clear()


# User-defined functions


# User-defined constants


# Initialize the network
def pyx_create(double dt, long seed):
    initialize(dt, seed)

# Simulation for the given number of steps
def pyx_run(int nb_steps):
    cdef int nb, rest
    cdef int batch = 1000
    if nb_steps < batch:
        with nogil:
            run(nb_steps)
    else:
        nb = int(nb_steps/batch)
        rest = nb_steps % batch
        for i in range(nb):
            with nogil:
                run(batch)
            PyErr_CheckSignals()
        if rest > 0:
            run(rest)

# Simulation for the given number of steps except if a criterion is reached
def pyx_run_until(int nb_steps, list populations, bool mode):
    cdef int nb
    nb = run_until(nb_steps, populations, mode)
    return nb

# Simulate for one step
def pyx_step():
    step()

# Access time
def set_time(t):
    setTime(t)
def get_time():
    return getTime()

# Access dt
def set_dt(double dt):
    setDt(dt)
def get_dt():
    return getDt()


# Set number of threads
def set_number_threads(int n):
    setNumberThreads(n)


# Set seed
def set_seed(long seed):
    setSeed(seed)
