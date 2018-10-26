/*
 *  ANNarchy-version: 4.6.6
 */
#pragma once
#include "ANNarchy.h"
#include <random>


extern double dt;
extern long int t;
extern std::mt19937 rng;


///////////////////////////////////////////////////////////////
// Main Structure for the population of id 1 (Output)
///////////////////////////////////////////////////////////////
struct PopStruct1{
    int size; // Number of neurons
    bool _active; // Allows to shut down the whole population
    // Access functions used by cython wrapper
    int get_size() { return size; }
    void set_size(int s) { size  = s; }
    bool is_active() { return _active; }
    void set_active(bool val) { _active = val; }

    // Structures for managing spikes
    std::vector<long int> last_spike;
    std::vector<int> spiked;

    // Neuron specific parameters and variables

    // Local parameter tau_m
    std::vector< double > tau_m;

    // Local parameter tau_e
    std::vector< double > tau_e;

    // Local parameter vt
    std::vector< double > vt;

    // Local parameter vr
    std::vector< double > vr;

    // Local parameter El
    std::vector< double > El;

    // Local parameter Ee
    std::vector< double > Ee;

    // Local variable v
    std::vector< double > v;

    // Local variable g_exc
    std::vector< double > g_exc;

    // Local variable r
    std::vector< double > r;

    // Global operations

    // Random numbers



    // Mean Firing rate
    std::vector< std::queue<long int> > _spike_history;
    long int _mean_fr_window;
    double _mean_fr_rate;
    void compute_firing_rate(double window){
        if(window>0.0){
            _mean_fr_window = int(window/dt);
            _mean_fr_rate = 1000./window;
        }
    };


    // Access methods to the parameters and variables

    // Local parameter tau_m
    std::vector< double > get_tau_m() { return tau_m; }
    double get_single_tau_m(int rk) { return tau_m[rk]; }
    void set_tau_m(std::vector< double > val) { tau_m = val; }
    void set_single_tau_m(int rk, double val) { tau_m[rk] = val; }

    // Local parameter tau_e
    std::vector< double > get_tau_e() { return tau_e; }
    double get_single_tau_e(int rk) { return tau_e[rk]; }
    void set_tau_e(std::vector< double > val) { tau_e = val; }
    void set_single_tau_e(int rk, double val) { tau_e[rk] = val; }

    // Local parameter vt
    std::vector< double > get_vt() { return vt; }
    double get_single_vt(int rk) { return vt[rk]; }
    void set_vt(std::vector< double > val) { vt = val; }
    void set_single_vt(int rk, double val) { vt[rk] = val; }

    // Local parameter vr
    std::vector< double > get_vr() { return vr; }
    double get_single_vr(int rk) { return vr[rk]; }
    void set_vr(std::vector< double > val) { vr = val; }
    void set_single_vr(int rk, double val) { vr[rk] = val; }

    // Local parameter El
    std::vector< double > get_El() { return El; }
    double get_single_El(int rk) { return El[rk]; }
    void set_El(std::vector< double > val) { El = val; }
    void set_single_El(int rk, double val) { El[rk] = val; }

    // Local parameter Ee
    std::vector< double > get_Ee() { return Ee; }
    double get_single_Ee(int rk) { return Ee[rk]; }
    void set_Ee(std::vector< double > val) { Ee = val; }
    void set_single_Ee(int rk, double val) { Ee[rk] = val; }

    // Local variable v
    std::vector< double > get_v() { return v; }
    double get_single_v(int rk) { return v[rk]; }
    void set_v(std::vector< double > val) { v = val; }
    void set_single_v(int rk, double val) { v[rk] = val; }

    // Local variable g_exc
    std::vector< double > get_g_exc() { return g_exc; }
    double get_single_g_exc(int rk) { return g_exc[rk]; }
    void set_g_exc(std::vector< double > val) { g_exc = val; }
    void set_single_g_exc(int rk, double val) { g_exc[rk] = val; }

    // Local variable r
    std::vector< double > get_r() { return r; }
    double get_single_r(int rk) { return r[rk]; }
    void set_r(std::vector< double > val) { r = val; }
    void set_single_r(int rk, double val) { r[rk] = val; }



    // Method called to initialize the data structures
    void init_population() {
        _active = true;

        // Local parameter tau_m
        tau_m = std::vector<double>(size, 0.0);

        // Local parameter tau_e
        tau_e = std::vector<double>(size, 0.0);

        // Local parameter vt
        vt = std::vector<double>(size, 0.0);

        // Local parameter vr
        vr = std::vector<double>(size, 0.0);

        // Local parameter El
        El = std::vector<double>(size, 0.0);

        // Local parameter Ee
        Ee = std::vector<double>(size, 0.0);

        // Local variable v
        v = std::vector<double>(size, 0.0);

        // Local variable g_exc
        g_exc = std::vector<double>(size, 0.0);

        // Local variable r
        r = std::vector<double>(size, 0.0);


        // Spiking variables
        spiked = std::vector<int>(0, 0);
        last_spike = std::vector<long int>(size, -10000L);



        // Mean Firing Rate
        _spike_history = std::vector< std::queue<long int> >(size, std::queue<long int>());
        _mean_fr_window = 0;
        _mean_fr_rate = 1.0;


    }

    // Method called to reset the population
    void reset() {

        spiked.clear();
        last_spike.clear();
        last_spike = std::vector<long int>(size, -10000L);



    }

    // Method to draw new random numbers
    void update_rng() {

    }

    // Method to update global operations on the population (min/max/mean...)
    void update_global_ops() {

    }

    // Method to enqueue output variables in case outgoing projections have non-zero delay
    void update_delay() {

    }

    // Main method to update neural variables
    void update() {

        if( _active ) {
            spiked.clear();

            // Updating local variables
            
            for(int i = 0; i < size; i++){
                
                // tau_m * dv/dt = El - v + g_exc *  (Ee - vr)
                double _v = (Ee[i]*g_exc[i] + El[i] - g_exc[i]*vr[i] - v[i])/tau_m[i];
                
                // tau_e * dg_exc/dt = - g_exc
                double _g_exc = -g_exc[i]/tau_e[i];
                
                // tau_m * dv/dt = El - v + g_exc *  (Ee - vr)
                v[i] += dt*_v ;
                
                
                // tau_e * dg_exc/dt = - g_exc
                g_exc[i] += dt*_g_exc ;
                
                
                // Spike emission
                if(v[i] > vt[i]){ // Condition is met
                    // Reset variables

                    v[i] = vr[i];

                    // Store the spike
                    
                    {
                    spiked.push_back(i);
                    }
                    last_spike[i] = t;

                    // Refractory period
                    
                    
                    // Update the mean firing rate
                    if(_mean_fr_window> 0)
                        _spike_history[i].push(t);
            
                }
                
                // Update the mean firing rate
                if(_mean_fr_window> 0){
                    while((_spike_history[i].size() != 0)&&(_spike_history[i].front() <= t - _mean_fr_window)){
                        _spike_history[i].pop(); // Suppress spikes outside the window
                    }
                    r[i] = _mean_fr_rate * float(_spike_history[i].size());
                }
            

            }
        } // active

    }

    

    // Memory management: track the memory consumption
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(double);	// tau_m
        size_in_bytes += sizeof(double);	// tau_e
        size_in_bytes += sizeof(double);	// vt
        size_in_bytes += sizeof(double);	// vr
        size_in_bytes += sizeof(double);	// El
        size_in_bytes += sizeof(double);	// Ee
        // Variables
        size_in_bytes += sizeof(double) * v.capacity();	// v
        size_in_bytes += sizeof(double) * g_exc.capacity();	// g_exc
        size_in_bytes += sizeof(double) * r.capacity();	// r
        
        return size_in_bytes;
    }

    // Memory management: track the memory consumption
    void clear() {
        // Variables
        v.clear();
        v.shrink_to_fit();
        g_exc.clear();
        g_exc.shrink_to_fit();
        r.clear();
        r.shrink_to_fit();
        
    }
};
