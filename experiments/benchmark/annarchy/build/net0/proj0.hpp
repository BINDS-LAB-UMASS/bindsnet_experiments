#pragma once

#include "pop0.hpp"
#include "pop1.hpp"



extern PopStruct0 pop0;
extern PopStruct1 pop1;


/////////////////////////////////////////////////////////////////////////////
// proj0: Input -> Output with target exc
/////////////////////////////////////////////////////////////////////////////
struct ProjStruct0{
    // Number of dendrites
    int size;

    // Transmission and plasticity flags
    bool _transmission, _plasticity, _update;
    int _update_period;
    long int _update_offset;


    // Connectivity
    std::vector<int> post_rank;
    std::vector< std::vector< int > > pre_rank;

    // LIL weights
    std::vector< std::vector< double > > w;


    std::map< int, std::vector< std::pair<int, int> > > inv_pre_rank ;
    std::vector< int > inv_post_rank ;








    // Method called to initialize the projection
    void init_projection() {
        _transmission = true;
        _update = true;
        _plasticity = true;
        _update_period = 1;
        _update_offset = 0L;





        // Inverse the connectivity matrix if spiking neurons
        inverse_connectivity_matrix();






    }

    // Spiking networks: inverse the connectivity matrix
    void inverse_connectivity_matrix() {

        inv_pre_rank =  std::map< int, std::vector< std::pair<int, int> > > ();
        for(int i=0; i<pre_rank.size(); i++){
            for(int j=0; j<pre_rank[i].size(); j++){
                inv_pre_rank[pre_rank[i][j]].push_back(std::pair<int, int>(i,j));
            }
        }
        inv_post_rank =  std::vector< int > (pop1.size, -1);
        for(int i=0; i<post_rank.size(); i++){
            inv_post_rank[post_rank[i]] = i;
        }

    }

    // Computes the weighted sum of inputs or updates the conductances
    void compute_psp() {

#ifdef _OPENMP
        std::vector< double > pop1_exc_thr(pop1.get_size()*omp_get_max_threads(), 0.0);
#endif
        int nb_post;
        double sum;
        
        // Event-based summation
        if (_transmission && pop1._active){
            // Iterate over all incoming spikes
        #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
        #endif
            for(int _idx_j = 0; _idx_j < pop0.spiked.size(); _idx_j++){
                int rk_j = pop0.spiked[_idx_j];
                auto inv_post_ptr = inv_pre_rank.find(rk_j);
                if (inv_post_ptr == inv_pre_rank.end())
                    continue;
                std::vector< std::pair<int, int> >& inv_post = inv_post_ptr->second;
                int nb_post = inv_post.size();
        #ifdef _OPENMP
                int thr = omp_get_thread_num();
        #endif
                // Iterate over connected post neurons
                for(int _idx_i = 0; _idx_i < nb_post; _idx_i++){
                    // Retrieve the correct indices
                    int i = inv_post[_idx_i].first;
                    int j = inv_post[_idx_i].second;
                    
                    
                    // Increase the post-synaptic conductance g_target += w
        #ifndef _OPENMP
                    pop1.g_exc[post_rank[i]] +=  w[i][j];
        #else
                    pop1_exc_thr[thr*pop1.get_size() + post_rank[i]] +=  w[i][j];
        #endif
        
                    
                }
            }
            #ifdef _OPENMP
                if (_transmission && pop1._active){
                    auto pop_size = pop1.get_size();
                for (int i = 0; i < omp_get_max_threads(); i++)
                    for (int j = 0; j < pop_size; j++)
                        pop1.g_exc[j] +=
                            pop1_exc_thr[i*pop_size + j];
                }
        #endif
        } // active
        
    }

    // Draws random numbers
    void update_rng() {

    }

    // Updates synaptic variables
    void update_synapse() {


    }

    // Post-synaptic events
    void post_event() {


    }

    // Accessors for default attributes
    int get_size() { return size; }
    void set_size(int new_size) { size = new_size; }

    // Additional access methods

    // Accessor to connectivity data
    std::vector<int> get_post_rank() { return post_rank; }
    void set_post_rank(std::vector<int> ranks) { post_rank = ranks; }
    std::vector< std::vector<int> > get_pre_rank() { return pre_rank; }
    void set_pre_rank(std::vector< std::vector<int> > ranks) { pre_rank = ranks; }
    int nb_synapses(int n) { return pre_rank[n].size(); }

    // Local parameter w
    std::vector<std::vector< double > > get_w() { return w; }
    std::vector< double > get_dendrite_w(int rk) { return w[rk]; }
    double get_synapse_w(int rk_post, int rk_pre) { return w[rk_post][rk_pre]; }
    void set_w(std::vector<std::vector< double > >value) { w = value; }
    void set_dendrite_w(int rk, std::vector< double > value) { w[rk] = value; }
    void set_synapse_w(int rk_post, int rk_pre, double value) { w[rk_post][rk_pre] = value; }




    // Memory management
    long int size_in_bytes() {
        long int size_in_bytes = 0;
        // Parameters
        size_in_bytes += sizeof(double);	// w
        // Variables
        
        return size_in_bytes;
    }

    void clear() {
        // Variables
        
    }
};
