//--------------------------------------------------------------------------
/*! \file main.cu

\brief Main entry point for the running a model simulation. 
*/
//--------------------------------------------------------------------------

#include "main.h"
#include "magicnetwork_model.cpp"
#include "magicnetwork_model_CODE/definitions.h"

#include "network.h"
#include "objects.h"
#include "b2glib/convert_synapses.h"
#include "b2glib/copy_static_arrays.h"

#include "engine.cpp"


//--------------------------------------------------------------------------
/*! \brief This function is the entry point for running the simulation of the MBody1 model network.
*/
//--------------------------------------------------------------------------
int which;

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    fprintf(stderr, "usage: main <basename> <time (s)> <CPU=0, GPU=1> \n");
    return 1;
  }
  double totalTime= atof(argv[2]);
  which= atoi(argv[3]);
  string OutDir = toString(argv[1]) +"_output";
  string cmd= toString("mkdir ") +OutDir;
  system(cmd.c_str());
  string name;
  name= OutDir+ "/"+ toString(argv[1]) + toString(".time");
  FILE *timef= fopen(name.c_str(),"a");  

  timer.startTimer();
  fprintf(stderr, "# DT %f \n", DT);
  fprintf(stderr, "# totalTime %f \n", totalTime);
  
  //-----------------------------------------------------------------
  // build the neuronal circuitery
  engine eng;

  //-----------------------------------------------------------------
  // load variables and parameters and translate them from Brian to Genn
  _init_arrays();
  _load_arrays();
  rk_randomseed(brian::_mersenne_twister_states[0]);
  {
	  using namespace brian;
   	  
   _array_defaultclock_dt[0] = 0.0001;
   _array_defaultclock_dt[0] = 0.0001;
   _array_defaultclock_dt[0] = 0.0001;

  }

  // translate to GeNN synaptic arrays
   initmagicnetwork_model();

  // copy variable arrays

  // copy scalar variables
  
  // initialise random seeds (if any are used)

  //-----------------------------------------------------------------
  
  eng.init(which);         // this includes copying g's for the GPU version
#ifndef CPU_ONLY
  copyStateToDevice();
#endif

  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation
  fprintf(stderr, "# We are running with fixed time step %f \n", DT);

  t= -DT;
  void *devPtr;
  eng.run(totalTime, which); // run for the full duration
  timer.stopTimer();
  cerr << t << " done ..." << endl;
  fprintf(timef,"%f \n", timer.getElapsedTime());

  // get the final results from the GPU 
#ifndef CPU_ONLY
  if (which == GPU) {
    eng.getStateFromGPU();
    eng.getSpikesFromGPU();
  }
#endif
  // translate GeNN arrays back to synaptic arrays
 
  // copy variable arrays

  // copy scalar variables

  _write_arrays();
  _dealloc_arrays();
  cerr << "everything finished." << endl;
  return 0;
}

