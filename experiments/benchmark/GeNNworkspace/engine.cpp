#ifndef _ENGINE_CC_
#define _ENGINE_CC_

//--------------------------------------------------------------------------
/*! \file engine.cc
\brief Implementation of the engine class.
*/
//--------------------------------------------------------------------------

#include "engine.h"
#include "network.h"

engine::engine()
{
  modelDefinition(model);
  allocateMem();
  initialize();
  Network::_last_run_time= 0.0;
  Network::_last_run_completed_fraction= 0.0;
}

//--------------------------------------------------------------------------
/*! \brief Method for initialising variables
 */
//--------------------------------------------------------------------------

void engine::init(unsigned int which)
{
#ifndef CPU_ONLY
  if (which == CPU) {
  }
  if (which == GPU) {
    copyStateToDevice();
  }
#endif
}

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------

engine::~engine()
{
}


//--------------------------------------------------------------------------
/*! \brief Method for simulating the model for a given period of time
 */
//--------------------------------------------------------------------------

void engine::run(double duration, //!< Duration of time to run the model for 
		  unsigned int which //!< Flag determining whether to run on GPU or CPU only
		  )
{
  std::clock_t start, current; 
  const double t_start = t;
  unsigned int pno;
  unsigned int offset= 0;

  start = std::clock();
  int riT= (int) (duration/DT+1e-2);
  double elapsed_realtime;

  for (int i= 0; i < riT; i++) {
      // report state
      t+= DT;
      // Execute scalar code for run_regularly operations (if any)
#ifndef CPU_ONLY
      if (which == GPU) {
	  stepTimeGPU();
	  t= t-DT;
      }
#endif
      if (which == CPU) {
	  stepTimeCPU();
	  t= t-DT;
      }
      // report state 
      // report spikes
  }  
  Network::_last_run_time = elapsed_realtime;
  if (duration > 0.0)
  {
      Network::_last_run_completed_fraction = (t-t_start)/duration;
  } else {
      Network::_last_run_completed_fraction = 1.0;
  }
}


#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Method for copying all variables of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copyStateFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void engine::getStateFromGPU()
{
  copyStateFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU
 
  This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void engine::getSpikesFromGPU()
{
  copySpikeNFromDevice();
  copySpikesFromDevice();
}


#endif


#endif	

