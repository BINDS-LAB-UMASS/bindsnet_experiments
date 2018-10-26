// define the time step

#include <stdint.h>
#include "modelSpec.h"
#include "global.h"

//--------------------------------------------------------------------------
/*! \brief This function defines the magicnetwork_model model 
*/
//--------------------------------------------------------------------------

// 
// define the neuron model types as integer variables

// parameter values
// neurons

// synapses

// initial variables (neurons)
 
// initial variables (synapses)
// one additional initial variable for hidden_weightmatrix
 

void modelDefinition(NNmodel &model)
{
  initGeNN();
  GENN_PREFERENCES::autoRefractory = 0;
  // Compiler optimization flags
  GENN_PREFERENCES::userCxxFlagsWIN = "/Ox /w /arch:AVX2 /MP";
  GENN_PREFERENCES::userCxxFlagsGNU = "-w -O3 -ffast-math -fno-finite-math-only -march=native";
  GENN_PREFERENCES::userNvccFlags = "-O3";

  // GENN_PREFERENCES set in brian2genn
  GENN_PREFERENCES::autoChooseDevice= 1;
  GENN_PREFERENCES::defaultDevice= 0;

  model.setDT(0.0001);
  // Define the relevant neuron models
  neuronModel n;


  weightUpdateModel s;
  postSynModel ps;  

  model.setName("magicnetwork_model");
  model.setPrecision(GENN_DOUBLE);
  unsigned int delaySteps;
  model.finalize();
}
