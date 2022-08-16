//==========================================================================
// This file has been automatically generated for Pythia 8 by
// MadGraph5_aMC@NLO v. 2.5.1, 2016-11-04
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "PY8MEs.h"

namespace PY8MEs_namespace
{

//==========================================================================

// Constructors
PY8MEs::PY8MEs(string card_path)
{
  model = new Parameters_heft_ckm(); 
  if (card_path != string())
    initModelFromSLHACard(card_path); 
  releaseModelOnExit = true; 
  load_processes(); 
}
PY8MEs::PY8MEs(Parameters_heft_ckm * model_input) : model(model_input)
{
  releaseModelOnExit = false; 
  load_processes(); 
}

// Destructor
PY8MEs::~PY8MEs()
{
  // Release all process instances
  for(unsigned int i = 0; i < loaded_processes.size(); i++ )
  {
    delete loaded_processes[i]; 
    loaded_processes[i] = NULL; 
  }

  // Also release the model instance if it was instantiated here
  if (releaseModelOnExit)
  {
    releaseModel(); 
  }
}

// Function to instantiate a model. The user can do that himself, but
// when using the function below he will not need to know the class name of the
// model.
Parameters_heft_ckm * PY8MEs::instantiateModel(string param_card_path) 
{
  Parameters_heft_ckm * new_model = new Parameters_heft_ckm(); 
  if (param_card_path != string())
  {
    SLHAReader slha(param_card_path); 
    new_model->setIndependentParameters(slha); 
    new_model->setIndependentCouplings(); 
    new_model->printIndependentParameters(); 
    new_model->printIndependentCouplings(); 
  }
  return new_model; 
}

//--------------------------------------------------------------------------
// Preload processes
void PY8MEs::load_processes() 
{
  loaded_processes = vector < PY8ME * > (); 
  #include "all_processes_loading.inc"
}

// Test the availability of a process and returns pointer to it if found
PY8ME * PY8MEs::getProcess(vec_int in_pdgs, vec_int out_pdgs, set_int
    schannels)
{
  // A process was found if the pointer returned is not NULL
  // For now, always create the corresponding process instance (if necessary)
  // when this function is called
  process_accessor proc = getProcess(getProcessSpecifier(in_pdgs, out_pdgs,
      schannels), true);
  if (proc.first)
  {
    // Make sure to initialize the instance properly
    proc.first->setPermutation(proc.second.first); 
    proc.first->setProcID(proc.second.second); 
    return proc.first; 
  }
  else
  {
    return proc.first; 
  }
}

//--------------------------------------------------------------------------
// Obtain the ME for a specific process and color/helicity configuration
// The first element of the pair indicates whether the process was available or
// not.
pair < double, bool > PY8MEs::calculateME(vec_int in_pdgs, vec_int out_pdgs,
    vector < vec_double > momenta, set_int schannels, vec_int colors, vec_int
    helicities)
{

  // Access the process
  process_accessor proc_handle = getProcess(getProcessSpecifier(in_pdgs,
      out_pdgs, schannels));

  // Return right away if unavailable
  if (proc_handle.second.second < 0)
    return make_pair(0.0, false); 

  PY8ME * proc_ptr = proc_handle.first; 
  vec_int perms = proc_handle.second.first; 
  int proc_ID = proc_handle.second.second; 

  proc_ptr->setMomenta(momenta); 
  proc_ptr->setProcID(proc_ID); 
  proc_ptr->setPermutation(perms); 
  proc_ptr->setColors(colors); 
  proc_ptr->setHelicities(helicities); 

  return make_pair(proc_ptr->sigmaKin(), true); 
}

//--------------------------------------------------------------------------
// Build a process specifier from its characteristics
struct process_specifier PY8MEs::getProcessSpecifier(vec_int in_pdgs, vec_int
    out_pdgs, set_int schannels)
{
  struct process_specifier proc_characteristics; 
  proc_characteristics.in_pdgs = in_pdgs; 
  proc_characteristics.out_pdgs = out_pdgs; 
  proc_characteristics.required_s_channels = schannels; 
  return proc_characteristics; 
}

//--------------------------------------------------------------------------
// Access a process specified, either from the map if already considered or
// create one if
// available and create_entry is true.
// If create_entry is not true but the process is available then a NULL pointer
// is provided.
// In any case, if available, a non-negative process ID is returned as well as
// the corresponding
// permutation in the process accessor.
process_accessor PY8MEs::getProcess(struct process_specifier proc, bool
    create_entry)
{

  // Check if process already available in the map
  map < process_specifier, process_accessor > ::iterator it =
      processes_map.find(proc);
  if (it != processes_map.end())
    return it->second; 

  // Check if available
  vec_int in_pdgs = proc.in_pdgs; 
  vec_int out_pdgs = proc.out_pdgs; 
  set_int schannels = proc.required_s_channels; 

  // Loop over loaded processes to try and find the required process
  for (unsigned int i = 0; i < loaded_processes.size(); i++ )
  {
    perm_and_id proc_handle = loaded_processes[i]->getPY8ME(in_pdgs, out_pdgs,
        schannels);
    if (proc_handle.second >= 0)
    {
      process_accessor returned_process_accessor =
          make_pair(loaded_processes[i], proc_handle);
      if (create_entry)
        processes_map[proc] = returned_process_accessor; 
      return returned_process_accessor; 
    }
  }

  // Process not found
  perm_and_id not_found = make_pair(vec_int(), -1); 
  return make_pair((PY8ME * ) NULL, not_found); 
}

//--------------------------------------------------------------------------
// Function to sync processes variables with the currently active model
void PY8MEs::syncProcessesWithModel()
{
  // Loop over loaded processes to broadcast the synchronization
  for (unsigned int i = 0; i < loaded_processes.size(); i++ )
  {
    loaded_processes[i]->syncProcModelParams(); 
  }
}

//--------------------------------------------------------------------------
// Set the external masses of all processes to an integer mode:
// 0 : Mass taken from the model
// 1 : Mass taken from p_i^2 if not massless to begin with
// 2 : Mass always taken from p_i^2.
void PY8MEs::setProcessesExternalMassesMode(int mode)
{
  // Loop over loaded processes to broadcast the chosen mode
  for (unsigned int i = 0; i < loaded_processes.size(); i++ )
  {
    loaded_processes[i]->setExternalMassesMode(mode); 
  }
}

//--------------------------------------------------------------------------
// Broadcast settings to all processes
void PY8MEs::seProcessesIncludeSymmetryFactors(bool OnOff) 
{
  for (unsigned int i = 0; i < loaded_processes.size(); i++ )
  {
    loaded_processes[i]->setIncludeSymmetryFactors(OnOff); 
  }
}
void PY8MEs::seProcessesIncludeHelicityAveragingFactors(bool OnOff) 
{
  for (unsigned int i = 0; i < loaded_processes.size(); i++ )
  {
    loaded_processes[i]->setIncludeHelicityAveragingFactors(OnOff); 
  }
}
void PY8MEs::seProcessesIncludeColorAveragingFactors(bool OnOff) 
{
  for (unsigned int i = 0; i < loaded_processes.size(); i++ )
  {
    loaded_processes[i]->setIncludeColorAveragingFactors(OnOff); 
  }
}

//--------------------------------------------------------------------------
// Function to initialize the model
void PY8MEs::initModelFromSLHACard(string card_path)
{
  SLHAReader slha(card_path); 
  model->setIndependentParameters(slha); 
  model->setIndependentCouplings(); 
  model->printIndependentParameters(); 
  model->printIndependentCouplings(); 
  syncProcessesWithModel(); 
}

//--------------------------------------------------------------------------
// Function to update the alpha_S dependent couplings each event.
void PY8MEs::updateModelDependentCouplings(double alpS)
{
  model->aS = alpS; 
  model->setDependentParameters(); 
  model->setDependentCouplings(); 
}

// 
// void PY8MEs::initModelWithPY8(ParticleData * & pd, Couplings * & csm,
// SusyLesHouches * & slhaPtr)
// {
// model->setIndependentParameters(particleDataPtr, couplingsPtr, slhaPtr);
// model->setIndependentCouplings();
// model->printIndependentParameters();
// model->printIndependentCouplings();
// syncProcessesWithModel();
// }
// void PY8MEs::updateModelDependentCouplingsWithPY8(ParticleData * & pd,
// Couplings * & csm, SusyLesHouches * & slhaPtr, double alpS)
// {
// model->setDependentParameters(particleDataPtr, couplingsPtr, slhaPtr, alpS);
// model->setDependentCouplings();
// }
// 


}  // End namespace PY8MEs_namespace

