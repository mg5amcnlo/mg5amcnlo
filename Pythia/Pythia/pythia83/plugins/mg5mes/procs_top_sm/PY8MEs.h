//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. 2.5.1, 2016-11-04
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef PY8MEs_H_sm
#define PY8MEs_H_sm

#include <vector> 
#include <map> 

#include "Parameters_sm.h"
#include "PY8ME.h"
#include "all_processes_headers.inc"

using namespace std; 

namespace PY8MEs_namespace
{

//==========================================================================

// Template to make initializing sets simpler, while not relying on C++11.
// Usage: set(createset<T>(a)(b)(c));

template < typename T > class createset 
{

  private:

    set<T> m_set; 

  public:
    createset() {}
    createset(const T& val) {m_set.insert(val);}
    createset<T> & operator()(const T& val) 
    {
      m_set.insert(val); 
      return * this; 
    }
    operator set<T> () {return m_set;}

}; 

//==========================================================================

// Template to make initializing maps simpler, while not relying on C++11.
// Usage: map(createmap<T,U>(a,b)(c,d)(e,f));

template < typename T, typename U > class createmap 
{

  private:

    map < T, U > m_map; 

  public:

    createmap() {}
    createmap(const T& key, const U& val) {m_map[key] = val;}
    createmap < T, U > & operator()(const T& key, const U& val) 
    {
      m_map[key] = val; 
      return * this; 
    }
    operator map < T, U > () {return m_map;}

}; 

//==========================================================================

// Template to make initializing maps simpler, while not relying on C++11.
// Usage: vector(creatvector<T>(a)(b)(c));

template < typename T > class createvector 
{

  private:

    vector<T> m_vector; 

  public:

    createvector() {}
    createvector(const T& val) {m_vector.push_back(val);}
    createvector<T> & operator()(const T& val) 
    {
      m_vector.push_back(val); 
      return * this; 
    }
    operator vector<T> () {return m_vector;}

}; 

typedef vector<int> vec_int; 
typedef set<int> set_int; 

// Instance of the Matrix element with the corresponding permutation and
// proc_ID to use
typedef pair < vec_int, int > perm_and_id; 
typedef pair < PY8ME * , perm_and_id > process_accessor; 

struct process_specifier 
{
  // List of incoming and list of incoming PDGs
  vec_int in_pdgs; 
  // List of incoming and list of outgoing PDGs
  vec_int out_pdgs; 
  // Set of the required s-channels
  set_int required_s_channels; 

  // Define the ordering operator for to use process_specifier as map keys.
  bool operator < (const process_specifier& other) const 
  {
    return (make_pair(make_pair(in_pdgs, out_pdgs), required_s_channels) < 
    make_pair(make_pair(other.in_pdgs, other.out_pdgs),
        other.required_s_channels)
    ); 
  }

}; 

//==========================================================================
// A class for easily accessing all Matrix Elements exported in this output
//--------------------------------------------------------------------------

class PY8MEs
{
  public:

    // Constructors
    PY8MEs(string param_card_path = string()); 
    PY8MEs(Parameters_sm * model_input); 

    // Destructor
    ~PY8MEs(); 

    // Model factory
    static Parameters_sm * instantiateModel(string param_card_path = string()); 

    // Test the availability of a process and return a pointer to it if found
    // The permutation and process ID is also automatically set.
    PY8ME * getProcess(vec_int in_pdgs, vec_int out_pdgs, set_int schannels =
        set<int> ());

    // A more advanced process accessor which returns the permutation and
    // proc_ID found.
    // It is up to the user to manipulate them accordingly, so this access
    // function
    // is mainly meant to be used internally.
    process_accessor getProcess(struct process_specifier proc, bool
        create_entry = true);

    // Factory for the the process_specified
    struct process_specifier getProcessSpecifier(vec_int in_pdgs, vec_int
        out_pdgs, set_int schannels);

    // Obtain the ME for a specific process and color/helicity configuration
    // The first element of the pair indicates whether the process was
    // available or not.
    pair < double, bool > calculateME(vec_int in_pdgs, vec_int out_pdgs, vector
        < vec_double > momenta, set_int schannels = set_int(), vec_int colors =
        vec_int(), vec_int helicities = vec_int());

    // Initialize the model
    void initModelFromSLHACard(string card_path = string()); 

    // Function to update the alpha_S dependent couplings each event.
    void updateModelDependentCouplings(double alpS); 

    // Function to initialize and update the model from a PY8 particle data
    // pointer.
    // One must link against Parameters_sm_PY8[.h|.cc] and the Pythia8.h include
    // in order to use this routine.
    // 
    // void initModelWithPY8(ParticleData * & pd, Couplings * & csm,
    // SusyLesHouches * & slhaPtr);
    // void updateModelDependentCouplingsWithPY8(ParticleData * & pd, Couplings
    // * & csm, SusyLesHouches * & slhaPtr, double alpS);
    // 

    // Preload processes
    void load_processes(); 

    // A handy function to release the model
    void releaseModel(){if(model) {delete model; model = NULL;}}

    // Accessors.
    // If the model pointer is accessed, it will be the user's responsability
    // to release it and we should no longer do it on his behalf in the
    // destructor as he
    // could be using it elsewhere.
    Parameters_sm * getModel(bool changeReleaseModelOnExit = true) 
    {
      if(changeReleaseModelOnExit)
      {
        releaseModelOnExit = false; 
      }
      return model; 
    }
    // If the model pointer is replace, we must release the old one if
    // necessary and
    // make sure that the new one will not be released by this class since it
    // originated
    // from outside.
    void setModel(Parameters_sm * model_input) 
    {
      if (releaseModelOnExit)
        releaseModel(); 
      releaseModelOnExit = false; 
      model = model_input; 
      syncProcessesWithModel(); 
    }

    // Sync processes variable with the currently active model
    void syncProcessesWithModel(); 

    // Set all values of the external masses od all processes to an integer
    // mode:
    // 0 : Mass taken from the model
    // 1 : Mass taken from p_i^2 if not massless to begin with
    // 2 : Mass always taken from p_i^2.
    void setProcessesExternalMassesMode(int mode); 

    // Broadcast settings to all processes
    void seProcessesIncludeSymmetryFactors(bool OnOff); 
    void seProcessesIncludeHelicityAveragingFactors(bool OnOff); 
    void seProcessesIncludeColorAveragingFactors(bool OnOff); 

  private:

    // Forbid the copy of the accessor as it defines pointers
    PY8MEs(const PY8MEs&); 

    // A map to store all processes accessed and matrix element instances
    // generated
    map < struct process_specifier, process_accessor > processes_map; 

    // A list of instances already available at this time
    vector < PY8ME * > loaded_processes; 

    // The model that the processes will be loaded with
    Parameters_sm * model; 
    // Keep track of whether the model instance was created in this class or
    // externally,
    // so as to decide whether we should release in this destructor or not.
    bool releaseModelOnExit; 
}; 

}  // End namespace PY8MEs_namespace

#endif  // PY8MEs_H_sm

