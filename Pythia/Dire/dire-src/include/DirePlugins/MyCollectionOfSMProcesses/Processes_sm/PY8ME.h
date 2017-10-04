//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. 2.5.1, 2016-11-04
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef PY8ME_H_sm
#define PY8ME_H_sm

#include <vector> 
#include <set> 

using namespace std; 

namespace PY8MEs_namespace 
{

typedef vector<double> vec_double; 
typedef vector<int> vec_int; 

//==========================================================================
// A mother class representing a matrix element associated to a list of
// mapped processes.
//--------------------------------------------------------------------------
class PY8ME
{
  public:

    // The definition of a virtual destructor is necessary for the destruction
    // to be sent downstream.
    virtual ~PY8ME() {}; 

    // Calculate squared ME.
    virtual double sigmaKin() = 0; 

    // Info on the subprocess.
    virtual string name() const = 0; 
    virtual int code() const = 0; 

    virtual string inFlux() const = 0; 

    // Obtain numerical value of the external masses
    virtual vector<double> getMasses() const = 0; 

    // Access to a specific process
    virtual pair < vector<int> , int > getPY8ME(vector<int> initial_pdgs, 
    vector<int> final_pdgs, set<int> schannels = set<int> ()) = 0; 

    // Set momenta
    virtual void setMomenta(vector < vec_double > momenta_picked) = 0; 

    // Set color configuration to use. An empty vector means sum over all.
    virtual void setColors(vector<int> colors_picked) = 0; 

    // Set the helicity configuration to use. Am empty vector means sum over
    // all.
    virtual void setHelicities(vector<int> helicities_picked) = 0; 

    // Set the permutation to use (will apply to momenta, colors and helicities)
    virtual void setPermutation(vector<int> perm_picked) = 0; 

    // Set the proc_ID to use
    virtual void setProcID(int procID_picked) = 0; 

    // Access to all the helicity and color configurations for a given process
    virtual vector < vec_int > getColorConfigs(int specify_proc_ID = -1,
        vector<int> permutation = vector<int> ()) = 0;
    virtual vector < vec_int > getHelicityConfigs(vector<int> permutation =
        vector<int> ()) = 0;

    // Maps of Helicity <-> hel_ID and ColorConfig <-> colorConfig_ID.
    virtual vector<int> getHelicityConfigForID(int hel_ID, vector<int>
        permutation = vector<int> ()) = 0;
    virtual int getHelicityIDForConfig(vector<int> hel_config, vector<int>
        permutation = vector<int> ()) = 0;
    virtual vector<int> getColorConfigForID(int color_ID, int specify_proc_ID =
        -1, vector<int> permutation = vector<int> ()) = 0;
    virtual int getColorIDForConfig(vector<int> color_config, int
        specify_proc_ID = -1, vector<int> permutation = vector<int> ()) = 0;
    //virtual int getColorFlowRelativeNCPower(int color_flow_ID, int
    //    specify_proc_ID = -1) = 0;

    // Access previously computed results
    virtual vector < vec_double > getAllResults(int specify_proc_ID = -1) = 0; 
    virtual double getResult(int helicity_ID, int color_ID, int specify_proc_ID
        = -1) = 0;

    vector< int > invert_mapping(vector< int > mapping) {
      vector< int > inverted_mapping;
      for (int i = 0; i < int(mapping.size()); i++ ) {
        for (int j = 0; j < int(mapping.size()); j++ ) {
          if (mapping[j]==i) {
            inverted_mapping.push_back(j);
            break;
          }
        }
      }
      return inverted_mapping;
    }

    // Control whether to include the symmetry factors or not
    //virtual void set_include_symmetry_factors(bool OnOff) = 0; 
    //virtual bool get_include_symmetry_factors() = 0; 

}; 

}  // End namespace PY8MEs_namespace

#endif  // PY8ME_H_sm


