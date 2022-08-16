//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef PY8MEs_R130_P0_sm_gg_ggg_H
#define PY8MEs_R130_P0_sm_gg_ggg_H

#include "Complex.h" 
#include <vector> 
#include <set> 
#include <exception> 
#include <iostream> 

#include "Parameters_sm.h"
#include "PY8MEs.h"

using namespace std; 

namespace PY8MEs_namespace 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: g g > g g g WEIGHTED<=3 @130
//--------------------------------------------------------------------------

typedef vector<double> vec_double; 
typedef vector < vec_double > vec_vec_double; 
typedef vector<int> vec_int; 
typedef vector<bool> vec_bool; 
typedef vector < vec_int > vec_vec_int; 

class PY8MEs_R130_P0_sm_gg_ggg : public PY8ME
{
  public:

    // Check for the availability of the requested proces.
    // If available, this returns the corresponding permutation and Proc_ID  to
    // use.
    // If not available, this returns a negative Proc_ID.
    static pair < vector<int> , int > static_getPY8ME(vector<int> initial_pdgs,
        vector<int> final_pdgs, set<int> schannels = set<int> ());

    // Constructor.
    PY8MEs_R130_P0_sm_gg_ggg(Parameters_sm * model) : pars(model) {initProc();}

    // Destructor.
    ~PY8MEs_R130_P0_sm_gg_ggg(); 

    // Initialize process.
    virtual void initProc(); 

    // Calculate squared ME.
    virtual double sigmaKin(); 

    // Info on the subprocess.
    virtual string name() const {return "gg_ggg (sm)";}

    virtual int code() const {return 23130;}

    virtual string inFlux() const {return "gg";}

    virtual vector<double> getMasses(); 

    virtual void setMasses(vec_double external_masses); 
    // Set all values of the external masses to an integer mode:
    // 0 : Mass taken from the model
    // 1 : Mass taken from p_i^2 if not massless to begin with
    // 2 : Mass always taken from p_i^2.
    virtual void setExternalMassesMode(int mode); 

    // Synchronize local variables of the process that depend on the model
    // parameters
    virtual void syncProcModelParams(); 

    // Tell Pythia that sigmaHat returns the ME^2
    virtual bool convertM2() const {return true;}

    // Access to getPY8ME with polymorphism from a non-static context
    virtual pair < vector<int> , int > getPY8ME(vector<int> initial_pdgs,
        vector<int> final_pdgs, set<int> schannels = set<int> ())
    {
      return static_getPY8ME(initial_pdgs, final_pdgs, schannels); 
    }

    // Set momenta
    virtual void setMomenta(vector < vec_double > momenta_picked); 

    // Set color configuration to use. An empty vector means sum over all.
    virtual void setColors(vector<int> colors_picked); 

    // Set the helicity configuration to use. Am empty vector means sum over
    // all.
    virtual void setHelicities(vector<int> helicities_picked); 

    // Set the permutation to use (will apply to momenta, colors and helicities)
    virtual void setPermutation(vector<int> perm_picked); 

    // Set the proc_ID to use
    virtual void setProcID(int procID_picked); 

    // Access to all the helicity and color configurations for a given process
    virtual vector < vec_int > getColorConfigs(int specify_proc_ID = -1,
        vector<int> permutation = vector<int> ());
    virtual vector < vec_int > getHelicityConfigs(vector<int> permutation =
        vector<int> ());

    // Maps of Helicity <-> hel_ID and ColorConfig <-> colorConfig_ID.
    virtual vector<int> getHelicityConfigForID(int hel_ID, vector<int>
        permutation = vector<int> ());
    virtual int getHelicityIDForConfig(vector<int> hel_config, vector<int>
        permutation = vector<int> ());
    virtual vector<int> getColorConfigForID(int color_ID, int specify_proc_ID =
        -1, vector<int> permutation = vector<int> ());
    virtual int getColorIDForConfig(vector<int> color_config, int
        specify_proc_ID = -1, vector<int> permutation = vector<int> ());
    virtual int getColorFlowRelativeNCPower(int color_flow_ID, int
        specify_proc_ID = -1);

    // Access previously computed results
    virtual vector < vec_double > getAllResults(int specify_proc_ID = -1); 
    virtual double getResult(int helicity_ID, int color_ID, int specify_proc_ID
        = -1);

    // Accessors
    Parameters_sm * getModel() {return pars;}
    void setModel(Parameters_sm * model) {pars = model;}

    // Invert the permutation mapping
    vector<int> invert_mapping(vector<int> mapping); 

    // Control whether to include the symmetry factors or not
    virtual void setIncludeSymmetryFactors(bool OnOff) 
    {
      include_symmetry_factors = OnOff; 
    }
    virtual bool getIncludeSymmetryFactors() {return include_symmetry_factors;}
    virtual int getSymmetryFactor() {return denom_iden[proc_ID];}

    // Control whether to include helicity averaging factors or not
    virtual void setIncludeHelicityAveragingFactors(bool OnOff) 
    {
      include_helicity_averaging_factors = OnOff; 
    }
    virtual bool getIncludeHelicityAveragingFactors() 
    {
      return include_helicity_averaging_factors; 
    }
    virtual int getHelicityAveragingFactor() {return denom_hels[proc_ID];}

    // Control whether to include color averaging factors or not
    virtual void setIncludeColorAveragingFactors(bool OnOff) 
    {
      include_color_averaging_factors = OnOff; 
    }
    virtual bool getIncludeColorAveragingFactors() 
    {
      return include_color_averaging_factors; 
    }
    virtual int getColorAveragingFactor() {return denom_colors[proc_ID];}

  private:

    // Private functions to calculate the matrix element for all subprocesses
    // Calculate wavefunctions
    void calculate_wavefunctions(const int hel[]); 
    static const int nwavefuncs = 33; 
    Complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 45; 
    Complex<double> amp[namplitudes]; 
    double matrix_13gg_ggg(); 

    // Constants for array limits
    static const int nexternal = 5; 
    static const int ninitial = 2; 
    static const int nprocesses = 1; 
    static const int nreq_s_channels = 0; 
    static const int ncomb = 32; 

    // Helicities for the process
    static int helicities[ncomb][nexternal]; 

    // Normalization factors the various processes
    static int denom_colors[nprocesses]; 
    static int denom_hels[nprocesses]; 
    static int denom_iden[nprocesses]; 

    // Control whether to include symmetry factors or not
    bool include_symmetry_factors; 
    // Control whether to include helicity averaging factors or not
    bool include_helicity_averaging_factors; 
    // Control whether to include color averaging factors or not
    bool include_color_averaging_factors; 

    // Color flows, used when selecting color
    vector < vec_double > jamp2; 

    // Store individual results (for each color flow, helicity configurations
    // and proc_ID)
    // computed in the last call to sigmaKin().
    vector < vec_vec_double > all_results; 

    // required s-channels specified
    static std::set<int> s_channel_proc; 

    // vector with external particle masses
    vector<double> mME; 

    // vector with momenta (to be changed for each event)
    vector < double * > p; 

    // external particles permutation (to be changed for each event)
    vector<int> perm; 

    // vector with colors (to be changed for each event)
    vector<int> user_colors; 

    // vector with helicities (to be changed for each event)
    vector<int> user_helicities; 

    // Process ID (to be changed for each event)
    int proc_ID; 

    // All color configurations
    void initColorConfigs(); 
    vector < vec_vec_int > color_configs; 

    // Color flows relative N_c power (conventions are such that all elements
    // on the color matrix diagonal are identical).
    vector < vec_int > jamp_nc_relative_power; 

    // Model pointer to be used by this matrix element
    Parameters_sm * pars; 

}; 

}  // end namespace PY8MEs_namespace

#endif  // PY8MEs_R130_P0_sm_gg_ggg_H

