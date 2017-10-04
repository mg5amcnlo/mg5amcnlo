//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef PY8MEs_R1_P7_sm_lb_lb_H
#define PY8MEs_R1_P7_sm_lb_lb_H

#include <complex> 
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
// Process: e- b > e- b WEIGHTED<=4 @1
// Process: mu- b > mu- b WEIGHTED<=4 @1
// Process: e- b~ > e- b~ WEIGHTED<=4 @1
// Process: mu- b~ > mu- b~ WEIGHTED<=4 @1
// Process: e+ b > e+ b WEIGHTED<=4 @1
// Process: mu+ b > mu+ b WEIGHTED<=4 @1
// Process: e+ b~ > e+ b~ WEIGHTED<=4 @1
// Process: mu+ b~ > mu+ b~ WEIGHTED<=4 @1
// Process: ve b > ve b WEIGHTED<=4 @1
// Process: vm b > vm b WEIGHTED<=4 @1
// Process: vt b > vt b WEIGHTED<=4 @1
// Process: ve b~ > ve b~ WEIGHTED<=4 @1
// Process: vm b~ > vm b~ WEIGHTED<=4 @1
// Process: vt b~ > vt b~ WEIGHTED<=4 @1
// Process: ve~ b > ve~ b WEIGHTED<=4 @1
// Process: vm~ b > vm~ b WEIGHTED<=4 @1
// Process: vt~ b > vt~ b WEIGHTED<=4 @1
// Process: ve~ b~ > ve~ b~ WEIGHTED<=4 @1
// Process: vm~ b~ > vm~ b~ WEIGHTED<=4 @1
// Process: vt~ b~ > vt~ b~ WEIGHTED<=4 @1
//--------------------------------------------------------------------------

typedef vector<double> vec_double; 
typedef vector < vec_double > vec_vec_double; 
typedef vector<int> vec_int; 
typedef vector<bool> vec_bool; 
typedef vector < vec_int > vec_vec_int; 

class PY8MEs_R1_P7_sm_lb_lb : public PY8ME
{
  public:

    // Check for the availability of the requested proces.
    // If available, this returns the corresponding permutation and Proc_ID  to
    // use.
    // If not available, this returns a negative Proc_ID.
    static pair < vector<int> , int > static_getPY8ME(vector<int> initial_pdgs,
        vector<int> final_pdgs, set<int> schannels = set<int> ());

    // Constructor.
    PY8MEs_R1_P7_sm_lb_lb(Parameters_sm * model) : pars(model) {initProc();}

    // Destructor.
    ~PY8MEs_R1_P7_sm_lb_lb(); 

    // Initialize process.
    virtual void initProc(); 

    // Calculate squared ME.
    virtual double sigmaKin(); 

    // Info on the subprocess.
    virtual string name() const {return "lb_lb (sm)";}

    virtual int code() const {return 10107;}

    virtual string inFlux() const {return "ff";}

    virtual vector<double> getMasses() const {return mME;}

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
    void set_include_symmetry_factors(bool OnOff) 
    {
      include_symmetry_factors = OnOff; 
    }
    bool get_include_symmetry_factors() {return include_symmetry_factors;}

  private:

    // Private functions to calculate the matrix element for all subprocesses
    // Calculate wavefunctions
    void calculate_wavefunctions(const int hel[]); 
    static const int nwavefuncs = 14; 
    std::complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 12; 
    std::complex<double> amp[namplitudes]; 
    double matrix_1_emb_emb(); 
    double matrix_1_embx_embx(); 
    double matrix_1_epb_epb(); 
    double matrix_1_epbx_epbx(); 
    double matrix_1_veb_veb(); 
    double matrix_1_vebx_vebx(); 
    double matrix_1_vexb_vexb(); 
    double matrix_1_vexbx_vexbx(); 

    // Constants for array limits
    static const int nexternal = 4; 
    static const int ninitial = 2; 
    static const int nprocesses = 8; 
    static const int nreq_s_channels = 0; 
    static const int ncomb = 16; 

    // Helicities for the process
    static int helicities[ncomb][nexternal]; 

    // Control whether to include symmetry factors or not
    bool include_symmetry_factors; 

    // required s-channels specified
    //static int req_s_channels[nreq_s_channels]; 

    // Color flows, used when selecting color
    vector < vec_double > jamp2; 

    // Store individual results (for each color flow, helicity configurations
    // and proc_ID)
    // computed in the last call to sigmaKin().
    vector < vec_vec_double > all_results; 

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

#endif  // PY8MEs_R1_P7_sm_lb_lb_H

