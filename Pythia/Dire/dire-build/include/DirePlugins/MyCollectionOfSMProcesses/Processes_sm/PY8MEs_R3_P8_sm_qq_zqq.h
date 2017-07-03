//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. 2.5.3, 2017-03-09
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef PY8MEs_R3_P8_sm_qq_zqq_H
#define PY8MEs_R3_P8_sm_qq_zqq_H

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
// Process: u u > z u u WEIGHTED<=4 @3
// Process: c c > z c c WEIGHTED<=4 @3
// Process: u u~ > z u u~ WEIGHTED<=4 @3
// Process: c c~ > z c c~ WEIGHTED<=4 @3
// Process: d d > z d d WEIGHTED<=4 @3
// Process: s s > z s s WEIGHTED<=4 @3
// Process: d d~ > z d d~ WEIGHTED<=4 @3
// Process: s s~ > z s s~ WEIGHTED<=4 @3
// Process: u~ u~ > z u~ u~ WEIGHTED<=4 @3
// Process: c~ c~ > z c~ c~ WEIGHTED<=4 @3
// Process: d~ d~ > z d~ d~ WEIGHTED<=4 @3
// Process: s~ s~ > z s~ s~ WEIGHTED<=4 @3
// Process: u c > z u c WEIGHTED<=4 @3
// Process: u d > z u d WEIGHTED<=4 @3
// Process: u s > z u s WEIGHTED<=4 @3
// Process: c d > z c d WEIGHTED<=4 @3
// Process: c s > z c s WEIGHTED<=4 @3
// Process: u u~ > z c c~ WEIGHTED<=4 @3
// Process: c c~ > z u u~ WEIGHTED<=4 @3
// Process: u u~ > z d d~ WEIGHTED<=4 @3
// Process: u u~ > z s s~ WEIGHTED<=4 @3
// Process: c c~ > z d d~ WEIGHTED<=4 @3
// Process: c c~ > z s s~ WEIGHTED<=4 @3
// Process: u c~ > z u c~ WEIGHTED<=4 @3
// Process: c u~ > z c u~ WEIGHTED<=4 @3
// Process: u d~ > z u d~ WEIGHTED<=4 @3
// Process: u s~ > z u s~ WEIGHTED<=4 @3
// Process: c d~ > z c d~ WEIGHTED<=4 @3
// Process: c s~ > z c s~ WEIGHTED<=4 @3
// Process: d s > z d s WEIGHTED<=4 @3
// Process: d u~ > z d u~ WEIGHTED<=4 @3
// Process: d c~ > z d c~ WEIGHTED<=4 @3
// Process: s u~ > z s u~ WEIGHTED<=4 @3
// Process: s c~ > z s c~ WEIGHTED<=4 @3
// Process: d d~ > z u u~ WEIGHTED<=4 @3
// Process: d d~ > z c c~ WEIGHTED<=4 @3
// Process: s s~ > z u u~ WEIGHTED<=4 @3
// Process: s s~ > z c c~ WEIGHTED<=4 @3
// Process: d d~ > z s s~ WEIGHTED<=4 @3
// Process: s s~ > z d d~ WEIGHTED<=4 @3
// Process: d s~ > z d s~ WEIGHTED<=4 @3
// Process: s d~ > z s d~ WEIGHTED<=4 @3
// Process: u~ c~ > z u~ c~ WEIGHTED<=4 @3
// Process: u~ d~ > z u~ d~ WEIGHTED<=4 @3
// Process: u~ s~ > z u~ s~ WEIGHTED<=4 @3
// Process: c~ d~ > z c~ d~ WEIGHTED<=4 @3
// Process: c~ s~ > z c~ s~ WEIGHTED<=4 @3
// Process: d~ s~ > z d~ s~ WEIGHTED<=4 @3
//--------------------------------------------------------------------------

typedef vector<double> vec_double; 
typedef vector < vec_double > vec_vec_double; 
typedef vector<int> vec_int; 
typedef vector<bool> vec_bool; 
typedef vector < vec_int > vec_vec_int; 

class PY8MEs_R3_P8_sm_qq_zqq : public PY8ME
{
  public:

    // Check for the availability of the requested proces.
    // If available, this returns the corresponding permutation and Proc_ID  to
    // use.
    // If not available, this returns a negative Proc_ID.
    static pair < vector<int> , int > static_getPY8ME(vector<int> initial_pdgs,
        vector<int> final_pdgs, set<int> schannels = set<int> ());

    // Constructor.
    PY8MEs_R3_P8_sm_qq_zqq(Parameters_sm * model) {pars = model; initProc();}

    // Destructor.
    ~PY8MEs_R3_P8_sm_qq_zqq(); 

    // Initialize process.
    virtual void initProc(); 

    // Calculate squared ME.
    virtual double sigmaKin(); 

    // Info on the subprocess.
    virtual string name() const {return "qq_zqq (sm)";}

    virtual int code() const {return 10308;}

    virtual string inFlux() const {return "qq";}

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

    // Access previously computed results
    virtual vector < vec_double > getAllResults(int specify_proc_ID = -1); 
    virtual double getResult(int helicity_ID, int color_ID, int specify_proc_ID
        = -1);

    // Accessors
    Parameters_sm * getModel() {return pars;}
    void setModel(Parameters_sm * model) {pars = model;}

  private:

    // Private functions to calculate the matrix element for all subprocesses
    // Calculate wavefunctions
    void calculate_wavefunctions(const int hel[]); 
    static const int nwavefuncs = 35; 
    std::complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 104; 
    std::complex<double> amp[namplitudes]; 
    double matrix_3_uu_zuu(); 
    double matrix_3_uux_zuux(); 
    double matrix_3_dd_zdd(); 
    double matrix_3_ddx_zddx(); 
    double matrix_3_uxux_zuxux(); 
    double matrix_3_dxdx_zdxdx(); 
    double matrix_3_uc_zuc(); 
    double matrix_3_ud_zud(); 
    double matrix_3_uux_zccx(); 
    double matrix_3_uux_zddx(); 
    double matrix_3_ucx_zucx(); 
    double matrix_3_udx_zudx(); 
    double matrix_3_ds_zds(); 
    double matrix_3_dux_zdux(); 
    double matrix_3_ddx_zuux(); 
    double matrix_3_ddx_zssx(); 
    double matrix_3_dsx_zdsx(); 
    double matrix_3_uxcx_zuxcx(); 
    double matrix_3_uxdx_zuxdx(); 
    double matrix_3_dxsx_zdxsx(); 

    // Constants for array limits
    static const int nexternal = 5; 
    static const int ninitial = 2; 
    static const int nprocesses = 20; 
    static const int nreq_s_channels = 0; 
    static const int ncomb = 48; 

    // Helicities for the process
    static int helicities[ncomb][nexternal]; 

    // required s-channels specified
    static int req_s_channels[nreq_s_channels]; 

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

    // Model pointer to be used by this matrix element
    Parameters_sm * pars; 
}; 

}  // end namespace PY8MEs_namespace

#endif  // PY8MEs_R3_P8_sm_qq_zqq_H

