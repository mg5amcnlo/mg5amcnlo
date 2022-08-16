//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef PY8MEs_R221_P2_sm_ckm_qq_qq_H
#define PY8MEs_R221_P2_sm_ckm_qq_qq_H

#include "Complex.h" 
#include <vector> 
#include <set> 
#include <exception> 
#include <iostream> 

#include "Parameters_sm_ckm.h"
#include "PY8MEs.h"

using namespace std; 

namespace PY8MEs_namespace 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: u u > u u QCD=0 @221
// Process: c c > c c QCD=0 @221
// Process: u u~ > u u~ QCD=0 @221
// Process: c c~ > c c~ QCD=0 @221
// Process: d d > d d QCD=0 @221
// Process: s s > s s QCD=0 @221
// Process: d d~ > d d~ QCD=0 @221
// Process: s s~ > s s~ QCD=0 @221
// Process: u~ u~ > u~ u~ QCD=0 @221
// Process: c~ c~ > c~ c~ QCD=0 @221
// Process: d~ d~ > d~ d~ QCD=0 @221
// Process: s~ s~ > s~ s~ QCD=0 @221
// Process: u d > u d QCD=0 @221
// Process: c s > c s QCD=0 @221
// Process: u s > u s QCD=0 @221
// Process: u u~ > d d~ QCD=0 @221
// Process: c c~ > s s~ QCD=0 @221
// Process: u u~ > s s~ QCD=0 @221
// Process: u d~ > u d~ QCD=0 @221
// Process: c s~ > c s~ QCD=0 @221
// Process: u s~ > u s~ QCD=0 @221
// Process: c d > c d QCD=0 @221
// Process: c c~ > d d~ QCD=0 @221
// Process: c d~ > c d~ QCD=0 @221
// Process: d u~ > d u~ QCD=0 @221
// Process: s c~ > s c~ QCD=0 @221
// Process: d c~ > d c~ QCD=0 @221
// Process: d d~ > u u~ QCD=0 @221
// Process: s s~ > c c~ QCD=0 @221
// Process: d d~ > c c~ QCD=0 @221
// Process: s u~ > s u~ QCD=0 @221
// Process: s s~ > u u~ QCD=0 @221
// Process: u~ d~ > u~ d~ QCD=0 @221
// Process: c~ s~ > c~ s~ QCD=0 @221
// Process: u~ s~ > u~ s~ QCD=0 @221
// Process: c~ d~ > c~ d~ QCD=0 @221
// Process: u c > u c QCD=0 @221
// Process: u u~ > c c~ QCD=0 @221
// Process: c c~ > u u~ QCD=0 @221
// Process: u c~ > u c~ QCD=0 @221
// Process: c u~ > c u~ QCD=0 @221
// Process: d s > d s QCD=0 @221
// Process: d d~ > s s~ QCD=0 @221
// Process: s s~ > d d~ QCD=0 @221
// Process: d s~ > d s~ QCD=0 @221
// Process: s d~ > s d~ QCD=0 @221
// Process: u~ c~ > u~ c~ QCD=0 @221
// Process: d~ s~ > d~ s~ QCD=0 @221
// Process: u d > u s QCD=0 @221
// Process: u s > c s QCD=0 @221
// Process: u d > c d QCD=0 @221
// Process: c d > c s QCD=0 @221
// Process: u d > c s QCD=0 @221
// Process: u s > u d QCD=0 @221
// Process: c s > u s QCD=0 @221
// Process: u s > c d QCD=0 @221
// Process: c d > u s QCD=0 @221
// Process: u u~ > d s~ QCD=0 @221
// Process: c u~ > s s~ QCD=0 @221
// Process: d s~ > u u~ QCD=0 @221
// Process: s s~ > c u~ QCD=0 @221
// Process: u u~ > s d~ QCD=0 @221
// Process: u c~ > s s~ QCD=0 @221
// Process: s d~ > u u~ QCD=0 @221
// Process: s s~ > u c~ QCD=0 @221
// Process: u c~ > d d~ QCD=0 @221
// Process: c c~ > s d~ QCD=0 @221
// Process: d d~ > u c~ QCD=0 @221
// Process: s d~ > c c~ QCD=0 @221
// Process: u c~ > d s~ QCD=0 @221
// Process: c u~ > s d~ QCD=0 @221
// Process: d s~ > u c~ QCD=0 @221
// Process: s d~ > c u~ QCD=0 @221
// Process: u c~ > s d~ QCD=0 @221
// Process: s d~ > u c~ QCD=0 @221
// Process: u d~ > u s~ QCD=0 @221
// Process: c s~ > u s~ QCD=0 @221
// Process: u d~ > c d~ QCD=0 @221
// Process: c s~ > c d~ QCD=0 @221
// Process: u d~ > c s~ QCD=0 @221
// Process: c s~ > u d~ QCD=0 @221
// Process: u s~ > u d~ QCD=0 @221
// Process: u s~ > c s~ QCD=0 @221
// Process: u s~ > c d~ QCD=0 @221
// Process: c d > u d QCD=0 @221
// Process: c s > c d QCD=0 @221
// Process: c s > u d QCD=0 @221
// Process: c u~ > d d~ QCD=0 @221
// Process: c c~ > d s~ QCD=0 @221
// Process: d d~ > c u~ QCD=0 @221
// Process: d s~ > c c~ QCD=0 @221
// Process: c u~ > d s~ QCD=0 @221
// Process: d s~ > c u~ QCD=0 @221
// Process: c d~ > u d~ QCD=0 @221
// Process: c d~ > c s~ QCD=0 @221
// Process: c d~ > u s~ QCD=0 @221
// Process: d u~ > d c~ QCD=0 @221
// Process: s c~ > d c~ QCD=0 @221
// Process: d u~ > s u~ QCD=0 @221
// Process: s c~ > s u~ QCD=0 @221
// Process: d u~ > s c~ QCD=0 @221
// Process: s c~ > d u~ QCD=0 @221
// Process: d c~ > d u~ QCD=0 @221
// Process: d c~ > s c~ QCD=0 @221
// Process: d c~ > s u~ QCD=0 @221
// Process: s u~ > d u~ QCD=0 @221
// Process: s u~ > s c~ QCD=0 @221
// Process: s u~ > d c~ QCD=0 @221
// Process: u~ d~ > u~ s~ QCD=0 @221
// Process: u~ s~ > c~ s~ QCD=0 @221
// Process: u~ d~ > c~ d~ QCD=0 @221
// Process: c~ d~ > c~ s~ QCD=0 @221
// Process: u~ d~ > c~ s~ QCD=0 @221
// Process: u~ s~ > u~ d~ QCD=0 @221
// Process: c~ s~ > u~ s~ QCD=0 @221
// Process: u~ s~ > c~ d~ QCD=0 @221
// Process: c~ d~ > u~ s~ QCD=0 @221
// Process: c~ d~ > u~ d~ QCD=0 @221
// Process: c~ s~ > c~ d~ QCD=0 @221
// Process: c~ s~ > u~ d~ QCD=0 @221
//--------------------------------------------------------------------------

typedef vector<double> vec_double; 
typedef vector < vec_double > vec_vec_double; 
typedef vector<int> vec_int; 
typedef vector<bool> vec_bool; 
typedef vector < vec_int > vec_vec_int; 

class PY8MEs_R221_P2_sm_ckm_qq_qq : public PY8ME
{
  public:

    // Check for the availability of the requested proces.
    // If available, this returns the corresponding permutation and Proc_ID  to
    // use.
    // If not available, this returns a negative Proc_ID.
    static pair < vector<int> , int > static_getPY8ME(vector<int> initial_pdgs,
        vector<int> final_pdgs, set<int> schannels = set<int> ());

    // Constructor.
    PY8MEs_R221_P2_sm_ckm_qq_qq(Parameters_sm_ckm * model) : pars(model) 
    {
      initProc(); 
    }

    // Destructor.
    ~PY8MEs_R221_P2_sm_ckm_qq_qq(); 

    // Initialize process.
    virtual void initProc(); 

    // Calculate squared ME.
    virtual double sigmaKin(); 

    // Info on the subprocess.
    virtual string name() const {return "qq_qq (sm_ckm)";}

    virtual int code() const {return 32102;}

    virtual string inFlux() const {return "qq";}

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
    Parameters_sm_ckm * getModel() {return pars;}
    void setModel(Parameters_sm_ckm * model) {pars = model;}

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
    static const int nwavefuncs = 50; 
    Complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 127; 
    Complex<double> amp[namplitudes]; 
    double matrix_221_uu_uu(); 
    double matrix_221_uux_uux(); 
    double matrix_221_dd_dd(); 
    double matrix_221_ddx_ddx(); 
    double matrix_221_uxux_uxux(); 
    double matrix_221_dxdx_dxdx(); 
    double matrix_221_ud_ud(); 
    double matrix_221_us_us(); 
    double matrix_221_uux_ddx(); 
    double matrix_221_uux_ssx(); 
    double matrix_221_udx_udx(); 
    double matrix_221_usx_usx(); 
    double matrix_221_cd_cd(); 
    double matrix_221_ccx_ddx(); 
    double matrix_221_cdx_cdx(); 
    double matrix_221_dux_dux(); 
    double matrix_221_dcx_dcx(); 
    double matrix_221_ddx_uux(); 
    double matrix_221_ddx_ccx(); 
    double matrix_221_sux_sux(); 
    double matrix_221_ssx_uux(); 
    double matrix_221_uxdx_uxdx(); 
    double matrix_221_uxsx_uxsx(); 
    double matrix_221_cxdx_cxdx(); 
    double matrix_221_uc_uc(); 
    double matrix_221_uux_ccx(); 
    double matrix_221_ucx_ucx(); 
    double matrix_221_ds_ds(); 
    double matrix_221_ddx_ssx(); 
    double matrix_221_dsx_dsx(); 
    double matrix_221_uxcx_uxcx(); 
    double matrix_221_dxsx_dxsx(); 
    double matrix_221_ud_us(); 
    double matrix_221_ud_cd(); 
    double matrix_221_ud_cs(); 
    double matrix_221_us_ud(); 
    double matrix_221_us_cd(); 
    double matrix_221_uux_dsx(); 
    double matrix_221_uux_sdx(); 
    double matrix_221_ucx_ddx(); 
    double matrix_221_ucx_dsx(); 
    double matrix_221_ucx_sdx(); 
    double matrix_221_udx_usx(); 
    double matrix_221_udx_cdx(); 
    double matrix_221_udx_csx(); 
    double matrix_221_usx_udx(); 
    double matrix_221_usx_cdx(); 
    double matrix_221_cd_ud(); 
    double matrix_221_cs_ud(); 
    double matrix_221_cux_ddx(); 
    double matrix_221_cux_dsx(); 
    double matrix_221_cdx_udx(); 
    double matrix_221_cdx_usx(); 
    double matrix_221_dux_dcx(); 
    double matrix_221_dux_sux(); 
    double matrix_221_dux_scx(); 
    double matrix_221_dcx_dux(); 
    double matrix_221_dcx_sux(); 
    double matrix_221_sux_dux(); 
    double matrix_221_sux_dcx(); 
    double matrix_221_uxdx_uxsx(); 
    double matrix_221_uxdx_cxdx(); 
    double matrix_221_uxdx_cxsx(); 
    double matrix_221_uxsx_uxdx(); 
    double matrix_221_uxsx_cxdx(); 
    double matrix_221_cxdx_uxdx(); 
    double matrix_221_cxsx_uxdx(); 

    // Constants for array limits
    static const int nexternal = 4; 
    static const int ninitial = 2; 
    static const int nprocesses = 67; 
    static const int nreq_s_channels = 0; 
    static const int ncomb = 16; 

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
    Parameters_sm_ckm * pars; 

}; 

}  // end namespace PY8MEs_namespace

#endif  // PY8MEs_R221_P2_sm_ckm_qq_qq_H

