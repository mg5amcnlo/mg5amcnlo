//==========================================================================
// This file has been automatically generated for Pythia 8
// MadGraph5_aMC@NLO v. 2.6.0, 2017-08-16
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef PY8MEs_R906_P4_heft_ckm_qq_hqq_H
#define PY8MEs_R906_P4_heft_ckm_qq_hqq_H

#include "Complex.h" 
#include <vector> 
#include <set> 
#include <exception> 
#include <iostream> 

#include "Parameters_heft_ckm.h"
#include "PY8MEs.h"

using namespace std; 

namespace PY8MEs_namespace 
{
//==========================================================================
// A class for calculating the matrix elements for
// Process: u u > h u u HIG=0 HIW=0 QCD=0 / b @906
// Process: c c > h c c HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d > h d d HIG=0 HIW=0 QCD=0 / b @906
// Process: s s > h s s HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ u~ > h u~ u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ c~ > h c~ c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d~ d~ > h d~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s~ s~ > h s~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c > h u c HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u d > h c s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u s > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h u s HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u u~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h u u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u c~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h u c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u d~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c s~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u s~ > h c d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h c d HIG=0 HIW=0 QCD=0 / b @906
// Process: c s > h u d HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c c~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h c c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c u~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h c u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h u d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h c s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c d~ > h u s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s > h d s HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d u~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s c~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d c~ > h s u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d d~ > h s s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s s~ > h d d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d s~ > h d s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s d~ > h s d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h d u~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h s c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: s u~ > h d c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ c~ > h u~ c~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ d~ > h c~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: u~ s~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h u~ s~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ d~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h c~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: c~ s~ > h u~ d~ HIG=0 HIW=0 QCD=0 / b @906
// Process: d~ s~ > h d~ s~ HIG=0 HIW=0 QCD=0 / b @906
//--------------------------------------------------------------------------

typedef vector<double> vec_double; 
typedef vector < vec_double > vec_vec_double; 
typedef vector<int> vec_int; 
typedef vector<bool> vec_bool; 
typedef vector < vec_int > vec_vec_int; 

class PY8MEs_R906_P4_heft_ckm_qq_hqq : public PY8ME
{
  public:

    // Check for the availability of the requested proces.
    // If available, this returns the corresponding permutation and Proc_ID  to
    // use.
    // If not available, this returns a negative Proc_ID.
    static pair < vector<int> , int > static_getPY8ME(vector<int> initial_pdgs,
        vector<int> final_pdgs, set<int> schannels = set<int> ());

    // Constructor.
    PY8MEs_R906_P4_heft_ckm_qq_hqq(Parameters_heft_ckm * model) : pars(model) 
    {
      initProc(); 
    }

    // Destructor.
    ~PY8MEs_R906_P4_heft_ckm_qq_hqq(); 

    // Initialize process.
    virtual void initProc(); 

    // Calculate squared ME.
    virtual double sigmaKin(); 

    // Info on the subprocess.
    virtual string name() const {return "qq_hqq (heft_ckm)";}

    virtual int code() const {return 100604;}

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
    Parameters_heft_ckm * getModel() {return pars;}
    void setModel(Parameters_heft_ckm * model) {pars = model;}

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
    static const int nwavefuncs = 53; 
    Complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 90; 
    Complex<double> amp[namplitudes]; 
    double matrix_906_uu_huu_no_b(); 
    double matrix_906_ud_hud_no_b(); 
    double matrix_906_us_hus_no_b(); 
    double matrix_906_uux_huux_no_b(); 
    double matrix_906_uux_hddx_no_b(); 
    double matrix_906_uux_hssx_no_b(); 
    double matrix_906_udx_hudx_no_b(); 
    double matrix_906_usx_husx_no_b(); 
    double matrix_906_cd_hcd_no_b(); 
    double matrix_906_ccx_hddx_no_b(); 
    double matrix_906_cdx_hcdx_no_b(); 
    double matrix_906_dd_hdd_no_b(); 
    double matrix_906_dux_hdux_no_b(); 
    double matrix_906_dcx_hdcx_no_b(); 
    double matrix_906_ddx_huux_no_b(); 
    double matrix_906_ddx_hccx_no_b(); 
    double matrix_906_ddx_hddx_no_b(); 
    double matrix_906_sux_hsux_no_b(); 
    double matrix_906_ssx_huux_no_b(); 
    double matrix_906_uxux_huxux_no_b(); 
    double matrix_906_uxdx_huxdx_no_b(); 
    double matrix_906_uxsx_huxsx_no_b(); 
    double matrix_906_cxdx_hcxdx_no_b(); 
    double matrix_906_dxdx_hdxdx_no_b(); 
    double matrix_906_uc_huc_no_b(); 
    double matrix_906_ud_hus_no_b(); 
    double matrix_906_ud_hcd_no_b(); 
    double matrix_906_ud_hcs_no_b(); 
    double matrix_906_us_hud_no_b(); 
    double matrix_906_us_hcd_no_b(); 
    double matrix_906_uux_hccx_no_b(); 
    double matrix_906_uux_hdsx_no_b(); 
    double matrix_906_uux_hsdx_no_b(); 
    double matrix_906_ucx_hucx_no_b(); 
    double matrix_906_ucx_hddx_no_b(); 
    double matrix_906_ucx_hdsx_no_b(); 
    double matrix_906_ucx_hsdx_no_b(); 
    double matrix_906_udx_husx_no_b(); 
    double matrix_906_udx_hcdx_no_b(); 
    double matrix_906_udx_hcsx_no_b(); 
    double matrix_906_usx_hudx_no_b(); 
    double matrix_906_usx_hcdx_no_b(); 
    double matrix_906_cd_hud_no_b(); 
    double matrix_906_cs_hud_no_b(); 
    double matrix_906_cux_hddx_no_b(); 
    double matrix_906_cux_hdsx_no_b(); 
    double matrix_906_cdx_hudx_no_b(); 
    double matrix_906_cdx_husx_no_b(); 
    double matrix_906_ds_hds_no_b(); 
    double matrix_906_dux_hdcx_no_b(); 
    double matrix_906_dux_hsux_no_b(); 
    double matrix_906_dux_hscx_no_b(); 
    double matrix_906_dcx_hdux_no_b(); 
    double matrix_906_dcx_hsux_no_b(); 
    double matrix_906_ddx_hssx_no_b(); 
    double matrix_906_dsx_hdsx_no_b(); 
    double matrix_906_sux_hdux_no_b(); 
    double matrix_906_sux_hdcx_no_b(); 
    double matrix_906_uxcx_huxcx_no_b(); 
    double matrix_906_uxdx_huxsx_no_b(); 
    double matrix_906_uxdx_hcxdx_no_b(); 
    double matrix_906_uxdx_hcxsx_no_b(); 
    double matrix_906_uxsx_huxdx_no_b(); 
    double matrix_906_uxsx_hcxdx_no_b(); 
    double matrix_906_cxdx_huxdx_no_b(); 
    double matrix_906_cxsx_huxdx_no_b(); 
    double matrix_906_dxsx_hdxsx_no_b(); 

    // Constants for array limits
    static const int nexternal = 5; 
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
    Parameters_heft_ckm * pars; 

}; 

}  // end namespace PY8MEs_namespace

#endif  // PY8MEs_R906_P4_heft_ckm_qq_hqq_H

