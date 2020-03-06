#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <sstream>

#include "appl_grid/appl_grid.h"
#include "appl_grid/lumi_pdf.h"
#include "LHAPDF/LHAPDF.h"
#include "orders.h"

/*
  fNLO mode of aMCatNLO
*/

// Declare grids
std::vector<appl::grid> grid_obs;

// translates an index from the range [0, __amp_split_size) to the indices need by `fill_grid`
std::vector<std::vector<int>> translation_tables;
  
// Declare the input and output grids
std::string grid_filename_in;
std::string grid_filename_out;
  
// Bin width
std::vector< std::vector<double> > binwidths;

////////////////////////////////////////////////////////////////////////////////////
  
// Maximum number of (pairwise) suprocesses
const int __max_nproc__ = 121;

// Information defined at the generation (configuration) step, that does 
// not vary event by event
typedef struct {
  int amp_split_size; // Maximum number of coupling-combinations
  int qcdpower[__amp_split_size]; // Power of alpha_s for each amp_split
  int qedpower[__amp_split_size]; // Power of alpha for each amp_split
} __amcatnlo_common_fixed__;

// Map of the PDF combinations from aMC@NLO - structure for each 
// "subprocess" i, has some number nproc[i] pairs of parton
// combinations. To be used together with the info in appl_flavmap.
typedef struct { 
  int lumimap[__max_nproc__][__max_nproc__][2]; // (paired) subprocesses per combination
  int nproc[__max_nproc__];                     // number of separate (pairwise) subprocesses for this combination
  int nlumi;                                    // overall number of combinations ( 0 < nlumi <= __mxpdflumi__ )
} __amcatnlo_common_lumi__;

// Event weights, kinematics, etc. that are different event by event
typedef struct {
  double  x1[4],x2[4]; 
  double  muF2[4],muR2[4],muQES2[4];
  double  W0[__amp_split_size][4],WR[__amp_split_size][4];
  double  WF[__amp_split_size][4],WB[__amp_split_size][4];
  int     flavmap[4];
} __amcatnlo_common_weights__;

// Parameters of the grids.
// These parameters can optionally be singularly specified by the user,
// but if no specification is given, the code will use the default values.
typedef struct {
  double Q2min,Q2max;
  double xmin,xmax;
  int    nQ2,Q2order;
  int    nx,xorder;
} __amcatnlo_common_grid__;

// Parameters of the histograms
typedef struct {
  double www_histo,norm_histo;
  double obs_histo;
  double obs_min,obs_max;
  double obs_bins[101];
  int    obs_nbins;
  int    itype_histo;
  int    amp_pos;
  int    obs_num;
} __amcatnlo_common_histokin__;

// Event weight and cross section
typedef struct {
  double event_weight,vegaswgt;
  double xsec12,xsec11,xsec20;
} __amcatnlo_common_reco__;


extern "C" __amcatnlo_common_fixed__    appl_common_fixed_;
extern "C" __amcatnlo_common_lumi__     appl_common_lumi_;
extern "C" __amcatnlo_common_weights__  appl_common_weights_;
extern "C" __amcatnlo_common_grid__     appl_common_grid_;
extern "C" __amcatnlo_common_histokin__ appl_common_histokin_;
extern "C" __amcatnlo_common_reco__     appl_common_reco_;

// Check if a file exists
bool file_exists(const std::string& s) {   
  if(std::FILE* testfile=std::fopen(s.c_str(),"r")) { 
    std::fclose(testfile);
    return true;
  }
  else return false;
}

// Banner
std::string Banner() {
    std::stringstream banner("\n");
    banner << "    █████╗ ███╗   ███╗ ██████╗██████╗ ██╗      █████╗ ███████╗████████╗\n";
    banner << "   ██╔══██╗████╗ ████║██╔════╝██╔══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝\n";
    banner << "   ███████║██╔████╔██║██║     ██████╔╝██║     ███████║███████╗   ██║\n";
    banner << "   ██╔══██║██║╚██╔╝██║██║     ██╔══██╗██║     ██╔══██║╚════██║   ██║\n";
    banner << "   ██║  ██║██║ ╚═╝ ██║╚██████╗██████╔╝███████╗██║  ██║███████║   ██║\n";
    banner << "   ╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝╚═════╝ ╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝ \n";
    return banner.str();
  }

extern "C" void appl_init_() {

    // Grid Initialization and definition of the observables.
  // Construct the input file name according to its position in the 
  // vector "grid_obs".
  std::ostringstream ss;
  ss << grid_obs.size();
  grid_filename_in = "grid_obs_" + ss.str() + "_in.root";

  // Check that the grid file exists. If so read the grid from the file,
  // otherwise create a new grid from scratch.
  if(file_exists(grid_filename_in)) { 
    std::cout << "amcblast INFO: Reading existing APPLgrid from file " << grid_filename_in << " ..." << std::endl;
    // Open the existing grid
    grid_obs.emplace_back(grid_filename_in);

    auto const& order_ids = grid_obs.back().order_ids();
    translation_tables.emplace_back();
    auto& translation_table = translation_tables.back();

    int qcd_power = -1;
    int qed_power = -1;

    // when loading the grids, the stored orders might be sorted differently
    for (int i = 0; i != appl_common_fixed_.amp_split_size; ++i)
    {
        int const alphs = appl_common_fixed_.qcdpower[i] / 2;
        int const alpha = appl_common_fixed_.qedpower[i] / 2;

        // try to find the W0/WB grid
        auto const it = std::find(order_ids.begin(), order_ids.end(),
            appl::order_id(alphs, alpha, 0, 0));

        // TODO: can this happen?
        assert( it != order_ids.end() );

        auto const index = std::distance(order_ids.begin(), it);

        std::cout << "[amcblast] mg5_aMC O(as^" << alphs << " a^" << alpha << ") -> " << index
            << '\n';

        translation_table.push_back(index);
    }

    std::cout << "[amcblast] loaded grid the following coupling orders:\n";

    for (auto const& order : order_ids)
    {
        std::cout << "[amcblast] O(as^" << order.alphs() << " a^" << order.alpha() << "), LR^"
            << order.lmur2() << ", LF^" << order.lmuf2() << '\n';
    }

    std::cout << std::flush;
  }
  // If the grid does not exist, book it after having defined all the
  // relevant parameters.
  else {
    std::cout << "amcblast INFO: Booking grid from scratch with name " << grid_filename_in << " ..." << std::endl;

    int lo_power = 9999;
    int nlo_power = 0;

    for (int i = 0; i != appl_common_fixed_.amp_split_size; ++i)
    {
        int sum = appl_common_fixed_.qcdpower[i] + appl_common_fixed_.qedpower[i];

        lo_power = std::min(lo_power, sum);
        nlo_power = std::max(nlo_power, sum);
    }

    // TODO: are there any situations is which there are NLOs but no LOs?

    // we assume that there is always at least one LO, and zero or one NLO
    assert( (nlo_power == (lo_power + 2)) || (nlo_power == lo_power) );

    std::vector<appl::order_id> order_ids;
    order_ids.reserve(appl_common_fixed_.amp_split_size);

    translation_tables.emplace_back();
    translation_tables.back().reserve(appl_common_fixed_.amp_split_size);

    for (int i = 0; i != appl_common_fixed_.amp_split_size; ++i)
    {
        int const qcd = appl_common_fixed_.qcdpower[i];
        int const qed = appl_common_fixed_.qedpower[i];
        int const sum = qcd + qed;

        translation_tables.back().push_back(order_ids.size());

        if (sum == lo_power)
        {
            // WB
            order_ids.emplace_back(qcd / 2, qed / 2, 0, 0);
        }
        else if (sum == nlo_power)
        {
            // W0
            order_ids.emplace_back(qcd / 2, qed / 2, 0, 0);
            // WR
            order_ids.emplace_back(qcd / 2, qed / 2, 0, 1);
            // WF
            order_ids.emplace_back(qcd / 2, qed / 2, 1, 0);
        }
    }

    std::cout << "[amcblast] booked grid the following coupling orders:\n";

    for (auto const& order : order_ids)
    {
        std::cout << "O(as^" << order.alphs() << " a^" << order.alpha() << "), LR^"
            << order.lmur2() << ", LF^" << order.lmuf2() << '\n';
    }

    std::cout << std::flush;

    // Define the settings for the interpolation in x and Q2.
    // These are common to all the grids computed.
    // If values larger than zero (i.e. set by the user) are found the default
    // settings are replaced with the new ones.
    int NQ2 = 30;
    // Max and min value of Q2
    double Q2min = 100;
    double Q2max = 1000000;
    // Order of the polynomial interpolation in Q2
    int Q2order = 3;
    // Number of points for the x interpolation
    int Nx = 50;
    // Min and max value of x
    double xmin = 2e-7;
    double xmax = 1;
    // Order of the polynomial interpolation in x
    int xorder = 3;

    // Replace the default values when needed
    if(appl_common_grid_.nQ2 > 0)     NQ2     = appl_common_grid_.nQ2;
    if(appl_common_grid_.Q2min > 0)   Q2min   = appl_common_grid_.Q2min;
    if(appl_common_grid_.Q2max > 0)   Q2max   = appl_common_grid_.Q2max;
    if(appl_common_grid_.Q2order > 0) Q2order = appl_common_grid_.Q2order;
    if(appl_common_grid_.nx > 0)      Nx      = appl_common_grid_.nx;
    if(appl_common_grid_.xmin > 0)    xmin    = appl_common_grid_.xmin;
    if(appl_common_grid_.xmax > 0)    xmax    = appl_common_grid_.xmax;
    if(appl_common_grid_.xorder > 0)  xorder  = appl_common_grid_.xorder;

    // Report of the grid parameters
    std::cout << std::endl;
    std::cout << "amcblast INFO: Report of the grid parameters:" << std::endl;
    std::cout << "- Q2 grid:" << std::endl;
    std::cout << "  * interpolation range: [ " << Q2min << " : " << Q2max << " ] GeV^2" << std::endl;
    std::cout << "  * number of nodes: " << NQ2 << std::endl;
    std::cout << "  * interpolation order: " << Q2order << std::endl;
    std::cout << "- x grid:" << std::endl;
    std::cout << "  * interpolation range: [ " << xmin << " : " << xmax << " ]" << std::endl;
    std::cout << "  * number of nodes: " << Nx << std::endl;
    std::cout << "  * interpolation order: " << xorder << std::endl;
    std::cout << std::endl;

    // Set up the APPLgrid PDF luminosities
    std::vector<int> pdf_luminosities;
    pdf_luminosities.push_back(appl_common_lumi_.nlumi);

    // Loop over parton luminosities
    for(int ilumi=0; ilumi<appl_common_lumi_.nlumi; ilumi++) {
      int nproc = appl_common_lumi_.nproc[ilumi];

      pdf_luminosities.push_back(ilumi);
      pdf_luminosities.push_back(nproc);

      // Loop over parton-parton combinations within each luminosity
      for(int iproc=0; iproc<nproc; iproc++) {
        pdf_luminosities.push_back(appl_common_lumi_.lumimap[ilumi][iproc][0]);
        pdf_luminosities.push_back(appl_common_lumi_.lumimap[ilumi][iproc][1]);
      }
    }

    // Use a name for the PDF combination type ending with .config.
    // This will configure from a file with the same format as the 
    // aMC@NLO initial_states_map.dat file unless a serialised 
    // vector including the combinations is also passed to the 
    // constructor, as is the case here.
    // Assign to the luminosity file the timestamp in order to avoid
    // conflicts between multiple applgrid files generated with this
    // code.
    time_t rawtime;
    struct tm * timeinfo;
    const int TZ = 16;
    char t[TZ];  
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(t, TZ, "%Y%m%d%H%M%S", timeinfo);
    std::string filename = "amcatnlo_obs_" + ss.str() + "_" + std::string(t) + ".config";
    new lumi_pdf(filename,pdf_luminosities);

    // Define binning
    int Nbins     = appl_common_histokin_.obs_nbins;
    double obsmin = appl_common_histokin_.obs_min;
    double obsmax = appl_common_histokin_.obs_max;

    // Create array with the bin edges
    std::vector<double> obsbins(Nbins+1);
    for(int i=0; i<=Nbins; i++) obsbins[i] = appl_common_histokin_.obs_bins[i];

    // Check if the actual lower and upper limits of the histogram are correct
    if(fabs(obsbins[0]-obsmin) >= 1e-5) {
      std::cout << "amcblast ERROR: mismatch in the lower limit of the histogram:" << std::endl;
      std::cout << "It is " << obsbins[0] << ", it should be " << obsmin << std::endl;
      exit(-10);
    } 
    if(fabs(obsbins[Nbins]-obsmax) >= 1e-5) {
      std::cout << "amcblast ERROR: mismatch in the upper limit of the histogram" << std::endl;
      std::cout << "It is " << obsbins[Nbins] << ", it should be " << obsmax << std::endl;
      exit(-10);
    }
    // Create a grid with the binning given in the "obsbins[Nbins+1]" array
    grid_obs.emplace_back(Nbins,    obsbins.data(),
                                      NQ2,      Q2min,         Q2max, Q2order,  
				      Nx,       xmin,          xmax,  xorder,
				      filename, order_ids);
    // Use the reweighting function
    grid_obs[grid_obs.size()-1].reweight(true);
    // Add documentation
    grid_obs[grid_obs.size()-1].addDocumentation(Banner());
  }

  // Compute all the bin widths of the h-th histogram
  std::vector<double> hbins;
  for(int i=0; i<appl_common_histokin_.obs_nbins; i++) hbins.push_back(appl_common_histokin_.obs_bins[i+1]-appl_common_histokin_.obs_bins[i]);
  binwidths.push_back(hbins);
}

extern "C" void appl_fill_() {
  // Check event weight reconstruction
  //reco();
  // Check that itype ranges from 1 to 3.
  int itype = appl_common_histokin_.itype_histo;
  if((itype < 1) || (itype > 5)) {
    std::cout << "amcblast ERROR: Invalid value of itype = " << itype << std::endl;
    exit(-10);
  }

  //this is the second index of the WB/R/F/0 arrays
  int index = appl_common_histokin_.amp_pos - 1;

  // aMC@NLO weights. Four grids, ordered as {W0,WR,WF,WB}.
  double (& W0) [4] = appl_common_weights_.W0[index];
  double (& WR) [4] = appl_common_weights_.WR[index];
  double (& WF) [4] = appl_common_weights_.WF[index];
  double (& WB) [4] = appl_common_weights_.WB[index];

  int ilumi;
  int nlumi = appl_common_lumi_.nlumi;
  double ttol = 1e-100;
  double x1,x2;
  double scale2;
  double obs = appl_common_histokin_.obs_histo;
  // Weight vector whose size is the total number of subprocesses
  std::vector<double> weight(nlumi,0);

  // Histogram number
  int nh = appl_common_histokin_.obs_num - 1;

  // translate (index,nh) -> index of the APPLgrid
  int const grid_index = translation_tables.at(nh).at(index);

  // (n+1)-body contribution (corresponding to xsec11 in aMC@NLO)
  // It uses only Events (k=0) and the W0 weight.
  if(itype == 1) {
    // Get Bjorken x's
    x1     = appl_common_weights_.x1[0];
    x2     = appl_common_weights_.x2[0];
    static std::vector<std::vector<double>> x1Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    static std::vector<std::vector<double>> x2Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    if(x1 == x1Saved[nh][grid_index] && x2 == x2Saved[nh][grid_index])
      return;
    else
    {
      x1Saved[nh][grid_index] = x1;
      x2Saved[nh][grid_index] = x2;
    }
    // Energy scale
    scale2 = appl_common_weights_.muF2[0];
    // Relevant parton luminosity combination (-1 offset in c++)
    ilumi  = appl_common_weights_.flavmap[0] - 1;
    if(x1 < 0 || x1 > 1 || x2 < 0 || x2 > 1) {
      std::cout << "amcblast ERROR: Invalid value of x1 and/or x2 = " << x1 << " " << x2 << std::endl;
      exit(-10);
    }

    // Fill grid only if x1 and x1 are non-zero
    if(x1 == 0 && x2 == 0) return;

    // Fill only if W0 is non zero
    if(fabs(W0[0]) < ttol) return;

    // Fill the grid with the values of the observables
    // W0
    weight.at(ilumi) = W0[0];
    grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+0);
    weight.at(ilumi) = 0;
  }
  // n-body contribution without Born (corresponding to xsec12 in aMC@NLO)
  // Soft CounterEvents (k=1) and uses all weights W0, WR and WF.
  else if(itype == 2) {
    int k=1;
    x1     = appl_common_weights_.x1[k];
    x2     = appl_common_weights_.x2[k];
    static std::vector<std::vector<double>> x1Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    static std::vector<std::vector<double>> x2Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    if(x1 == x1Saved[nh][grid_index] && x2 == x2Saved[nh][grid_index])
      return;
    else
    {
      x1Saved[nh][grid_index] = x1;
      x2Saved[nh][grid_index] = x2;
    }
    {
      scale2 = appl_common_weights_.muF2[k];
      ilumi  = appl_common_weights_.flavmap[k] - 1;

      if(x1 < 0 || x1 > 1 || x2 < 0 || x2 > 1) {
	std::cout << "amcblast ERROR: Invalid value of x1 and/or x2 = " << x1 << " " << x2 << std::endl;
	exit(-10);
      }
      if(x1 == 0 && x2 == 0) return;

      if(fabs(W0[k]) < ttol && fabs(WR[k]) < ttol && fabs(WF[k]) < ttol) return;

      // W0
      weight.at(ilumi) = W0[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+0);
      weight.at(ilumi) = 0;
      // WR
      weight.at(ilumi) = WR[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+1);
      weight.at(ilumi) = 0;
      // WF
      weight.at(ilumi) = WF[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+2);
      weight.at(ilumi) = 0;
    }
  }
  // Born (n-body) contribution (corresponding to xsec20 in aMC@NLO)
  // It uses only the soft kinematics (k=1) and the weight WB.
  else if(itype == 3) {
    x1     = appl_common_weights_.x1[1];
    x2     = appl_common_weights_.x2[1];
    static std::vector<std::vector<double>> x1Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    static std::vector<std::vector<double>> x2Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    if(x1 == x1Saved[nh][grid_index] && x2 == x2Saved[nh][grid_index])
      return;
    else
    {
      x1Saved[nh][grid_index] = x1;
      x2Saved[nh][grid_index] = x2;
    }
    scale2 = appl_common_weights_.muF2[1];
    ilumi  = appl_common_weights_.flavmap[1] - 1;

    if(x1 < 0 || x1 > 1 || x2 < 0 || x2 > 1) {
      std::cout << "amcblast ERROR: Invalid value of x1 and/or x2 = " << x1 << " " << x2 << std::endl;
      exit(-10);
    }

    if(x1 == 0 && x2 == 0) return;

    if(fabs(WB[1]) < ttol) return;

    // WB
    weight.at(ilumi) = WB[1];
    grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index);
    weight.at(ilumi) = 0;
  }
  // n-body contribution without Born (corresponding to xsec12 in aMC@NLO)
  // Collinear CounterEvents (k=2) and uses all weights W0, WR and WF.
  else if(itype == 4) {
    int k=2;
    x1     = appl_common_weights_.x1[k];
    x2     = appl_common_weights_.x2[k];
    static std::vector<std::vector<double>> x1Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    static std::vector<std::vector<double>> x2Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    if(x1 == x1Saved[nh][grid_index] && x2 == x2Saved[nh][grid_index])
      return;
    else
    {
      x1Saved[nh][grid_index] = x1;
      x2Saved[nh][grid_index] = x2;
    }
    {
      scale2 = appl_common_weights_.muF2[k];
      ilumi  = appl_common_weights_.flavmap[k] - 1;

      if(x1 < 0 || x1 > 1 || x2 < 0 || x2 > 1) {
	std::cout << "amcblast ERROR: Invalid value of x1 and/or x2 = " << x1 << " " << x2 << std::endl;
	exit(-10);
      }
      if(x1 == 0 && x2 == 0) return;

      if(fabs(W0[k]) < ttol && fabs(WR[k]) < ttol && fabs(WF[k]) < ttol) return;

      // W0
      weight.at(ilumi) = W0[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+0);
      weight.at(ilumi) = 0;
      // WR
      weight.at(ilumi) = WR[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+1);
      weight.at(ilumi) = 0;
      // WF
      weight.at(ilumi) = WF[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+2);
      weight.at(ilumi) = 0;
    }
  }
  // n-body contribution without Born (corresponding to xsec12 in aMC@NLO)
  // Soft-Collinear CounterEvents (k=2) and uses all weights W0, WR and WF.
  else if(itype == 5) {
    int k=3;
    x1     = appl_common_weights_.x1[k];
    x2     = appl_common_weights_.x2[k];
    static std::vector<std::vector<double>> x1Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    static std::vector<std::vector<double>> x2Saved(grid_obs.size(), std::vector<double>(grid_obs[nh].order_ids().size(), 0.0));
    {
      scale2 = appl_common_weights_.muF2[k];
      ilumi  = appl_common_weights_.flavmap[k] - 1;

      if(x1 < 0 || x1 > 1 || x2 < 0 || x2 > 1) {
	std::cout << "amcblast ERROR: Invalid value of x1 and/or x2 = " << x1 << " " << x2 << std::endl;
	exit(-10);
      }
      if(x1 == 0 && x2 == 0) return;

      if(fabs(W0[k]) < ttol && fabs(WR[k]) < ttol && fabs(WF[k]) < ttol) return;

      // W0
      weight.at(ilumi) = W0[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+0);
      weight.at(ilumi) = 0;
      // WR
      weight.at(ilumi) = WR[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+1);
      weight.at(ilumi) = 0;
      // WF
      weight.at(ilumi) = WF[k];
      grid_obs[nh].fill_grid(x1,x2,scale2,obs,&weight[0],grid_index+2);
      weight.at(ilumi) = 0;
    }
  }
}

extern "C" void appl_fill_ref_() {
   // Event weights to fill the histograms
  double www = appl_common_histokin_.www_histo;

  // Physical observables
  double obs = appl_common_histokin_.obs_histo;

  // Histogram number
  int nh = appl_common_histokin_.obs_num - 1;

  grid_obs[nh].getReference()->Fill(obs,www);
}

extern "C" void appl_fill_ref_out_() {
  // Normalization factor
  double norm = appl_common_histokin_.norm_histo;

  // Check normalization value
  if(norm <= 0.0 || norm > 1e50) {
    std::cout << "amcblast ERROR: Invalid value for histogram normalization = " << norm << std::endl;
    exit(-10);
  }

  // Histogram number
  int nh = appl_common_histokin_.obs_num - 1;

  // Apply normalization
  grid_obs[nh].getReference()->Scale(norm);

  // Rescale the reference histogram bins by the respective width
  for(unsigned i=0; i<binwidths[nh].size(); i++) {
    double bin = grid_obs[nh].getReference()->GetBinContent(i+1) / binwidths[nh][i];
    grid_obs[nh].getReference()->SetBinContent(i+1,bin); // Reference histogram doesn't get the bin corrections
  }
}

extern "C" void appl_term_() {
  // Conversion factor from natural units to pb
  double conv = 389379660;

  // Normalization factor
  double norm   = appl_common_histokin_.norm_histo;
  double n_runs = 1 / norm;

  // Histogram number
  int nh = appl_common_histokin_.obs_num - 1;

  // Construct the output file name according to its position in the vector "grid_obs"
  std::ostringstream ss;
  ss << nh;
  grid_filename_out = "grid_obs_" + ss.str() + "_out.root";

  // Normalize the grid by conversion factor and number of runs 
  grid_obs[nh] *= conv / n_runs;
  grid_obs[nh].getReference()->Scale(n_runs/conv); // Normalize the reference histogram back

  // Set run() to one for the combinantion.  
  grid_obs[nh].run() = 1;

  // Write grid to file
  grid_obs[nh].Write(grid_filename_out); 
}
