#include <iostream> 
#include <sstream> 
#include <iomanip> 
#include <vector> 
#include <set> 

#include "PY8ME.h"
#include "PY8MEs.h"
#include "rambo.h"

using namespace std; 

typedef vector<double> vec_double; 

// Nice string to display a process
string proc_string(vector<int> in_pdgs, vector<int> out_pdgs, set<int> 
req_s_channels = set<int> ())
{

  std::stringstream ss; 

  for (int i = 0; i < in_pdgs.size(); i++ )
    ss << in_pdgs[i] <<  " "; 
  if (req_s_channels.size() > 0)
  {
    ss <<  "> "; 
    for (set<int> ::iterator it = req_s_channels.begin(); it != 
      req_s_channels.end(); ++ it)
    ss << * it <<  " "; 
  }
  ss <<  "> "; 
  for (int i = 0; i < out_pdgs.size(); i++ )
  {
    ss << out_pdgs[i]; 
    if (i != (out_pdgs.size() - 1))
      ss <<  " "; 
  }

  return ss.str(); 
}

// Evaluate a given process with an accessor
void run_proc(PY8MEs_namespace::PY8MEs& accessor, vector<int> in_pdgs,
    vector<int> out_pdgs, set<int> req_s_channels = set<int> ())
{

  // This is not mandatory, we run it here only because we need an instance of
  // the process
  // to obtain the external masses to generate the PS point.
  PY8MEs_namespace::PY8ME * query = accessor.getProcess(in_pdgs, out_pdgs,
      req_s_channels);

  cout <<  " -> Process '"; 
  cout << proc_string(in_pdgs, out_pdgs, req_s_channels) <<  "'"; 
  if (query)
  {
    cout <<  " is available." << endl; 
  }
  else
  {
    cout <<  " is not available." << endl; 
    return; 
  }

  double energy = 1500; 
  double weight; 

  vector < vec_double > p_vec = vector < vec_double > (in_pdgs.size() +
      out_pdgs.size(), vec_double(4, 0.));

  //----
  // The line below 'get_momenta' that uses RAMBO has memory leaks. It is fine
  // for testing/debugging.
  // But replace with an hard-coded momentum configuration if you want to
  // memory-check the ME code.
  //----
  // Get phase space point
  vector < double * > p = get_momenta(in_pdgs.size(), energy,
      query->getMasses(), weight);
  // Cast to a sane type
  for (int i = 0; i < in_pdgs.size() + out_pdgs.size(); i++ )
  {
    for (int j = 0; j < 4; j++ )
    {
      p_vec[i][j] = p[i][j]; 
      // Copy-paste the printout below in this code to use an hard-coded
      // momentum configuration
      // cout<<setiosflags(ios::scientific) << setprecision(14) <<
      // setw(22)<<"p_vec["<<i<<"]["<<j<<"]"<<"="<<p_vec[i][j]<<";"<<endl;
    }
  }
  // Release arrays
  for(int i = 0; i < (in_pdgs.size() + out_pdgs.size()); i++ )
  {
    delete[] p[i]; 
    p[i] = NULL; 
  }
  //---
  // End of the segment that can cause memory leak.
  //---

  // You could require specific helicity and quantum numbers here
  // (these arguments, like req_s_channels are optional for calculateME and
  // considerd empty by default)
  // They are left empty here, meaning that these quantum numbers will be
  // summed/averaged over.
  // This vector's size would be twice the number of external legs (for color
  // and anti-color specification)
  vector<int> colors(0); 
  // This vector's size would be the number of external legs
  vector<int> helicities(0); 
  pair < double, bool > res = accessor.calculateME(in_pdgs, out_pdgs, p_vec,
      req_s_channels, colors, helicities);

  if ( !res.second)
  {
    cout <<  " | Its evaluation failed." << endl; 
    return; 
  }
  else
  {
    cout <<  " | Momenta:" << endl; 
    for(int i = 0; i < (in_pdgs.size() + out_pdgs.size()); i++ )
      cout <<  " | " << setw(4) << i + 1
     << setiosflags(ios::scientific) << setprecision(14) << setw(22) <<
        p_vec[i][0]
     << setiosflags(ios::scientific) << setprecision(14) << setw(22) <<
        p_vec[i][1]
     << setiosflags(ios::scientific) << setprecision(14) << setw(22) <<
        p_vec[i][2]
     << setiosflags(ios::scientific) << setprecision(14) << setw(22) <<
        p_vec[i][3] << endl;
    cout <<  " | Matrix element : " << setiosflags(ios::scientific) <<
        setprecision(17) << res.first;
    cout <<  " GeV^" << - (2 * (in_pdgs.size() + out_pdgs.size()) - 8) << endl; 
  }
}

int main(int argc, char * * argv)
{

  // Simplest way of creating a PY8MEs accessor
  PY8MEs_namespace::PY8MEs PY8MEs_accessor("param_card_sm.dat"); 

  //----------------------------------------------------------------------------
  // Here is an alternative way of creating a PY8MEs_accessor for which we
  // handle ourselves the instantiation, release and initialization of the
  // model. Notice that we need here the name of the model class because it
  // does not have a generic base class (I can add it if really necessary)
  // 
  // Parameters_sm* model = PY8MEs_namespace::PY8MEs::instantiateModel();
  // 
  // With here an example of the initialization of the model using
  // Pythia8 objects.
  // 
  // model->setIndependentParameters(particleDataPtr,couplingsPtr,slhaPtr);
  // model->setIndependentCouplings();
  // model->printIndependentParameters();
  // model->printIndependentCouplings();
  // 
  // And then finally obtain the accessor with this particular model
  // 
  // PY8MEs_namespace::PY8MEs PY8MEs_accessor(model);
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // Finally one last way of creating a PY8MEs_accessor, which does the same
  // as above but this time doesn't require to define a local pointer to the
  // model (and hence to know its class name):
  // 
  // PY8MEs_namespace::PY8MEs PY8MEs_accessor();
  // 
  // We could now initialize the model from PY8 directly using the accessor
  // without having to manipulate a local pointer of the model
  // 
  // PY8MEs_namespace.initModelWithPY8(particleDataPtr, couplingsPtr, slhaPtr);
  // 
  // If needed, one can still access an instance of the model (for example
  // to be used for instantiating another PY8MEs_accessor) as follows. Be
  // wary however that as soon as you call this accessor, the PY8MEs_accessor
  // destructor will no longer take care of releasing the model instance and
  // it will be your responsability to do so.
  // 
  // Parameters_sm* model = PY8MEs_accessor.getModel();
  //----------------------------------------------------------------------------

  vector<int> vec_in_pdgs; 
  vector<int> vec_out_pdgs; 
  set<int> set_req_s_channels; 

  // Test the existence of a non-available process
  cout << endl <<  "Testing the non-existence of a non-available process:" <<
      endl;

  const int in_pdgs[2] = {33, 43}; 
  const int out_pdgs[4] = {33, 2, 1, 5}; 
  const int req_s_channels[2] = {2, 5}; 
  vec_in_pdgs = vector<int> (in_pdgs, in_pdgs + 2); 
  vec_out_pdgs = vector<int> (out_pdgs, out_pdgs + 4); 
  set_req_s_channels = set<int> (req_s_channels, req_s_channels + 2); 

  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 

  // Testing available processes
  cout << endl <<  "Testing the evaluation of available processes:" << endl; 

  // Process Process: g u > z u WEIGHTED<=3 @2
  const int in_pdgs0[2] = {21, 2}; 
  const int out_pdgs0[2] = {23, 2}; 
  const int req_s_channels0[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs0, in_pdgs0 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs0, out_pdgs0 + 2); 
  set_req_s_channels = set<int> (req_s_channels0, req_s_channels0 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: g c > z c WEIGHTED<=3 @2
  const int in_pdgs1[2] = {21, 4}; 
  const int out_pdgs1[2] = {23, 4}; 
  const int req_s_channels1[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs1, in_pdgs1 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs1, out_pdgs1 + 2); 
  set_req_s_channels = set<int> (req_s_channels1, req_s_channels1 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: g d > z d WEIGHTED<=3 @2
  const int in_pdgs2[2] = {21, 1}; 
  const int out_pdgs2[2] = {23, 1}; 
  const int req_s_channels2[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs2, in_pdgs2 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs2, out_pdgs2 + 2); 
  set_req_s_channels = set<int> (req_s_channels2, req_s_channels2 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: g s > z s WEIGHTED<=3 @2
  const int in_pdgs3[2] = {21, 3}; 
  const int out_pdgs3[2] = {23, 3}; 
  const int req_s_channels3[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs3, in_pdgs3 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs3, out_pdgs3 + 2); 
  set_req_s_channels = set<int> (req_s_channels3, req_s_channels3 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: g u~ > z u~ WEIGHTED<=3 @2
  const int in_pdgs4[2] = {21, -2}; 
  const int out_pdgs4[2] = {23, -2}; 
  const int req_s_channels4[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs4, in_pdgs4 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs4, out_pdgs4 + 2); 
  set_req_s_channels = set<int> (req_s_channels4, req_s_channels4 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: g c~ > z c~ WEIGHTED<=3 @2
  const int in_pdgs5[2] = {21, -4}; 
  const int out_pdgs5[2] = {23, -4}; 
  const int req_s_channels5[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs5, in_pdgs5 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs5, out_pdgs5 + 2); 
  set_req_s_channels = set<int> (req_s_channels5, req_s_channels5 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: g d~ > z d~ WEIGHTED<=3 @2
  const int in_pdgs6[2] = {21, -1}; 
  const int out_pdgs6[2] = {23, -1}; 
  const int req_s_channels6[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs6, in_pdgs6 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs6, out_pdgs6 + 2); 
  set_req_s_channels = set<int> (req_s_channels6, req_s_channels6 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: g s~ > z s~ WEIGHTED<=3 @2
  const int in_pdgs7[2] = {21, -3}; 
  const int out_pdgs7[2] = {23, -3}; 
  const int req_s_channels7[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs7, in_pdgs7 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs7, out_pdgs7 + 2); 
  set_req_s_channels = set<int> (req_s_channels7, req_s_channels7 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: u u~ > z g WEIGHTED<=3 @2
  const int in_pdgs8[2] = {2, -2}; 
  const int out_pdgs8[2] = {23, 21}; 
  const int req_s_channels8[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs8, in_pdgs8 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs8, out_pdgs8 + 2); 
  set_req_s_channels = set<int> (req_s_channels8, req_s_channels8 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: c c~ > z g WEIGHTED<=3 @2
  const int in_pdgs9[2] = {4, -4}; 
  const int out_pdgs9[2] = {23, 21}; 
  const int req_s_channels9[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs9, in_pdgs9 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs9, out_pdgs9 + 2); 
  set_req_s_channels = set<int> (req_s_channels9, req_s_channels9 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: d d~ > z g WEIGHTED<=3 @2
  const int in_pdgs10[2] = {1, -1}; 
  const int out_pdgs10[2] = {23, 21}; 
  const int req_s_channels10[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs10, in_pdgs10 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs10, out_pdgs10 + 2); 
  set_req_s_channels = set<int> (req_s_channels10, req_s_channels10 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: s s~ > z g WEIGHTED<=3 @2
  const int in_pdgs11[2] = {3, -3}; 
  const int out_pdgs11[2] = {23, 21}; 
  const int req_s_channels11[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs11, in_pdgs11 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs11, out_pdgs11 + 2); 
  set_req_s_channels = set<int> (req_s_channels11, req_s_channels11 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: u u~ > z WEIGHTED<=2 @1
  const int in_pdgs12[2] = {2, -2}; 
  const int out_pdgs12[1] = {23}; 
  const int req_s_channels12[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs12, in_pdgs12 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs12, out_pdgs12 + 1); 
  set_req_s_channels = set<int> (req_s_channels12, req_s_channels12 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: c c~ > z WEIGHTED<=2 @1
  const int in_pdgs13[2] = {4, -4}; 
  const int out_pdgs13[1] = {23}; 
  const int req_s_channels13[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs13, in_pdgs13 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs13, out_pdgs13 + 1); 
  set_req_s_channels = set<int> (req_s_channels13, req_s_channels13 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: d d~ > z WEIGHTED<=2 @1
  const int in_pdgs14[2] = {1, -1}; 
  const int out_pdgs14[1] = {23}; 
  const int req_s_channels14[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs14, in_pdgs14 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs14, out_pdgs14 + 1); 
  set_req_s_channels = set<int> (req_s_channels14, req_s_channels14 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 
  // Process Process: s s~ > z WEIGHTED<=2 @1
  const int in_pdgs15[2] = {3, -3}; 
  const int out_pdgs15[1] = {23}; 
  const int req_s_channels15[0] = {}; 
  vec_in_pdgs = vector<int> (in_pdgs15, in_pdgs15 + 2); 
  vec_out_pdgs = vector<int> (out_pdgs15, out_pdgs15 + 1); 
  set_req_s_channels = set<int> (req_s_channels15, req_s_channels15 + 0); 
  run_proc(PY8MEs_accessor, vec_in_pdgs, vec_out_pdgs, set_req_s_channels); 


  #include "check_sa_additional_runs.inc"

}
