#include <iostream> 
#include <sstream> 
#include <iomanip> 
#include <vector> 
#include <set> 

#include "Pythia8/Pythia.h"
#include "DirePlugins/MyCollectionOfSMProcesses/Processes_sm/Parameters_sm.h"
#include "DirePlugins/MyCollectionOfSMProcesses/Processes_sm/PY8ME.h"
#include "DirePlugins/MyCollectionOfSMProcesses/Processes_sm/PY8MEs.h"

typedef vector<double> vec_double; 

void fill_ID_vec(const Pythia8::Event& event, vector<int>& in, vector<int>& out);
void fill_4V_vec(const Pythia8::Event& event, vector<Pythia8::Vec4>& p);
void fill_COL_vec(const Pythia8::Event& event, vector<int>& colors);

//bool isAvailableME(PY8MEs_namespace::PY8MEs& accessor, vector<int> in_pdgs,
//    vector<int> out_pdgs, set<int> req_s_channels = set<int> ());

bool isAvailableME(PY8MEs_namespace::PY8MEs& accessor, const Pythia8::Event& event);
double calcME(PY8MEs_namespace::PY8MEs& accessor, const Pythia8::Event& event);

//bool isAvailableME(PY8MEs_sm::PY8MEs& accessor, vector<int> in_pdgs,
//    vector<int> out_pdgs, set<int> req_s_channels = set<int> ());
//
//bool isAvailableME(PY8MEs_sm::PY8MEs& accessor, const Pythia8::Event& event);
//double calcME(PY8MEs_sm::PY8MEs& accessor, const Pythia8::Event& event);

