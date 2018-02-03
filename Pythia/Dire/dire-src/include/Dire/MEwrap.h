#include <iostream> 
#include <sstream> 
#include <iomanip> 
#include <vector> 
#include <set> 

#include "Pythia8/Pythia.h"
//#include "DirePlugins/MyCollectionOfSMProcesses/Processes_sm/Parameters_sm.h"
//#include "DirePlugins/MyCollectionOfSMProcesses/Processes_sm/PY8ME.h"
//#include "DirePlugins/MyCollectionOfSMProcesses/Processes_sm/PY8MEs.h"
//#include "Processes_sm/Parameters_sm.h"
//#include "Processes_sm/PY8ME.h"
//#include "Processes_sm/PY8MEs.h"

#ifdef MG5MES
#include "Processes_sm/Parameters_sm.h"
#include "Processes_sm/PY8ME.h"
#include "Processes_sm/PY8MEs.h"
#endif

typedef std::vector<double> vec_double; 

#ifdef MG5MES
bool isAvailableME(PY8MEs_namespace::PY8MEs& accessor, vector <int> in,
   vector<int> out);
bool isAvailableME(PY8MEs_namespace::PY8MEs& accessor,
   const Pythia8::Event& event);
double calcME(PY8MEs_namespace::PY8MEs& accessor,
   const Pythia8::Event& event);
#else
bool isAvailableME();
double calcME();
#endif

