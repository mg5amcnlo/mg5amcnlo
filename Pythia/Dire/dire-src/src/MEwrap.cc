
#include "Dire/MEwrap.h"

using namespace Pythia8;

void fill_ID_vec(const Pythia8::Event& event, vector<int>& in,
  vector<int>& out) {
  in.push_back(event[3].id());
  in.push_back(event[4].id());
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() ) out.push_back(event[i].id());
  }
}

void fill_4V_vec(const Pythia8::Event& event, vector<Pythia8::Vec4>& p) {
  p.push_back(event[3].p());
  p.push_back(event[4].p());
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() ) p.push_back(event[i].p());
  }
}

void fill_COL_vec(const Pythia8::Event& event, vector<int>& colors) {
  colors.push_back(event[3].col()); colors.push_back(event[3].acol());
  colors.push_back(event[4].col()); colors.push_back(event[4].acol());
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() ) {
      colors.push_back(event[i].col());
      colors.push_back(event[i].acol());
    }
  }
}

vector < vec_double > getMG5MomVec ( vector<Pythia8::Vec4> p ) {
  vector < vec_double > ret;
  for (int i = 0; i < int(p.size()); i++ ) {
    vec_double p_tmp(4, 0.);
    p_tmp[0] = abs(p[i].e())  > 1e-10 ? p[i].e() : 0.0;
    p_tmp[1] = abs(p[i].px()) > 1e-10 ? p[i].px() : 0.0;
    p_tmp[2] = abs(p[i].py()) > 1e-10 ? p[i].py() : 0.0;
    p_tmp[3] = abs(p[i].pz()) > 1e-10 ? p[i].pz() : 0.0;
    ret.push_back(p_tmp);
  }
  return ret;
}

vector < vec_double > fill_MG5Mom_vec ( const Pythia8::Event event) {
  vector<Pythia8::Vec4> p;
  fill_4V_vec(event,p);
  return  getMG5MomVec(p);
}

#ifdef MG5MES
bool isAvailableME(PY8MEs_namespace::PY8MEs& accessor,
  vector <int> in, vector<int> out) {
  set<int> req_s_channels; 
  PY8MEs_namespace::PY8ME * query
    = accessor.getProcess(in, out, req_s_channels);
  return (query != 0);
}

bool isAvailableME(PY8MEs_namespace::PY8MEs& accessor,
  const Pythia8::Event& event) {
  vector <int> in, out;
  fill_ID_vec(event, in, out);
  set<int> req_s_channels; 
  PY8MEs_namespace::PY8ME * query
    = accessor.getProcess(in, out, req_s_channels);
  return (query != 0);
}

// Evaluate a given process with an accessor
double calcME( PY8MEs_namespace::PY8MEs& accessor,
  const Pythia8::Event& event) {
  vector <int> in, out;
  fill_ID_vec(event, in, out);
  vector<int> cols; 
  fill_COL_vec(event, cols);
  vector< vec_double > pvec = fill_MG5Mom_vec( event);
  set<int> req_s_channels; 
  vector<int> helicities; 

  // Redirect output so that exceptions can be printed by Dire.
  std::streambuf *old = cerr.rdbuf();
  stringstream ss;
  cerr.rdbuf (ss.rdbuf());
  bool success = true;
  pair < double, bool > res;
  try {
    res = accessor.calculateME(in, out, pvec, req_s_channels, cols, helicities);
  } catch (const std::exception& e) {
    success = false;
    cout << "Caught exception in " << __PRETTY_FUNCTION__ << ": "
         << ss << endl;
  }
  // Restore print-out.
  cerr.rdbuf (old);
  if (!success) return 0.0;

  if (res.second) {
    double me = res.first;
    PY8MEs_namespace::PY8ME * query
      = accessor.getProcess(in, out, req_s_channels);
    me *= 1./query->getHelicityAveragingFactor();
    // no symmetry factors me *= query->getSymmetryFactor();
    me *= 1./query->getColorAveragingFactor();
    return me;
  }
  // Done
  return 0.0;
}
#else
bool isAvailableME() { return false; }
double calcME() { return false; }
#endif
