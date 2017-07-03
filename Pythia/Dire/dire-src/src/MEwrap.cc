
#include "Dire/MEwrap.h"

using namespace Pythia8;

void fill_ID_vec(const Pythia8::Event& event, vector<int>& in, vector<int>& out) {
  in.push_back(event[3].id());
  in.push_back(event[4].id());
  for (int i = 4; i < event.size(); ++i) {
//    if ( event[i].id() == 21 ) continue;
//    if ( event[i].idAbs() < 6 ) continue;
    if ( event[i].isFinal() ) out.push_back(event[i].id());
  }
/*  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].id() != 21 ) continue;
    if ( event[i].isFinal() ) out.push_back(event[i].id());
  }
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].idAbs() > 5 ) continue;
    if ( event[i].isFinal() ) out.push_back(event[i].id());
  }
*/
}

void fill_4V_vec(const Pythia8::Event& event, vector<Pythia8::Vec4>& p) {
  p.push_back(event[3].p());
  p.push_back(event[4].p());
  for (int i = 4; i < event.size(); ++i) {
//    if ( event[i].id() == 21 ) continue;
//    if ( event[i].idAbs() < 6 ) continue;
    if ( event[i].isFinal() ) p.push_back(event[i].p());
  }
/*  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].id() != 21 ) continue;
    if ( event[i].isFinal() ) p.push_back(event[i].p());
  }
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].idAbs() > 5 ) continue;
    if ( event[i].isFinal() ) p.push_back(event[i].p());
  }*/
}

void fill_COL_vec(const Pythia8::Event& event, vector<int>& colors) {

  colors.push_back(event[3].col()); colors.push_back(event[3].acol());
  colors.push_back(event[4].col()); colors.push_back(event[4].acol());
  for (int i = 4; i < event.size(); ++i) {
//    if ( event[i].id() == 21 ) continue;
//    if ( event[i].idAbs() < 6 ) continue;
    if ( event[i].isFinal() ) {
      colors.push_back(event[i].col());
      colors.push_back(event[i].acol());
    }
  }
/*  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].id() != 21 ) continue;
    if ( event[i].isFinal() ) {
      colors.push_back(event[i].col());
      colors.push_back(event[i].acol());
    }
  }
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].idAbs() > 5 ) continue;
    if ( event[i].isFinal() ) {
      colors.push_back(event[i].col());
      colors.push_back(event[i].acol());
    }
  }*/
}

vector<pair<int,int> > fillColPairs(const Pythia8::Event& event) {
  vector<pair<int,int> > ret;
  ret.push_back( make_pair(event[3].col(), event[3].acol()));
  ret.push_back( make_pair(event[4].col(), event[4].acol()));
  for (int i = 4; i < event.size(); ++i)
    if ( event[i].isFinal() )
      ret.push_back( make_pair(event[i].col(), event[i].acol()));
  return ret;
}

vector<int> fillColVec( vector< pair<int,int> > pairs) {
  vector<int> ret;
  for (unsigned int i = 0; i < pairs.size(); ++i) {
    ret.push_back(pairs[i].first);
    ret.push_back(pairs[i].second);
  }
  return ret;
}

pair<vector<Pythia8::Vec4>, vector<Pythia8::Vec4> > fillMomVec(const Pythia8::Event& event) {
  vector<Pythia8::Vec4> pi, pf;
  pi.push_back(event[3].p());
  pi.push_back(event[4].p());
  for (int i = 4; i < event.size(); ++i) {
    if ( event[i].isFinal() ) pf.push_back(event[i].p());
  }
  return make_pair(pi,pf);
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

bool isAvailableME(PY8MEs_namespace::PY8MEs& accessor,
  const Pythia8::Event& event) {

  vector <int> in_pdgs, out_pdgs;
  fill_ID_vec(event, in_pdgs, out_pdgs);
  vector<pair<int,int> > pairs(fillColPairs(event));
  vector<pair<int,int> > in_pairs(pairs.begin(), pairs.begin()+2);
  vector<pair<int,int> > fi_pairs(pairs.begin()+2, pairs.end());
  vector<int> cols = fillColVec(pairs);
  set<int> req_s_channels; 
  PY8MEs_namespace::PY8ME * query = 0;

//cout << "enter " << __LINE__  << endl;
  int iperm(0), colID(-2);
  do {
    do {
      iperm++;
//      cout << "permutation=" << iperm<< endl;

      vector<pair<int,int> > new_pairs(in_pairs.begin(), in_pairs.end());
      new_pairs.insert(new_pairs.end(), fi_pairs.begin(), fi_pairs.end());
      cols = fillColVec(new_pairs);

//      for (int i=0; i < cols.size(); ++i)
//        cout << "\t" << cols[i];
//      cout << endl;

      query = accessor.getProcess(in_pdgs, out_pdgs, req_s_channels);
      if (query != 0) {
        //vector<pair<int,int> > new_pairs(in_pairs.begin(), in_pairs.end());
        //new_pairs.insert(new_pairs.end(), fi_pairs.begin(), fi_pairs.end());
        //cols = fillColVec(new_pairs);
        query->setColors(cols);
        colID = query->getColorIDForConfig(cols);
        if ( colID != -2) break;
      }
      if ( colID != -2) break;

/*      if ( query != 0 && colID == -2) {

      for (int i=0; i < in_pdgs.size(); ++i)
        cout << "\t" << in_pdgs[i];
      for (int i=0; i < out_pdgs.size(); ++i)
        cout << "\t" << out_pdgs[i];
      cout << endl;

      for (int i=0; i < cols.size(); ++i)
        cout << "\t" << cols[i];
      cout << endl;
//      abort();
      }
*/
//      cout << "Warning in "<<  __PRETTY_FUNCTION__ << ": Try next permutation of final states" << endl;
      std::next_permutation(out_pdgs.begin(),out_pdgs.end());
    } while(std::next_permutation(fi_pairs.begin(),fi_pairs.end()));
//    if ( colID != -2) { cout << "found!" << endl; break;}
//    if ( colID != -2) break;
//    cout << "Warning in "<<  __PRETTY_FUNCTION__ << ": Try next permutation of initial states" << endl;
    std::next_permutation(in_pdgs.begin(),in_pdgs.end());
  } while(std::next_permutation(in_pairs.begin(),in_pairs.end()));

if (query != 0 && colID == -2) {cout << "fail " << endl; abort();}
//cout << "done " << __LINE__ << " --> " << (query != 0) << endl;

  return (query != 0 && colID != -2);

}

// Evaluate a given process with an accessor
double calcME(PY8MEs_namespace::PY8MEs& accessor, const Pythia8::Event& event) {

  vector <int> in_pdgs, out_pdgs;
  fill_ID_vec(event, in_pdgs, out_pdgs);
  set<int> req_s_channels; 

  pair<vector<Pythia8::Vec4>, vector<Pythia8::Vec4> > ps(fillMomVec(event));
  vector < vec_double > pi_mg5( getMG5MomVec(ps.first));
  vector < vec_double > pf_mg5( getMG5MomVec(ps.second));
  vector< vec_double > pvec2;
  pvec2.insert(pvec2.end(), pi_mg5.begin(), pi_mg5.end());
  pvec2.insert(pvec2.end(), pf_mg5.begin(), pf_mg5.end());
  
  /*for (int i = 0; i < int(p_vec.size()); i++ ) {
    cout <<  p_vec[i][0] << "\t" <<  p_vec[i][1] << "\t" <<  p_vec[i][2] << "\t" <<  p_vec[i][3] << endl;
    cout <<  pvec[i][0] << "\t" <<  pvec[i][1] << "\t" <<  pvec[i][2] << "\t" <<  pvec[i][3] << endl;
    cout <<  pvec2[i][0] << "\t" <<  pvec2[i][1] << "\t" <<  pvec2[i][2] << "\t" <<  pvec2[i][3] << endl;
  }*/

  vector<int> helicities; 
  pair < double, bool > res;
  vector<pair<int,int> > pairs(fillColPairs(event));
  vector<pair<int,int> > in_pairs(pairs.begin(), pairs.begin()+2);
  vector<pair<int,int> > fi_pairs(pairs.begin()+2, pairs.end());
  vector<int> cols = fillColVec(pairs);

//cout << "enter " << __LINE__ << endl;
  int iperm=0;
  do {
    bool found=false;
    do {
      iperm++;
//      cout << "permutation=" << iperm<< endl;

      vector<pair<int,int> > new_pairs(in_pairs.begin(), in_pairs.end());
      new_pairs.insert(new_pairs.end(), fi_pairs.begin(), fi_pairs.end());
      cols.clear();
      cols = fillColVec(new_pairs);

      pvec2.clear();
      pvec2.insert(pvec2.end(), pi_mg5.begin(), pi_mg5.end());
      pvec2.insert(pvec2.end(), pf_mg5.begin(), pf_mg5.end());

//      for (int i=0; i < cols.size(); ++i)
//        cout << "\t" << cols[i];
//      cout << endl;

      PY8MEs_namespace::PY8ME * query = accessor.getProcess(in_pdgs, out_pdgs, req_s_channels);
//      accessor.getProcess(in_pdgs, out_pdgs, req_s_channels);
      res = accessor.calculateME(in_pdgs, out_pdgs, pvec2, req_s_channels, cols, helicities);
      query->setColors(cols);
      int colID = query->getColorIDForConfig(cols);
//      if (colID != -2) cout << "FOUND!" << endl;

      if (res.second && res.first > 0.) { found=true; break;}

      //vector<pair<int,int> > new_pairs(in_pairs.begin(), in_pairs.end());
      //new_pairs.insert(new_pairs.end(), fi_pairs.begin(), fi_pairs.end());
      //cols = fillColVec(new_pairs);
      //pvec2.clear();
      //pvec2.insert(pvec2.end(), pi_mg5.begin(), pi_mg5.end());
      //pvec2.insert(pvec2.end(), pf_mg5.begin(), pf_mg5.end());

//      cout << "Warning in "<<  __PRETTY_FUNCTION__ << ": Try next permutation of final states" << endl;
      std::next_permutation(out_pdgs.begin(),out_pdgs.end());
      std::next_permutation(pf_mg5.begin(),pf_mg5.end());
    } while(std::next_permutation(fi_pairs.begin(),fi_pairs.end()));

//    if (found) { cout << "found!" << endl; break;}
    if (found) break;

//    cout << "Warning in "<<  __PRETTY_FUNCTION__ << ": Try next permutation of initial states" << endl;
    std::next_permutation(in_pdgs.begin(),in_pdgs.end());
    std::next_permutation(pi_mg5.begin(),pi_mg5.end());
  } while(std::next_permutation(in_pairs.begin(),in_pairs.end()));

//cout << "done " << __LINE__ << endl;

  if (res.second && res.first < 0.) { cout << res.first << endl; abort(); }
  if (res.second) return res.first;

  //pair < double, bool > res = accessor.calculateME(in_pdgs, out_pdgs, p_vec,
  //    req_s_channels, colors, helicities);
  //if (res.second && res.first < 0.) abort(); 
  //if (res.second) return res.first;

  // Done
  return 0.0;

}
