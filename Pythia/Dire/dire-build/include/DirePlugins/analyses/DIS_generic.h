#include "Pythia8/Pythia.h"
#include "Analysis.h"
//#include "Hermes_Flux.h"

class MyAnalysis : public Analysis {

  public:

  void init();
  void fill(const Event&, double);
  void fill(const Event&, const Event&, double) { return; };
  void fill(const Event&, vector<double> ) { return; }
  void fill(const Event&, const Event&, vector<double>) { return; };
  void finalize();
  void print();

  int nTried, nAccepted;

};

void MyAnalysis::init() {
  // Initialize histograms.
  histograms.insert(
    make_pair("dsigma_vs_Q", Hist("dsigma_vs_Q",100,0.,10)) );
  histograms.insert(
    make_pair("dsigma_vs_xB", Hist("dsigma_vs_xB",100,0.,1.0)) );
  histograms.insert(
    make_pair("dsigma_vs_W", Hist("dsigma_vs_W",100,0.,10)) );
  histograms.insert(
    make_pair("dsigma_vs_y", Hist("dsigma_vs_y",100,0.,1.0)) );

  nTried=nAccepted=0;

}

void MyAnalysis::fill(const Event & e, double w) {

  // Find position of incoming proton, incoming electron,
  // outgoing electron.
  int iBeamA(0), iBeamB(0);
  for ( int i=0; i < e.size(); ++i ) {
    if      ( e[i].statusAbs() == 12 && e[i].pz() > 0) iBeamA = i;
    else if ( e[i].statusAbs() == 12 ) iBeamB = i;
    //if ( e[i].statusAbs() == 12 && e[i].pz() < 0) iBeamB = i;
  }
  int iProton = e[iBeamA].isHadron() ? iBeamA : iBeamB;
  int iOther  = iProton == iBeamA ? iBeamB : iBeamA;
  int iInElectron(0), iScatElHard(0);
  for ( int i=0; i < e.size(); ++i ) {
    if ( !e[i].isFinal() && e[i].isAncestor(iOther) ) iInElectron = i;
    //if ( e[i].statusAbs() == 23 && e[i].idAbs() == 11) iScatElHard = i;
    if ( e[i].statusAbs() == 23 && e[i].idAbs() == 13) iScatElHard = i;
  }

  int iScatElectron(0);
  for ( int i=e.size()-1; i > 0; --i )
    //if ( e[i].isFinal() && e[i].idAbs() == 11
    if ( e[i].isFinal() && e[i].idAbs() == 13
      && e[i].isAncestor(iScatElHard) ) { iScatElectron = i; break;}
  iScatElectron = max(iScatElHard, iScatElectron);

  // Construct Q2, W2, y, xbj.
  Vec4 pInEl   ( e[iInElectron].p() );
  Vec4 pScatEl ( e[iScatElectron].p() );
  Vec4 pProton ( e[iProton].p() );
  Vec4 q       ( pInEl - pScatEl );
  double Q2  = -q.m2Calc();
  double W2  = ( pProton + q ).m2Calc();
  double y   = (pProton*q) / (pProton*pInEl);
  double xbj = Q2 / (2.*pProton*q);

  if ( Q2 < 0.5 ) return;
  if ( W2 < 4.0 ) return;

  // Accumulate sum of weights.
  sumOfWeights += w;

  nTried++;

  if ( y < 0.05 ) return;
  if ( xbj < 0.002 || xbj > 0.99 ) return;

  nAccepted++;

  histograms["dsigma_vs_Q"].fill  ( sqrt(Q2), w / 0.1 );
  histograms["dsigma_vs_xB"].fill ( xbj,      w / 0.01 );
  histograms["dsigma_vs_W"].fill  ( sqrt(W2), w / 0.1 );
  histograms["dsigma_vs_y"].fill  ( y,        w / 0.01 );

}

void MyAnalysis::finalize() {
  // Normalize histograms.
  for ( map<string,Hist>::const_iterator it_h  = histograms.begin();
    it_h != histograms.end(); ++it_h )
    histograms[it_h->first] /= sumOfWeights;
}

void MyAnalysis::print() {

  // Write histograms to file
  ofstream write;
  for ( map<string,Hist>::const_iterator it_h  = histograms.begin();
    it_h != histograms.end(); ++it_h ) {
    ostringstream fname;
    fname << "DIS_generic_" << it_h->first << ".dat";
    write.open( (fname.str()).c_str() );
    histograms[it_h->first].table(write);
    write.close();
  }

}
