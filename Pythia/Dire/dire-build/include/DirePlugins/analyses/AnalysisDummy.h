#include "Analysis.h"

class MyAnalysis : public Analysis {

  public:

  void init();
  void fill(const Event&, double);
  void fill(const Event&, const Event&, double) { return; };
  void finalize();
  void print();

};

void MyAnalysis::init() {
  // Initialize histograms here.

  // Example: Charged hadron multiplicity.
  histograms.insert(make_pair("nch_vs_pt",
    Hist("nch_vs_pt",100,0.0,100.0)));
}

void MyAnalysis::fill(const Event& e, double w) {
  // Calculate values from event record, then fill
  // histograms here.

  // Example: Accumulate sum of weights.
  sumOfWeights += w;

  // Example: histogram charged hadron multiplicity.
  for ( int i=0; i < e.size(); ++i )
    if ( e[i].isFinal() && e[i].isCharged() && e[i].isHadron() ) {
      double pt = e[i].pT();
      // Fill histogram.
      histograms["nch_vs_pt"].fill ( pt, w );
    }
}

void MyAnalysis::finalize() {
  // Finalize histograms here, e,g. normalize by sum of
  // event weights or cross section.

  // Example: Normalize all histograms to sum of weights.
  for ( map<string,Hist>::const_iterator it_h  = histograms.begin();
    it_h != histograms.end(); ++it_h )
    histograms[it_h->first] /= sumOfWeights;

}

void MyAnalysis::print() {
  // Table histograms to a gnuplot-usable file here.

  // Write histograms to file
  ofstream write;
  for ( map<string,Hist>::const_iterator it_h  = histograms.begin();
    it_h != histograms.end(); ++it_h ) {
    //ostringstream fname;
    //fname << "My_" << it_h->first << ".dat";
    //write.open( (fname.str()).c_str() );
    write.open( (it_h->first + ".dat").c_str() );
    histograms[it_h->first].table(write);
    write.close();
  }

}

