
class TestHook : public UserHooks {

public:

  TestHook(int maxNumberOfEmissionsIn, Hist* hHIn, Hist* hSIn, Hist* hHIn2, Hist* hSIn2) : nISR(0), nFSR(0) {
    // Here, I assume that as default you want to terminate the evolution after the
    // first emission.
    maxNumberOfEmissions=maxNumberOfEmissionsIn;
    h_Hevent=hHIn;
    h_Sevent=hSIn;
    h_Hevent2=hHIn2;
    h_Sevent2=hSIn2;
    isHevent = isSevent = false;
  }

  bool canVetoISREmission() { return true; }
  bool canVetoFSREmission() { return true; }

  bool doVetoISREmission(int, const Event& e, int) {
    // Discard the emission if the maximal number of emission has been reached.
    if (nISR + nFSR >= maxNumberOfEmissions) return true;
    // At this stage, the emission is allowed, and we update the number of
    // emissions.
    nISR++;

    // Pythia radiator after, emitted and recoiler after.
    int iRadAft = -1, iEmt = -1, iRecAft = -1;
    for (int i = e.size() - 1; i > 0; i--) {
      if      (iRadAft == -1 && e[i].status() == -41) iRadAft = i;
      else if (iEmt    == -1 && e[i].status() ==  43) iEmt    = i;
      else if (iRecAft == -1 && e[i].status() == -42) iRecAft = i;
      if (iRadAft != -1 && iEmt != -1 && iRecAft != -1) break;
    }

    //h->fill(pTpythia(e, iRadAft, iEmt, iRecAft, false), weight);
    if (isHevent) h_Hevent->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), weight);
    else if (isSevent) h_Sevent->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), weight);

    if (isHevent) h_Hevent2->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), infoPtr->weight());
    else if (isSevent) h_Sevent2->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), infoPtr->weight());

    // Done.
    return false;
  }

  bool doVetoFSREmission(int, const Event& e, int, bool ) {
    // Discard the emission if the maximal number of emission has been reached.
    if (nISR + nFSR >= maxNumberOfEmissions) return true;
    // At this stage, the emission is allowed, and we update the number of
    // emissions.
    nFSR++;

    // Pythia radiator (before and after), emitted and recoiler (after)
    int iRecAft = e.size() - 1;
    int iEmt    = e.size() - 2;
    int iRadAft = e.size() - 3;

    if (isHevent) h_Hevent->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), weight);
    else if (isSevent) h_Sevent->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), weight);

    if (isHevent) h_Hevent2->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), infoPtr->weight());
    else if (isSevent) h_Sevent2->fill(pTpythia(e, iRadAft, iEmt, iRecAft, true), infoPtr->weight());

    // Done.
    return false;
  }

  bool canVetoProcessLevel() { return true; }
  bool doVetoProcessLevel(Event& process) {
    // Reset the emission counters.
    nISR = nFSR = 0;
    isHevent = isSevent = false;

    int npart=0;
    for (int i=0; i < process.size(); ++i)
      if (process[i].isFinal() && process[i].colType()!=0)
        npart++;

    if (settingsPtr->mode("TimeShower:nPartonsInBorn") == npart) isSevent = true;
    else isHevent = true;

    weight = 1.;
    if (true) {
      int nscales = 0;
      double sumscales = 0.;
      for ( map<string,double>::const_iterator
        it  = infoPtr->scales->attributes.begin();
        it != infoPtr->scales->attributes.end(); ++it ) {
        nscales++;
        sumscales += it->second;
      }
      weight = sumscales / double(nscales);
    }

    // No veto here.
    return false;
  }

  // Compute the Pythia pT separation. Based on pTLund function in History.cc
  inline double pTpythia(const Event &e, int RadAfterBranch,
    int EmtAfterBranch, int RecAfterBranch, bool FSR) {

    // Convenient shorthands for later
    Vec4 radVec = e[RadAfterBranch].p();
    Vec4 emtVec = e[EmtAfterBranch].p();
    Vec4 recVec = e[RecAfterBranch].p();
    int  radID  = e[RadAfterBranch].id();

    // Calculate virtuality of splitting
    double sign = (FSR) ? 1. : -1.;
    Vec4 Q(radVec + sign * emtVec);
    double Qsq = sign * Q.m2Calc();

    // Mass term of radiator
    double m2Rad = (abs(radID) >= 4 && abs(radID) < 7) ?
                   pow2(particleDataPtr->m0(radID)) : 0.;

    // z values for FSR and ISR
    double z, pTnow;
    if (FSR) {
      // Construct 2 -> 3 variables
      Vec4 sum = radVec + recVec + emtVec;
      double m2Dip = sum.m2Calc();
      double x1 = 2. * (sum * radVec) / m2Dip;
      double x3 = 2. * (sum * emtVec) / m2Dip;
      z     = x1 / (x1 + x3);
      pTnow = z * (1. - z);

    } else {
      // Construct dipoles before/after splitting
      Vec4 qBR(radVec - emtVec + recVec);
      Vec4 qAR(radVec + recVec);
      z     = qBR.m2Calc() / qAR.m2Calc();
      pTnow = (1. - z);
    }

    // Virtuality with correct sign
    pTnow *= (Qsq - sign * m2Rad);

    // Can get negative pT for massive splittings
    if (pTnow < 0.) {
      cout << "Warning: pTpythia was negative" << endl;
      return -1.;
    }

    // Return pT
    return sqrt(pTnow);
  }

  // Members.
  int nISR, nFSR, maxNumberOfEmissions;
  Hist* h_Hevent;
  Hist* h_Sevent;
  Hist* h_Hevent2;
  Hist* h_Sevent2;
  bool isHevent, isSevent;
  double weight;

};
