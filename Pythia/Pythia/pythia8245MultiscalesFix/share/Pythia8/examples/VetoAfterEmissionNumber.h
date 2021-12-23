
class VetoAfterEmissionNumber : public UserHooks {

public:

  VetoAfterEmissionNumber(int maxNumberOfEmissionsIn = 1) : nISR(0), nFSR(0) {
    // Here, I assume that as default you want to terminate the evolution after the
    // first emission.
    maxNumberOfEmissions=maxNumberOfEmissionsIn;
  }

  bool canVetoISREmission() { return true; }
  bool canVetoFSREmission() { return true; }

  bool doVetoISREmission(int, const Event&, int) {
    // Discard the emission if the maximal number of emission has been reached.
    if (nISR + nFSR >= maxNumberOfEmissions) return true;
    // At this stage, the emission is allowed, and we update the number of
    // emissions.
    nISR++;
    // Done.
    return false;
  }

  bool doVetoFSREmission(int, const Event&, int, bool acceptEmission ) {
    // Discard the emission if the maximal number of emission has been reached.
    if (nISR + nFSR >= maxNumberOfEmissions) return true;
    // At this stage, the emission is allowed, and we update the number of
    // emissions.
    nFSR++;
    // Done.
    return false;
  }

  bool canVetoProcessLevel() { return true; }
  bool doVetoProcessLevel(Event&) {
    // Reset the emission counters.
    nISR = nFSR = 0;
     // No veto here.
    return false;
  }

  // Members.
  int nISR, nFSR, maxNumberOfEmissions;

};
