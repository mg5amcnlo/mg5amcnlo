// main182.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Authors: Torbjorn Sjostrand <torbjorn.sjostrand@thep.lu.se>

// Keywords: switch beam; switch collision energy; reuse MPI initialization.
// arXiv:2108.03481 [hep-ph]

// Test that events behave as intended when incoming beam is switched
// within one Pythia instance, and also that intermediate storage of
// MPI data works. Check execution time slowdown from switching.
// Specifically, compare three scenarios:
// Fixed:  each of the ten beams are handled separately, with no variation.
// Switch: initialize "all" beams,  switch beam for each new event.
// Reuse:  reuse initialization above, switch beam and energy for each event.
// Warning: the runtime is rather long (~15 minutes), since it involves
// the initialization of and looping over multiple incoming beam types.

#include "Pythia8/Pythia.h"
#include <time.h>

using namespace Pythia8;

int main() {

  // Number of test events per beam configuration (iType).
  int nEvent = 10000;

  // List of alternating incoming hadrons.
  vector<int> idAtype = { 2212, 213, 323, 2224, 331, 421, -421, 3212,
    3334, 4214};

  // Histograms.
  Hist typefixed[10], typeswitch[10], typereuse[10],
       nMPIfixed[10], nMPIswitch[10], nMPIreuse[10],
       nchgfixed[10], nchgswitch[10], nchgreuse[10];
  for (int iType = 0; iType < 10;++iType) {
    typefixed[iType].book( "event type, fixed  ", 10, -0.5, 9.5);
    typeswitch[iType].book("event type, switch ", 10, -0.5, 9.5);
    typereuse[iType].book( "event type, reuse  ", 10, -0.5, 9.5);
    nMPIfixed[iType].book( "number of MPIs, fixed  ", 40, -0.5, 39.5);
    nMPIswitch[iType].book("number of MPIs, switch ", 40, -0.5, 39.5);
    nMPIreuse[iType].book( "number of MPIs, reuse  ", 40, -0.5, 39.5);
    nchgfixed[iType].book( "charged multiplicity, fixed  ", 100, -0.5, 399.5);
    nchgswitch[iType].book("charged multiplicity, switch ", 100, -0.5, 399.5);
    nchgreuse[iType].book( "charged multiplicity, reuse  ", 100, -0.5, 399.5);
  }

  // Timing info.
  clock_t tstart, tstop, tFixedInit, tFixedRun, tSwitchInit, tSwitchRun,
    tReuseInit, tReuseRun;
  tFixedInit = tFixedRun = 0;
  tstart = clock();

  // First case: fixed.  ------------------------------------------------

  // Do each incoming beam particle separately, without any switching.
  for (int iType = 0; iType < 10; ++iType) {

    // Object with fixed beam hadron. (New for each idA value.)
    Pythia pythiaFixed;
    // Fixed incoming beam type (and energy).
    pythiaFixed.settings.mode("Beams:idA", idAtype[iType]);
    pythiaFixed.readString("Beams:eCM = 8000.");
    // SoftQCD processes to compare with above.
    pythiaFixed.readString("SoftQCD:all = on");
    // Reduce output.
    pythiaFixed.readString("Print:quiet = on");
    // Initialize.
    if (!pythiaFixed.init()) {
      cout << "pythiaFixed failed to initialize." << endl;
      return -2;
    }

    // Timing.
    tstop = clock();
    tFixedInit += tstop - tstart;
    tstart = tstop;

    // Generate test events.
    for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
      pythiaFixed.next();

      // Fill histograms.
      typefixed[iType].fill( pythiaFixed.info.code() - 100);
      nMPIfixed[iType].fill(  pythiaFixed.info.nMPI() );
      nchgfixed[iType].fill(  pythiaFixed.event.nFinal(true) );
    }
    pythiaFixed.stat();

    // Timing. End of beam particle loop.
    tstop = clock();
    tFixedRun += tstop - tstart;
    tstart = tstop;
  }

  // Second case: switch.  ------------------------------------------------

  // Object which allows switching ids, creating a new MPI init file.
  Pythia pythiaSwitch;
  // Variable incoming beam type (and energy).
  pythiaSwitch.readString("Beams:allowVariableEnergy = on");
  pythiaSwitch.readString("Beams:allowIDAswitch = on");
  pythiaSwitch.readString("Beams:eCM = 8000.");
  // Must use SoftQCD processes. Save MPI initialization results.
  pythiaSwitch.readString("SoftQCD:all = on");
  pythiaSwitch.readString("MultipartonInteractions:reuseInit = 1");
  pythiaSwitch.readString("MultipartonInteractions:initFile = main182.mpi");
  // Reduce output and reset statistics after each subrun.
  pythiaSwitch.readString("Print:quiet = on");
  // Initialize.
  if (!pythiaSwitch.init()) {
    cout << "pythiaSwitch failed to initialize." << endl;
    return -1;
  }

  // Timing.
  tstop = clock();
  tSwitchInit = tstop - tstart;
  tstart = tstop;

  // Generate events, switching incoming particle, but same energy.
  for (int iEvent = 0; iEvent < 10 * nEvent; ++iEvent) {
    int iType = iEvent%10;
    pythiaSwitch.setBeamIDs(idAtype[iType]);
    pythiaSwitch.next();

    // Fill histograms.
    typeswitch[iType].fill( pythiaSwitch.info.code() - 100);
    nMPIswitch[iType].fill( pythiaSwitch.info.nMPI() );
    nchgswitch[iType].fill( pythiaSwitch.event.nFinal(true) );
  }
  pythiaSwitch.stat();

  // Timing.
  tstop = clock();
  tSwitchRun = tstop - tstart;
  tstart = tstop;

  // Third case: reuse.  ------------------------------------------------

  // Object which allows switching ids, reading an existing MPI init file.
  Pythia pythiaReuse;
  // Variable incoming beam type (and energy).
  pythiaReuse.readString("Beams:allowVariableEnergy = on");
  pythiaReuse.readString("Beams:allowIDAswitch = on");
  pythiaReuse.readString("Beams:eCM = 8000.");
  // Must use SoftQCD processes. Read MPI initialization results from above.
  pythiaReuse.readString("SoftQCD:all = on");
  pythiaReuse.readString("MultipartonInteractions:reuseInit = 2");
  pythiaReuse.readString("MultipartonInteractions:initFile = main182.mpi");
  // Reduce output and reset statistics after each subrun.
  pythiaReuse.readString("Print:quiet = on");
  // Initialize.
  if (!pythiaReuse.init()) {
    cout << "pythiaReuse failed to initialize." << endl;
    return -1;
  }

  // Timing.
  tstop = clock();
  tReuseInit = tstop - tstart;
  tstart = tstop;

  // Generate events, switching incoming particle and energy.
  for (int iEvent = 0; iEvent < 10 * nEvent; ++iEvent) {
    int iType = iEvent%10;
    double eCMnow = 7990 + 10. * pythiaReuse.rndm.flat();
    pythiaReuse.setBeamIDs(idAtype[iType]);
    pythiaReuse.setKinematics(eCMnow);
    pythiaReuse.next();

    // Fill histograms.
    typereuse[iType].fill( pythiaReuse.info.code() - 100);
    nMPIreuse[iType].fill( pythiaReuse.info.nMPI() );
    nchgreuse[iType].fill( pythiaReuse.event.nFinal(true) );
  }
  pythiaReuse.stat();

  // Timing.
  tstop = clock();
  tReuseRun = tstop - tstart;

  // Output processing. ------------------------------------------------

  // Print timing info (in seconds).
  double conv = 1. / double(CLOCKS_PER_SEC);
  cout << endl << fixed << setprecision(3)
       << " initialization time, fixed  " << setw(8)
       << conv * tFixedInit  << " s" << endl
       << " initialization time, switch " << setw(8)
       << conv * tSwitchInit << " s" << endl
       << " initialization time, reuse  " << setw(8)
       << conv * tReuseInit  << " s" << endl
       << " generation time, fixed      " << setw(8)
       << conv * tFixedRun   << " s" << endl
       << " generation time, switch     " << setw(8)
       << conv * tSwitchRun  << " s" << endl
       << " generation time, reuse      " << setw(8)
       << conv * tReuseRun   << " s" << endl;

    // Plotting object. Names of incoming beam hadrons.
  HistPlot hpl("main182plot");
  vector<string> idAname = { "p", "$\\rho^+$", "K$^{*+}$", "$\\Delta^{++}$",
    "$\\eta^{\\mathrm{prime}}$", "D$^0$", "$\\overline{\\mathrm{D}}^0$",
    "$\\Sigma^0$", "$\\Omega^-$", "$\\Lambda_{\\mathrm{c}}^+$" };

    // Normalize histograms, one beam hadron at a time.
  for (int iType = 0; iType < 10; ++iType) {
    typefixed[iType]  /= nEvent;
    typeswitch[iType] /= nEvent;
    typereuse[iType]  /= nEvent;
    nMPIfixed[iType]  /= nEvent;
    nMPIswitch[iType] /= nEvent;
    nMPIreuse[iType]  /= nEvent;
    nchgfixed[iType]  /= 4 * nEvent;
    nchgswitch[iType] /= 4 * nEvent;
    nchgreuse[iType]  /= 4 * nEvent;

    // Plot histograms.
    string label0 = idAname[iType] + "p event type (ND/EL/SD(XB)/SD(AX)/DD)";
    string label1 = idAname[iType] + "p number of MPIs";
    string label2 = idAname[iType] + "p charged multiplicity";
    hpl.frame( "out182plot", label0, "type - 100", "Probability");
    hpl.add( typefixed[iType],  "h,black", "fixed beam");
    hpl.add( typeswitch[iType], "h,red",   "switch beam");
    hpl.add( typereuse[iType],  "h,blue",  "reuse beam");
    hpl.plot();
    hpl.frame( "out182plot", label1, "$n_{\\mathrm{MPI}}$", "Probability");
    hpl.add( nMPIfixed[iType],  "h,black", "fixed beam");
    hpl.add( nMPIswitch[iType], "h,red",   "switch beam");
    hpl.add( nMPIreuse[iType],  "h,blue",  "reuse beam");
    hpl.plot();
    hpl.frame( "out182plot", label2, "$n_{\\mathrm{ch}}$", "Probability");
    hpl.add( nchgfixed[iType],  "h,black", "fixed beam");
    hpl.add( nchgswitch[iType], "h,red",   "switch beam");
    hpl.add( nchgreuse[iType],  "h,blue",  "reuse beam");
    hpl.plot();
  }

  return 0;
}
