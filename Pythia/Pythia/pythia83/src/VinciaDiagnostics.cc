// VinciaDiagnstoics.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Peter Skands, Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for Vincia's
// diagnostics class.

#include "Pythia8/VinciaCommon.h"
#include "Pythia8/VinciaDiagnostics.h"

namespace Pythia8 {

using namespace VinciaConstants;

//==========================================================================

// Vincia diagnostics.

//--------------------------------------------------------------------------

// Define and increment a counter.

void VinciaDiagnostics::increment(string methodName, string variableName,
  double inc) {
    map<string, double> counter;
    if (counters.find(methodName) != counters.end())
      counter = counters[methodName];
    else counters[methodName] = counter;
    if (counter.find(variableName) != counter.end())
      counters[methodName][variableName] += inc;
    else
      counters[methodName][variableName] = inc;
}

//--------------------------------------------------------------------------

// Called when "name" starts.

void VinciaDiagnostics::start(string name) {
  // Measure number of calls.
  ++nStarts[name];
  // If this is already running, this is a restart.
  if (isRunning.find(name) != isRunning.end()) {
    if (isRunning[name]) ++nRestarts[name];
    else startTime[name] = clock();
  } else {
    // If not already running, start clock.
    startTime[name] = clock();
    isRunning[name] = true;
  }
}

//--------------------------------------------------------------------------

// Called when "name" starts.

void VinciaDiagnostics::stop(string name, string counter, double inc) {
  // Measure CPU runtime.
  isRunning[name] = false;
  clock_t stopTime = clock();
  double runNow = 1000. * (double) (stopTime-startTime[name])/CLOCKS_PER_SEC;
  if (runTime.find(name) == runTime.end()) {
    hRunTime[name] = Hist("runTime in milliseconds",100,0.0,10.);
    runTime[name]  = runNow;
  }
  else {
    runTime[name] += runNow;
  }
  hRunTime[name].fill(runNow);
  if (counter != "") increment(name, counter, inc);
}

//--------------------------------------------------------------------------

// Print diagnostics.

void VinciaDiagnostics::print() {
  cout<<"\n *-------  VINCIA Diagnostics and Profiling -------------------"
    "----------------------------------------------------------*\n";
  double nEvent = max((double)infoPtr->nAccepted(),1.);
  for (auto it = nStarts.begin(); it != nStarts.end(); ++it) {
    string name = it->first;
    cout<<" |\n"
        <<" | Diagnostics for " << name<<endl;
    cout<<" |   total time = " << num2str(runTime[name]/nEvent,9)
        <<"ms/Event   nCalls/event = " << num2str(nStarts[name]/nEvent,9)
        <<"   failure rate = "
        << nRestarts[name]/max(1.,nStarts[name]) << endl;
    // Produce tables with over- and underflow bins.
    string fileName = name;
    // Remove last 2 characters () from name
    fileName.resize(fileName.size() - 2);
    fileName += ".runtime";
    hRunTime[name].table(fileName, true);
    // Also print counters associated with the same method.
    if (counters.find(name) != counters.end()) {
      for (auto jt = counters[name].begin(); jt != counters[name].end();
           ++jt) {
        string variable = jt->first;
        double rate     = jt->second/nEvent;
        cout<<" |   "<<num2str(rate,9)<<" : "<<variable<<" / event"<<endl;
      }
      counters.erase(name);
    }
  }
  // Print any counters not associated with profiled methods.
  for (auto it = counters.begin(); it != counters.end(); ++it) {
    string name = it->first;
    cout<<" | \n"
        <<" | Diagnostics for "<<name<<endl;
    for (auto jt = counters[name].begin(); jt != counters[name].end();
         ++jt) {
      string variable = jt->first;
      double rate     = jt->second/nEvent;
      cout<<" |   "<<num2str(rate,9)<<" : "<<variable<<" / event"<<endl;
    }
  }
  cout<<" |\n"
      <<" | See also the generated .runtime files for "
      <<"histograms of run times per call.\n";
  cout<<" |\n *-------  End VINCIA Diagnostics and Profiling -----------"
    "--------------------------------------------------------------*\n\n";
}

//==========================================================================

} // end namespace Pythia8
