#include <iostream>
#include <fstream> 

#include "SystCalc.h"

/**************************************************************************
Main program to convert .sys files from MG5 to systematics variation files.

Usage: syst_calc sysfile configfile outfile

Example of a config file:
      # Central scale factors
      scalefact:
      0.5 1 2
      # \alpha_s emission scale factors
      alpsfact:
      0.5 1 2
      # matching scales
      matchscale:
      20 30 40
      # PDF sets and number of members
      PDF:
      CT10.LHgrid 52
      MSTW2008nlo68cl.LHgrid 40

Using tinyXML2 for XML parsing of syst file.
**************************************************************************/

int main( int argc, const char ** argv)
{
  if (argc < 4){
    cout << "Usage: syst_calc sysfile configfile outfile" << endl;
    exit(1);
  }
  ifstream conffile(argv[2]);
  if (conffile.fail()) { 
    cout << "Failed opening config file " << argv[2] << endl; 
    exit(1); }

  // Initialize SystCalc object with conffile and sysfile
  SystCalc* systcalc = new SystCalc(conffile, argv[1]);
  if (systcalc->noFile()) { 
    cout << "Failed opening .sys file " << argv[1] << endl; 
    exit(1); }

  ofstream outfile(argv[3]);
  if (outfile.fail()) { 
    cout << "Failed opening output file " << argv[3] << endl; 
    exit(1); }
  // Write XML header for outfile
  systcalc->writeHeader(outfile);

  // Parse events one by one
  while (systcalc->parseEvent()){
    // Calculate event weights for systematics parameters
    systcalc->convertEvent();
    // Write out new rwt block to outfile
    systcalc->writeEvent(outfile);
  }
  cout << "Finished parsing " << systcalc->parsedEvents() << " events." << endl;
}
