#include <iostream>
#include <fstream> 

#include "SysCalc.h"

/**************************************************************************
Main program to convert systematics files from MG5 to systematics variation files.

Usage: sys_calc sysfile configfile outfile

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
      # PDF sets, number of members (default/0 means use all members), 
      # combination method (default hessian, note that NNPDF uses gaussian)
      PDF:
      CT10.LHgrid 52 hessian
      MSTW2008nlo68cl.LHgrid

Using tinyXML2 for XML parsing of syst file.
**************************************************************************/

int main( int argc, const char ** argv)
{
  if (argc < 4){
    cout << "Usage: sys_calc sysfile configfile outfile" << endl;
    cout << " Note: If input file is .lhe, full event info is copied to outfile." << endl;
    exit(1);
  }
  ifstream conffile(argv[2]);
  if (conffile.fail()) { 
    cout << "Failed opening config file " << argv[2] << endl; 
    exit(1); }

  // Initialize SysCalc object with conffile and sysfile
  SysCalc* syscalc = new SysCalc(conffile, argv[1]);
  if (!syscalc->fileStatus()) { 
    cout << "Failed opening or parsing systematics file " << argv[1] << endl; 
    cout << "XML Error: " << syscalc->fileStatus() << endl;
    exit(1); }

  if(string(argv[1]).find(".lhe") != string::npos)
    syscalc->lheOutput(true);
  else
    syscalc->lheOutput(false);    

  // Open output file
  bool first = true;

  ofstream* outfile = 0;

  // Parse events one by one
  while (syscalc->parseEvent()){
    // If first time, write header
    if (first){
      outfile = new ofstream(argv[3]);
      if (outfile->fail()) { 
	cout << "Failed opening output file " << argv[3] << endl; 
	exit(1); 
      }
      // Write XML header for outfile
      syscalc->writeHeader(*outfile);
      first = false;
    }
    // Calculate event weights for systematics parameters
    syscalc->convertEvent();
    // Write out new rwt block to outfile
    syscalc->writeEvent(*outfile);
  }
  if(syscalc->lheOutput())
    *outfile << "</LesHouchesEvents>";

  cout << "Finished parsing " << syscalc->parsedEvents() << " events." << endl;
  syscalc->write_xsec();
}
