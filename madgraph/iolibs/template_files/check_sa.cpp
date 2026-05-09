#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "CPPProcess.h"
#include "rambo.h"

// ---------------------------------------------------------------------------
// Flavor combinations and corresponding PDG codes.
// %(maxflavor)d flavor combination(s); %(nexternal)d external particles each.
// These arrays are filled by the MG5_aMC@NLO code generator.
// ---------------------------------------------------------------------------
static const int maxflavor  = %(maxflavor)d;
static const int _nexternal = %(nexternal)d;
// flavor_arr[iflav][ipart] : flavor index (0 = default/non-merged)
static const int flavor_arr[%(maxflavor)d][%(nexternal)d] = %(flavor_arr)s;
// pdg_arr[iflav][ipart]    : actual PDG code for this particle/flavor
static const int pdg_arr[%(maxflavor)d][%(nexternal)d]    = %(pdg_arr)s;

int main(int argc, char** argv){

  // Create a process object
  CPPProcess process;

  // Read param_card and set parameters
  process.initProc("../../Cards/param_card.dat");

  // Centre-of-mass energy: use argv[1] if provided, else default 1000 GeV
  double energy = 1000.0;
  if(argc > 1) energy = atof(argv[1]);
  double total_mass = 0.0;
  vector<double> masses = process.getMasses();
  for(unsigned int i = 0; i < masses.size(); ++i) total_mass += masses[i];
  if(energy <= 2.0 * total_mass) energy = 2.0 * total_mass;
  double weight;

  // Get phase space point (RAMBO with fixed seed -> reproducible)
  vector<double*> p = get_momenta(process.ninitial, energy,
                                  process.getMasses(), weight);

  // Print phase space point (same format as the Fortran check_sa.f)
  cout << endl << " Phase space point:" << endl << endl;
  cout << "-----------------------------------------------------------------------------" << endl;
  cout << "n        E             px             py              pz" << endl;
  for(int i = 0; i < process.nexternal; i++)
    cout << setw(4) << i+1
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][0]
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][1]
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][2]
         << setiosflags(ios::scientific) << setprecision(7) << setw(15) << p[i][3] << endl;
  cout << "-----------------------------------------------------------------------------" << endl;

  // Set momenta once (shared across all flavor combinations)
  process.setMomenta(p);

  // Loop over flavor combinations
  for(int iflav = 0; iflav < maxflavor; iflav++){
    int flavor[_nexternal];
    for(int j = 0; j < _nexternal; j++) flavor[j] = flavor_arr[iflav][j];

    // Evaluate matrix element for this flavor combination
    process.sigmaKin(flavor);
    const double* matrix_elements = process.getMatrixElements();

    // Print PDG line (same keyword as Fortran so _parse_sa_output can match it)
    cout << " PDG";
    for(int j = 0; j < _nexternal; j++) cout << " " << pdg_arr[iflav][j];
    cout << endl;

    // Print matrix element values
    for(int iproc = 0; iproc < process.nprocesses; iproc++)
      cout << " Matrix element = " << matrix_elements[iproc]
           << " GeV^" << -(2*process.nexternal-8) << endl;

    cout << " -----------------------------------------------------------------------------" << endl;
  }

  return 0;
}
