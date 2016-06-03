#include <iostream>
#include <iomanip> 

#include "CPPProcess.h"
#include "rambo.h"

int main(int argc, char** argv){

  // Create a process object
  CPPProcess process;

  // Read param_card and set parameters
  process.initProc("../../Cards/param_card.dat");

  double energy = 1500;
  double weight;

  // Get phase space point
  vector<double*> p = get_momenta(process.ninitial, energy, 
				 process.getMasses(), weight);

  // Set momenta for this event
  process.setMomenta(p);

  // Evaluate matrix element
  process.sigmaKin();

  const double* matrix_elements = process.getMatrixElements();

  cout << "Momenta:" << endl;
  for(int i=0;i < process.nexternal; i++)
    cout << setw(4) << i+1 
	 << setiosflags(ios::scientific) << setw(14) << p[i][0]
	 << setiosflags(ios::scientific) << setw(14) << p[i][1]
	 << setiosflags(ios::scientific) << setw(14) << p[i][2]
	 << setiosflags(ios::scientific) << setw(14) << p[i][3] << endl;
  cout << " -----------------------------------------------------------------------------" << endl;

  // Display matrix elements
  for(int i=0; i<process.nprocesses;i++)
    cout << " Matrix element = " 
//	 << setiosflags(ios::fixed) << setprecision(17)
	 << matrix_elements[i]
	 << " GeV^" << -(2*process.nexternal-8) << endl;

  cout << " -----------------------------------------------------------------------------" << endl;
}
