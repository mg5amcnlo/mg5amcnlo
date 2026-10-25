// Allow an analysis to select the HEPEVT layout declared in its Fortran common.
// Other analyses retain HepMC's default layout unless they call this function.
#ifndef MG5_HEPEVT_SIZE_H
#define MG5_HEPEVT_SIZE_H

#include "HepMC/HEPEVT_Wrapper.h"
#include <cstdlib>
#include <iostream>

extern "C" void hepmc_set_max_number_entries_(const int& entries) {
  if (entries < 1 || entries > HEPEVT_EntriesAllocation) {
    std::cerr << "Unsupported analysis HEPEVT capacity: " << entries
              << " (maximum " << HEPEVT_EntriesAllocation << ")\n";
    std::exit(1);
  }
  HepMC::HEPEVT_Wrapper::set_max_number_entries(entries);
}

#endif
