#include <iostream>
#include <stdlib.h>

/*
  NLO mode of aMCatNLO
*/

void (*appl_initptr)() = 0;
void (*appl_fillptr)() = 0;
void (*appl_fillrefptr)() = 0;
void (*appl_fillrefoutptr)() = 0;
void (*appl_recoptr)() = 0;
void (*appl_termptr)() = 0;

extern "C" void appl_init_() {
  // std::cerr<<"I am in appl_init (C++ version)"<<std::endl;
  if (appl_initptr) appl_initptr(); 
}

extern "C" void appl_fill_() {
  //std::cerr<<"I am in appl_fill (C++ version)"<<std::endl;
  if (appl_fillptr) appl_fillptr(); 
}

extern "C" void appl_fill_ref_() {
  //std::cerr<<"I am in appl_fill_ref (C++ version)"<<std::endl;
  if (appl_fillrefptr) appl_fillrefptr(); 
}

extern "C" void appl_fill_ref_out_() {
  // std::cerr<<"I am in appl_fill_ref_out (C++ version)"<<std::endl;
  if (appl_fillrefoutptr) appl_fillrefoutptr(); 
}

extern "C" void appl_reco_() {
  // std::cerr<<"I am in appl_reco (C++ version)"<<std::endl;
  if (appl_recoptr) appl_recoptr(); 
}

extern "C" void appl_term_() {
  //std::cerr<<"I am in appl_term (C++ version)"<<std::endl;
  if (appl_termptr) appl_termptr();  
}


/*
  NLO+PS mode of aMCatNLO
*/

void (*applmc_initptr)() = 0;
void (*applmc_fillrefptr)() = 0;
void (*applmc_termptr)() = 0;

extern "C" void applmc_init_() {
  std::cerr<<"I am in applmc_init (C++ version)"<<std::endl;
  if (applmc_initptr) applmc_initptr(); 
}

extern "C" void applmc_fill_ref_() {
  std::cerr<<"I am in applmc_fill_ref_ (C++ version)"<<std::endl;
  if (applmc_fillrefptr) applmc_fillrefptr(); 
}

extern "C" void applmc_term_() {
  std::cerr<<"I am in applmc_term_ (C++ version)"<<std::endl;
  if (applmc_termptr) applmc_termptr(); 
}

