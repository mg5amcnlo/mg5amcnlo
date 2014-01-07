// -*- C++ -*-
//
// This is the implementation of the non-inlined, non-templated member
// functions of the HepMCFortran class.
//

#include "HepMCFortran.h"
#include "ThePEG/PDT/EnumParticles.h"
#include "ThePEG/Repository/EventGenerator.h"
#include "ThePEG/EventRecord/Event.h"
#include "ThePEG/Interface/ClassDocumentation.h"
#include "ThePEG/Config/HepMCHelper.h"

#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"
#include "HepMC/IO_BaseClass.h"

using namespace Herwig;
using namespace MCatNLO;

extern "C" {
    extern struct {
       double EVWGT;
    } cevwgt_;
}
#define cevwgt cevwgt_

extern "C" { 
  //void foranalysis_(); 
  void hwabeg_();
  void hwaend_();
  void hwanal_();

}

void HepMCFortran::analyze(tEventPtr event, long ieve, int loop, int state) {
  AnalysisHandler::analyze(event, ieve, loop, state);

  //define the IO_HEPEVT
  _hepevtio = new HepMC::IO_HEPEVT;
  
  //choose units to be GeV and MM
  Energy eUnit; Length lUnit;
  eUnit = GeV; lUnit = millimeter;
 
  //convert the event from the Herwig++ format to the HepMC format and write it to the common block
  HepMC::GenEvent * hepevtf = HepMCConverter<HepMC::GenEvent>::convert(*event, false,eUnit, lUnit);
  _hepevtio->write_event(hepevtf); 
  //event weight
  cevwgt.EVWGT=hepevtf->weights()[0];
  
  
  //call the FORTRAN analysis for this event (see example file foranalysis.f)
  //foranalysis_();
  hwanal_();

  return;
  
}

IBPtr HepMCFortran::clone() const {
  return new_ptr(*this);
}

IBPtr HepMCFortran::fullclone() const {
  return new_ptr(*this);
}

NoPIOClassDescription<HepMCFortran> HepMCFortran::initHepMCFortran;
// Definition of the static class description member.

void HepMCFortran::Init() {

  static ClassDocumentation<HepMCFortran> documentation
    ("There is no documentation for the HepMCFortran class");

}

void HepMCFortran::dofinish() {
  AnalysisHandler::dofinish();

  hwaend_();

}

void HepMCFortran::doinitrun() {
  AnalysisHandler::doinitrun();

  hwabeg_();

}
