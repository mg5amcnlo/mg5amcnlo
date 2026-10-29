// C ABI for exactly the Pythia PDF selected by the shower probe.
#include "Pythia8/Pythia.h"
#include <memory>
static std::unique_ptr<Pythia8::Pythia> pythia;
static Pythia8::PDFPtr pdf;
extern "C" int pythia_pdf_init(const char* xmldir,const char* pdfset) {
  pythia.reset(new Pythia8::Pythia(xmldir,false));
  if (!pythia->readString(std::string("PDF:pSet = ")+pdfset)) return 1;
  pdf=pythia->getPDFPtr(2212);
  return (!pdf || !pdf->isSetup()) ? 2:0;
}
extern "C" double pythia_pdf_xf(int pdg,double x,double q) {
  return pdf ? pdf->xf(pdg,x,q*q) : -1.;
}
