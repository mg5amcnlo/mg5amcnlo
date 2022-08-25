// HepMC3.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.
//
// Author: HepMC 3 Collaboration, hepmc-dev@.cern.ch
// Based on the HepMC2 interface by Mikhail Kirsanov, Mikhail.Kirsanov@cern.ch.
// Header file and function definitions for the Pythia8ToHepMC class,
// which converts a PYTHIA event record to the standard HepMC format.

#ifndef Pythia8_HepMC3_H
#define Pythia8_HepMC3_H

#ifdef Pythia8_HepMC2_H
#error Cannot include HepMC3.h if HepMC2.h has already been included.
#endif

#include <vector>
#include "Pythia8/Pythia.h"
#include "Pythia8/HIUserHooks.h"
#include "HepMC3/GenVertex.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/GenEvent.h"
#include "HepMC3/WriterAscii.h"
#include "HepMC3/WriterAsciiHepMC2.h"
#include "HepMC3/GenHeavyIon.h"
#include "HepMC3/GenPdfInfo.h"

namespace HepMC3 {

//==========================================================================

class Pythia8ToHepMC3 {

public:

  // Constructor and destructor
  Pythia8ToHepMC3(): m_internal_event_number(0), m_print_inconsistency(true),
    m_free_parton_warnings(true), m_crash_on_problem(false),
    m_convert_gluon_to_0(false), m_store_pdf(true), m_store_proc(true),
    m_store_xsec(true), m_store_weights(true) {}
  virtual ~Pythia8ToHepMC3() {}

  // The recommended method to convert Pythia events into HepMC3 ones.
  bool fill_next_event( Pythia8::Pythia& pythia, GenEvent* evt,
    int ievnum = -1 ) { return fill_next_event( pythia.event, evt,
    ievnum, &pythia.info, &pythia.settings); }
  bool fill_next_event( Pythia8::Pythia& pythia, GenEvent& evt) {
    return fill_next_event( pythia, &evt); }

  // Alternative method to convert Pythia events into HepMC3 ones.
  bool fill_next_event( Pythia8::Event& pyev, GenEvent&evt, int ievnum = -1,
    const Pythia8::Info* pyinfo = 0, Pythia8::Settings* pyset = 0) {
    return fill_next_event(pyev, &evt, ievnum, pyinfo, pyset); }
  bool fill_next_event( Pythia8::Event& pyev, GenEvent* evt, int ievnum = -1,
    const Pythia8::Info* pyinfo = 0, Pythia8::Settings* pyset = 0) {

    // 1. Error if no event passed.
    if (!evt) {
      std::cerr << "Pythia8ToHepMC3::fill_next_event error - "
                << "passed null event." << std::endl;
      return false;
    }

    // Event number counter.
    if ( ievnum >= 0 ) {
      evt->set_event_number(ievnum);
      m_internal_event_number = ievnum;
    }
    else {
      evt->set_event_number(m_internal_event_number);
      ++m_internal_event_number;
    }

    // Set units to be GeV and mm, to agree with Pythia ones.
    evt->set_units(Units::GEV,Units::MM);

    // 1a. If there is a HIInfo object fill info from that.
    if ( pyinfo && pyinfo->hiInfo ) {
      auto ion = make_shared<HepMC3::GenHeavyIon>();
      ion->Ncoll_hard = pyinfo->hiInfo->nCollNDTot();
      ion->Ncoll = pyinfo->hiInfo->nAbsProj() +
                   pyinfo->hiInfo->nDiffProj() +
                   pyinfo->hiInfo->nAbsTarg() +
                   pyinfo->hiInfo->nDiffTarg() -
                   pyinfo->hiInfo->nCollND() -
                   pyinfo->hiInfo->nCollDD();
      ion->Npart_proj = pyinfo->hiInfo->nAbsProj() +
                        pyinfo->hiInfo->nDiffProj();
      ion->Npart_targ = pyinfo->hiInfo->nAbsTarg() +
                        pyinfo->hiInfo->nDiffTarg();
      ion->impact_parameter = pyinfo->hiInfo->b();
      evt->set_heavy_ion(ion);
    }

    // 2. Fill particle information.
    std::vector<GenParticlePtr> hepevt_particles;
    hepevt_particles.reserve( pyev.size() );
    for(int i = 0; i < pyev.size(); ++i) {
      hepevt_particles.push_back( std::make_shared<GenParticle>(
        FourVector( pyev[i].px(), pyev[i].py(), pyev[i].pz(), pyev[i].e() ),
        pyev[i].id(), pyev[i].statusHepMC() ) );
      hepevt_particles[i]->set_generated_mass( pyev[i].m() );
    }

    // 3. Fill vertex information.
    std::vector<GenVertexPtr> vertex_cache;
    for (int i = 1; i < pyev.size(); ++i) {
      std::vector<int> mothers = pyev[i].motherList();
      if (mothers.size()) {
        GenVertexPtr prod_vtx = hepevt_particles[mothers[0]]->end_vertex();
        if (!prod_vtx) {
          prod_vtx = make_shared<GenVertex>();
          vertex_cache.push_back(prod_vtx);
          for (unsigned int j = 0; j < mothers.size(); ++j)
            prod_vtx->add_particle_in( hepevt_particles[mothers[j]] );
        }
        FourVector prod_pos( pyev[i].xProd(), pyev[i].yProd(),pyev[i].zProd(),
          pyev[i].tProd() );

        // Update vertex position if necessary.
        if (!prod_pos.is_zero() && prod_vtx->position().is_zero())
          prod_vtx->set_position( prod_pos );
        prod_vtx->add_particle_out( hepevt_particles[i] );
      }
    }

    // Reserve memory for the event.
    evt->reserve( hepevt_particles.size(), vertex_cache.size() );

    // Here we assume that the first two particles are the beam particles.
    vector<GenParticlePtr> beam_particles;
    beam_particles.push_back(hepevt_particles[1]);
    beam_particles.push_back(hepevt_particles[2]);

    // Add particles and vertices in topological order.
    evt->add_tree( beam_particles );
    // Attributes should be set after adding the particles to event.
    for (int i = 0; i < pyev.size(); ++i) {
      /* TODO: Set polarization */
      // Colour flow uses index 1 and 2.
      int colType = pyev[i].colType();
      if (colType ==  -1 ||colType ==  1 || colType == 2) {
        int flow1 = 0, flow2 = 0;
        if (colType ==  1 || colType == 2) flow1 = pyev[i].col();
        if (colType == -1 || colType == 2) flow2 = pyev[i].acol();
        hepevt_particles[i]->add_attribute("flow1",
          make_shared<IntAttribute>(flow1));
        hepevt_particles[i]->add_attribute("flow2",
          make_shared<IntAttribute>(flow2));
      }
    }

    // If hadronization switched on then no final coloured particles.
    bool doHadr = (pyset == 0) ? m_free_parton_warnings
      : pyset->flag("HadronLevel:all") && pyset->flag("HadronLevel:Hadronize");

    // 4. Check for particles which come from nowhere, i.e. are without
    // mothers or daughters. These need to be attached to a vertex, or else
    // they will never become part of the event.
    for (int i = 1; i < pyev.size(); ++i) {

      // Check for particles not added to the event.
      // NOTE: We have to check if this step makes any sense in
      // the HepMC event standard.
      if ( !hepevt_particles[i] ) {
        std::cerr << "hanging particle " << i << std::endl;
        GenVertexPtr prod_vtx;
        prod_vtx->add_particle_out( hepevt_particles[i] );
        evt->add_vertex(prod_vtx);
      }

      // Also check for free partons (= gluons and quarks; not diquarks?).
      if ( doHadr && m_free_parton_warnings ) {
        if ( hepevt_particles[i]->pid() == 21
           && !hepevt_particles[i]->end_vertex() ) {
           std::cerr << "gluon without end vertex " << i << std::endl;
           if ( m_crash_on_problem ) exit(1);
        }
        if ( std::abs(hepevt_particles[i]->pid()) <= 6
          && !hepevt_particles[i]->end_vertex() ) {
          std::cerr << "quark without end vertex " << i << std::endl;
          if ( m_crash_on_problem ) exit(1);
        }
      }
    }

    // 5. Store PDF, weight, cross section and other event information.
    // Flavours of incoming partons.
    if (m_store_pdf && pyinfo != 0) {
      int id1pdf = pyinfo->id1pdf();
      int id2pdf = pyinfo->id2pdf();
      if ( m_convert_gluon_to_0 ) {
        if (id1pdf == 21) id1pdf = 0;
        if (id2pdf == 21) id2pdf = 0;
      }

      // Store PDF information.
      GenPdfInfoPtr pdfinfo = make_shared<GenPdfInfo>();
      pdfinfo->set(id1pdf, id2pdf, pyinfo->x1pdf(), pyinfo->x2pdf(),
        pyinfo->QFac(), pyinfo->pdf1(), pyinfo->pdf2() );
      evt->set_pdf_info( pdfinfo );
    }

    // Store process code, scale, alpha_em, alpha_s.
    if (m_store_proc && pyinfo != 0) {
      evt->add_attribute("signal_process_id",
        std::make_shared<IntAttribute>( pyinfo->code()));
      evt->add_attribute("event_scale",
        std::make_shared<DoubleAttribute>(pyinfo->QRen()));
      evt->add_attribute("alphaQCD",
        std::make_shared<DoubleAttribute>(pyinfo->alphaS()));
      evt->add_attribute("alphaQED",
        std::make_shared<DoubleAttribute>(pyinfo->alphaEM()));
    }

    // Store event weights.
    if (m_store_weights && pyinfo != 0) {
      evt->weights().clear();
      for (int iWeight = 0; iWeight < pyinfo->numberOfWeights(); ++iWeight)
        evt->weights().push_back(pyinfo->weightValueByIndex(iWeight));
    }

    // Store cross-section information in pb.
    if (m_store_xsec && pyinfo != 0) {
      // First set atribute to event, such that
      // GenCrossSection::set_cross_section knows how many weights the
      // event has and sets the number of cross sections accordingly.
      GenCrossSectionPtr xsec = make_shared<GenCrossSection>();
      evt->set_cross_section(xsec);
      xsec->set_cross_section( pyinfo->sigmaGen() * 1e9,
        pyinfo->sigmaErr() * 1e9);
      // If multiweights with possibly different xsec, overwrite central value
      vector<double> xsecVec = pyinfo->weightContainerPtr->getTotalXsec();
      if (xsecVec.size() > 0) {
        for (unsigned int iXsec = 0; iXsec < xsecVec.size(); ++iXsec) {
          xsec->set_xsec(iXsec, xsecVec[iXsec]*1e9);
        }
      }
    }

    // Done.
    return true;
  }

  // Read out values for some switches.
  bool print_inconsistency()  const { return m_print_inconsistency; }
  bool free_parton_warnings() const { return m_free_parton_warnings; }
  bool crash_on_problem()     const { return m_crash_on_problem; }
  bool convert_gluon_to_0()   const { return m_convert_gluon_to_0; }
  bool store_pdf()            const { return m_store_pdf; }
  bool store_proc()           const { return m_store_proc; }
  bool store_xsec()           const { return m_store_xsec; }
  bool store_weights()        const { return m_store_weights; }

  // Set values for some switches.
  void set_print_inconsistency(bool b = true)  { m_print_inconsistency  = b; }
  void set_free_parton_warnings(bool b = true) { m_free_parton_warnings = b; }
  void set_crash_on_problem(bool b = false)    { m_crash_on_problem     = b; }
  void set_convert_gluon_to_0(bool b = false)  { m_convert_gluon_to_0   = b; }
  void set_store_pdf(bool b = true)            { m_store_pdf            = b; }
  void set_store_proc(bool b = true)           { m_store_proc           = b; }
  void set_store_xsec(bool b = true)           { m_store_xsec           = b; }
  void set_store_weights(bool b = true)        { m_store_weights        = b; }

private:

  // Following methods are not implemented for this class.
  virtual bool fill_next_event( GenEvent*  )  { return 0; }
  virtual void write_event( const GenEvent* ) {}

  // Use of copy constructor is not allowed.
  Pythia8ToHepMC3( const Pythia8ToHepMC3& ) {}

  // Data members.
  int  m_internal_event_number;
  bool m_print_inconsistency, m_free_parton_warnings, m_crash_on_problem,
       m_convert_gluon_to_0, m_store_pdf, m_store_proc, m_store_xsec,
       m_store_weights;

};

//==========================================================================

} // end namespace HepMC3

namespace Pythia8 {

//==========================================================================
// This a wrapper around HepMC::Pythia8ToHepMC in the Pythia8
// namespace that simplify the most common use cases. It stores the
// current GenEvent and output stream internally to avoid cluttering
// of user code. This class is also defined in HepMC2.h with the same
// signatures, and the user can therefore switch between HepMC version
// 2 and 3, by simply changing the include file.
class Pythia8ToHepMC : public HepMC3::Pythia8ToHepMC3 {

public:

  // We can either have standard ascii output version 2 or three or
  // none at all.
  enum OutputType { none, ascii2, ascii3 };

  // Typedef for the version 3 specific classes used.
  typedef HepMC3::GenEvent GenEvent;
  typedef shared_ptr<GenEvent> EventPtr;
  typedef HepMC3::Writer Writer;
  typedef shared_ptr<Writer> WriterPtr;

  // The empty constructor does not creat an aoutput stream.
  Pythia8ToHepMC() : runinfo(make_shared<HepMC3::GenRunInfo>()) {}

  // Construct an object with an internal output stream.
  Pythia8ToHepMC(string filename, OutputType ft = ascii3)
    : runinfo(make_shared<HepMC3::GenRunInfo>()) {
    setNewFile(filename, ft);
  }

  // Open a new external output stream.
  bool setNewFile(string filename, OutputType ft = ascii3) {
    switch ( ft ) {
    case ascii3:
      writerPtr = make_shared<HepMC3::WriterAscii>(filename);
      break;
    case ascii2:
      writerPtr = make_shared<HepMC3::WriterAsciiHepMC2>(filename);
      break;
    case none:
      break;
    }
    return writerPtr != nullptr;
  }

  // Create a new GenEvent object and fill it with information from
  // the given Pythia object.
  bool fillNextEvent(Pythia & pythia) {
    geneve = make_shared<HepMC3::GenEvent>(runinfo);
    if (runinfo->weight_names().size() == 0)
      setWeightNames(pythia.info.weightNameVector());
    return fill_next_event(pythia, *geneve);
  }

  // Write out the current GenEvent to the internal stream.
  void writeEvent() {
    writerPtr->write_event(*geneve);
  }

  // Create a new GenEvent object and fill it with information from
  // the given Pythia object and write it out directly to the
  // internal stream.
  bool writeNextEvent(Pythia & pythia) {
    if ( !fillNextEvent(pythia) ) return false;
    writeEvent();
    return !writerPtr->failed();
  }

  // Get a reference to the current GenEvent.
  GenEvent & event() {
    return *geneve;
  }

  // Get a pointer to the current GenEvent.
   EventPtr getEventPtr() {
    return geneve;
  }

  // Get a reference to the internal stream.
  Writer & output() {
    return *writerPtr;
  }

  // Get a pointer to the internal stream.
  WriterPtr outputPtr() {
    return writerPtr;
  }

  // Set cross section information in the current GenEvent.
  void setXSec(double xsec, double xsecerr) {
    auto xsecptr = geneve->cross_section();
    if ( !xsecptr ) {
      xsecptr = make_shared<HepMC3::GenCrossSection>();
      geneve->set_cross_section(xsecptr);
    }
    xsecptr->set_cross_section(xsec, xsecerr);
  }

  // Update all weights in the current GenEvent.
  void setWeights(const vector<double> & wv) {
    geneve->weights() = wv;
  }

  // Set all weight names in the current run.
  void setWeightNames(const vector<string> &wnv) {
    runinfo->set_weight_names(wnv);
  }

  // Update the PDF information in the current GenEvent
  void setPdfInfo(int id1, int id2, double x1, double x2,
                  double scale, double xf1, double xf2,
                  int pdf1 = 0, int pdf2 = 0) {
    auto pdf = make_shared<HepMC3::GenPdfInfo>();
    pdf->set(id1, id2, x1, x2, scale, xf1, xf2, pdf1, pdf2);
    geneve->set_pdf_info(pdf);
  }

  // Add an additional attribute derived from HepMC3::Attribute
  // to the current event.
  template<class T>
  void addAtribute(const string& name, T& attribute) {
    shared_ptr<HepMC3::Attribute> att = make_shared<T>(attribute);
    geneve->add_attribute(name, att);
  }

  // Add an attribute of double type.
  template<class T=double>
  void addAttribute(const string& name, double& attribute) {
    auto dAtt = HepMC3::DoubleAttribute(attribute);
    shared_ptr<HepMC3::Attribute> att =
      make_shared<HepMC3::DoubleAttribute>(dAtt);
    geneve->add_attribute(name, att);
  }

  // Remove an attribute from the current event.
  void removeAttribute(const string& name) {
    geneve->remove_attribute(name);
  }

private:

  // The current GenEvent
  EventPtr geneve = nullptr;

  // The output stream.
  WriterPtr writerPtr = nullptr;

  // The current run info.
  shared_ptr<HepMC3::GenRunInfo> runinfo;

};

}

#endif // end Pythia8_HepMC3_H
