
// ProMC file. Google does not like these warnings.
#pragma GCC diagnostic ignored "-pedantic"
#pragma GCC diagnostic ignored "-Wshadow"
#include "ProMCBook.h"

// DIRE includes.
#include "Dire/Dire.h"

// Pythia includes.
#include "Pythia8/Pythia.h"

using namespace Pythia8;

//--------------------------------------------------------------------------

string getEnvVar( std::string const & key ) {
  char * val = getenv( key.c_str() );
  return (val == NULL) ? std::string("") : std::string(val);
}

//--------------------------------------------------------------------------

void readPDG( ProMCHeader * header  ) {

  string temp_string;
  istringstream curstring;
  string PdgTableFilename = getEnvVar("PROMC");
  if (PdgTableFilename.size() < 2) PdgTableFilename = string(PROMC);
  PdgTableFilename += "/data/particle.tbl";

  ifstream file_to_read(PdgTableFilename.c_str());
  if (!file_to_read.good()) {
    cout << "**        ERROR: PDG Table (" << PdgTableFilename
         <<  ") not found! exit.                        **" << endl;
    exit(1);
    return;
  }

  // First three lines of the file are useless.
  getline(file_to_read,temp_string);
  getline(file_to_read,temp_string);
  getline(file_to_read,temp_string);

  while (getline(file_to_read,temp_string)) {
    // Needed when using several times istringstream::str(string).
    curstring.clear();
    curstring.str(temp_string);
    long int ID; std::string name; int charge; float mass; float width;
    float lifetime;
    // ID name   chg       mass    total width   lifetime
    //  1 d      -1      0.33000     0.00000   0.00000E+00
    //  in the table, the charge is in units of e+/3
    //  the total width is in GeV
    //  the lifetime is ctau in mm
    curstring >> ID >> name >> charge >> mass >> width >> lifetime;
    ProMCHeader_ParticleData* pp= header->add_particledata();
    pp->set_id(ID);
    pp->set_mass(mass);
    pp->set_name(name);
    pp->set_width(width);
    pp->set_lifetime(lifetime);
    cout << ID << " " << name << " " << mass << endl;
  }

}

//--------------------------------------------------------------------------

//==========================================================================

int main( int argc, char* argv[] ){

  // Check that correct number of command-line arguments
  if (argc < 3) {
    cerr << " Unexpected number of command-line arguments ("<<argc-1<<"). \n"
         << " You are expected to provide the arguments" << endl
         << " 1. Input file for settings" << endl
         << " 2. Output ProMC file name" << endl
         << argc-1 << " arguments provided:";
         for ( int i=1; i<argc; ++i) cerr << " " << argv[i];
         cerr << "\n Program stopped. " << endl;
    return 1;
  }

  Pythia pythia;

  // Create and initialize DIRE shower plugin.
  Dire dire;
  dire.init(pythia, argv[1]);

  int nEventEst = pythia.settings.mode("Main:numberOfEvents");

  // Switch off all showering and MPI when estimating the cross section,
  // and re-initialise (unfortunately).
  bool fsr = pythia.flag("PartonLevel:FSR");
  bool isr = pythia.flag("PartonLevel:ISR");
  bool mpi = pythia.flag("PartonLevel:MPI");
  bool had = pythia.flag("HadronLevel:all");
  bool rem = pythia.flag("PartonLevel:Remnants");
  bool chk = pythia.flag("Check:Event");
  pythia.settings.flag("PartonLevel:FSR",false);
  pythia.settings.flag("PartonLevel:ISR",false);
  pythia.settings.flag("PartonLevel:MPI",false);
  pythia.settings.flag("HadronLevel:all",false);
  pythia.settings.flag("PartonLevel:Remnants",false);
  pythia.settings.flag("Check:Event",false);
  pythia.init();

  // Cross section estimate run.
  double sumSH = 0.;
  double nAcceptSH = 0.;
  for( int iEvent=0; iEvent<nEventEst; ++iEvent ){
    // Generate next event
    if( !pythia.next() ) {
      if( pythia.info.atEndOfFile() )
        break;
      else continue;
    }
    sumSH     += pythia.info.weight();
    map <string,string> eventAttributes;
    if (pythia.info.eventAttributes)
      eventAttributes = *(pythia.info.eventAttributes);
    string trials = (eventAttributes.find("trials") != eventAttributes.end())
                  ?  eventAttributes["trials"] : "";
    if (trials != "") nAcceptSH += atof(trials.c_str());
  }
  pythia.stat();
  double xs = pythia.info.sigmaGen();
  int nA    = pythia.info.nAccepted();

  // Histogram the weight.
  Hist histWT("weight",100000,-5000.,5000.);

  int nEvent = pythia.settings.mode("Main:numberOfEvents");

  // ****************  book ProMC file **********************

  // Make a soft-link to the proto data.
  std::system("ln -s /nfs/farm/g/theory/qcdsim/sp/hepsoft/PROMC/promc/proto/promc proto");

  //string promcfile = string(argv[2]);
  ProMCBook* epbook = new ProMCBook(argv[2],"w", true);
  epbook->setDescription(nEvent,"PYTHIA8");

  // Info on incoming beams and CM energy.
  ProMCHeader header;
  header.set_id1( pythia.info.idA() );
  header.set_id2( pythia.info.idB() );
  header.set_ecm( pythia.info.eCM() );
  header.set_s( pythia.info.s() );
  header.set_name(pythia.info.name());
  header.set_code(pythia.info.code());

  // Use the range 0.01 MeV to 20 TeV using varints (integers).
  // With particle in GeV, we multiply it by kEV, to get 0.01 MeV = 1 unit.
  const double kEV = 1000*100;
  // With particle in mm, we multiply it by kL to get 0.01 mm = 1 unit.
  const double kL = 100;

  // Set units.
  header.set_momentumunit( (int)kEV );
  header.set_lengthunit( (int)kL );

   // Store a map with PDG information (stored in the header).
  readPDG( &header );
  epbook->setHeader(header); // write header

  // Cross section an error.
  double sigmaTotal  = 0.;
  double errorTotal  = 0.;

  cout << endl << endl << endl;
  cout << "Start generating events" << endl;

  // Switch showering and multiple interaction back on.
  pythia.settings.flag("PartonLevel:FSR",fsr);
  pythia.settings.flag("PartonLevel:ISR",isr);
  pythia.settings.flag("HadronLevel:all",had);
  pythia.settings.flag("PartonLevel:MPI",mpi);
  pythia.settings.flag("PartonLevel:Remnants",rem);
  pythia.settings.flag("Check:Event",chk);
  pythia.init();

  double wmax =-1e15;
  double wmin = 1e15;
  double sumwt = 0.;
  double sumwtsq = 0.;

  // Start generation loop
  for( int iEvent=0; iEvent<nEvent; ++iEvent ){

    // Generate next event
    if( !pythia.next() ) {
      if( pythia.info.atEndOfFile() )
        break;
      else continue;
    }

    // Get event weight(s).
    double evtweight         = pythia.info.weight();

    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    // Retrieve the shower weight.
    dire.weightsPtr->calcWeight(0.);
    dire.weightsPtr->reset();
    double wt = dire.weightsPtr->getShowerWeight();
    evtweight *= wt;

    if (abs(wt) > 1e3) {
      cout << scientific << setprecision(8)
      << "Warning in DIRE main program dire03.cc: Large shower weight wt="
      << wt << endl;
      if (abs(wt) > 1e4) { 
        cout << "Warning in DIRE main program dire03.cc: Shower weight larger"
        << " than 10000. Discard event with rare shower weight fluctuation."
        << endl;
        evtweight = 0.;
      }
    }
    // Do not print zero-weight events.
    if ( evtweight == 0. ) continue;

    wmin = min(wmin,wt);
    wmax = max(wmax,wt);
    sumwt += wt;
    sumwtsq+=pow2(wt);
    histWT.fill( wt, 1.0);

    double norm = xs / double(nA);

    // Weighted events with additional number of trial events to consider.
    if ( pythia.info.lhaStrategy() != 0
      && pythia.info.lhaStrategy() != 3
      && nAcceptSH > 0)
      norm = 1. / (1e9*nAcceptSH);
    // Weighted events.
    else if ( pythia.info.lhaStrategy() != 0
      && pythia.info.lhaStrategy() != 3
      && nAcceptSH == 0)
      norm = 1. / (1e9*nA);

    if(pythia.event.size() > 3){

    // Add the weight of the current event to the cross section.
    sigmaTotal  += evtweight*norm;
    errorTotal  += pow2(evtweight*norm);

    //************  ProMC file ***************//
    ProMCEvent promc;

    // Fill event information.
    ProMCEvent_Event *eve = promc.mutable_event();
    eve->set_number(iEvent);
    eve->set_process_id( pythia.info.code() );     // process ID
    eve->set_scale( pythia.info.pTHat( ));         // relevant for 2 -> 2 only
    eve->set_alpha_qed( pythia.info.alphaEM() );
    eve->set_alpha_qcd( pythia.info.alphaS() );
    eve->set_scale_pdf( pythia.info.QFac() );
    eve->set_x1( pythia.info.x1pdf() );
    eve->set_x2( pythia.info.x2pdf() );
    eve->set_id1( pythia.info.id1pdf() );
    eve->set_id2( pythia.info.id2pdf() );
    eve->set_pdf1( pythia.info.pdf1() );
    eve->set_pdf2( pythia.info.pdf2() );
    eve->set_weight( evtweight*norm );

    // Fill truth particle information, looping over all particles in event.
    ProMCEvent_Particles *pa= promc.mutable_particles();
    for (int i = 0; i < pythia.event.size(); i++) {

      // Fill information particle by particle.
      pa->add_id( i  );
      pa->add_pdg_id( pythia.event[i].id() );
      // Particle status in HEPMC style.
      // pa->add_status(  pythia.event.statusHepMC(i) );
      pa->add_status(  pythia.event[i].status() );
      pa->add_mother1( pythia.event[i].mother1() );
      pa->add_mother2( pythia.event[i].mother2() );
      pa->add_daughter1( pythia.event[i].daughter1() );
      pa->add_daughter2( pythia.event[i].daughter2() );
      // Only store three-momentum and mass, so need to calculate energy.
      pa->add_px( (int)(pythia.event[i].px()*kEV) );
      pa->add_py( (int)(pythia.event[i].py()*kEV) );
      pa->add_pz( (int)(pythia.event[i].pz()*kEV) );
      pa->add_mass( (int)(pythia.event[i].m()*kEV) );
      // Store production vertex; will often be the origin.
      pa->add_x( int(pythia.event[i].xProd()*kL) );
      pa->add_y( int(pythia.event[i].yProd()*kL) );
      pa->add_z( int(pythia.event[i].zProd()*kL) );
      pa->add_t( int(pythia.event[i].tProd()*kL) );

    } // end loop over particles in the event
    epbook->write(promc); // write event

    }

  } // end loop over events to generate

  // print cross section, errors
  pythia.stat();

  cout << scientific << setprecision(6)
       << "\t Minimal shower weight     = " << wmin << "\n"
       << "\t Maximal shower weight     = " << wmax << "\n"
       << "\t Mean shower weight        = " << sumwt/double(nEvent) << "\n"
       << "\t Variance of shower weight = "
       << sqrt(1/double(nEvent)*(sumwtsq - pow(sumwt,2)/double(nEvent)))
       << endl;

  ofstream write;
  // Write histograms to file
  write.open("wt.dat");
  histWT.table(write);
  write.close();

  // Save post-generation statistics for ProMC.
  ProMCStat stat;
  stat.set_cross_section_accumulated( sigmaTotal*1e9 ); // in pb
  stat.set_cross_section_error_accumulated( pythia.info.sigmaErr() * 1e9 );

  stat.set_luminosity_accumulated( double(nEvent)/(sigmaTotal*1e9));
  stat.set_ntried( pythia.info.nTried() );
  stat.set_nselected( pythia.info.nSelected() );
  stat.set_naccepted( pythia.info.nAccepted() );
  epbook->setStatistics( stat );

  // Close the ProMC file.
  epbook->close(); // close

  // Remove soft-link to the proto data.
  std::system("rm proto");

  // Done
  return 0;

}
