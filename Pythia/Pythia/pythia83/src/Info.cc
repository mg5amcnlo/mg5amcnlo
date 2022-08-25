// Info.cc is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the Info class.

#include "Pythia8/Info.h"
#include "Pythia8/Settings.h"
#include <limits>

namespace Pythia8 {

//==========================================================================

// Info class.
// This class contains a mixed bag of information on the event generation
// activity, especially on the current subprocess properties.

//--------------------------------------------------------------------------

// Constants: could be changed here if desired, but normally should not.
// These are of technical nature, as described for each.

// LHA convention with cross section in pb may require conversion from mb.
const double Info::CONVERTMB2PB = 1e9;

//--------------------------------------------------------------------------

// Initialize settings for error printing.

void Info::init() {
  printErrors = settingsPtr->flag("Print:errors");
}

//--------------------------------------------------------------------------

// List (almost) all information currently set.

void Info::list() const {

  // Header and beam info.
  cout << "\n --------  PYTHIA Info Listing  ------------------------"
       << "---------------- \n \n"
       << scientific << setprecision(3)
       << " Beam A: id = " << setw(6) << idASave << ", pz = " << setw(10)
       << pzASave << ", e = " << setw(10) << eASave << ", m = " << setw(10)
       << mASave << ".\n"
       << " Beam B: id = " << setw(6) << idBSave << ", pz = " << setw(10)
       << pzBSave << ", e = " << setw(10) << eBSave << ", m = " << setw(10)
       << mBSave << ".\n\n";

  // Done if no subprocess has been defined.
  if (codeSave == 0 && nFinalSave == 0) {
    cout << " No process has been set; something must have gone wrong! \n"
         << "\n --------  End PYTHIA Info Listing  --------------------"
         << "----------------" << endl;
    return;
  }

  // Colliding parton info.
  if (isRes) {
    cout << " In 1: id = " << setw(4) << id1pdfSave[0] << ", x = "
         << setw(10) << x1pdfSave[0] << ", pdf = " << setw(10) << pdf1Save[0]
         << " at Q2 = " << setw(10) << Q2FacSave[0] << ".\n"
         << " In 2: id = " << setw(4) << id2pdfSave[0] << ", x = "
         << setw(10) << x2pdfSave[0] << ", pdf = " << setw(10) << pdf2Save[0]
         << " at same Q2.\n";
    bool matchIdX = true;
    if (id1pdfSave[0] != id1Save[0] || id2pdfSave[0] != id2Save[0])
      matchIdX = false;
    if (abs(x1pdfSave[0] - x1Save[0]) > 1e-4 * x1Save[0]) matchIdX = false;
    if (abs(x2pdfSave[0] - x2Save[0]) > 1e-4 * x2Save[0]) matchIdX = false;
    if (!matchIdX) cout << " Warning: above flavour/x info does not match"
         << " incoming partons in event!\n";
    cout << "\n";
  }

  // Process name and code.
  cout << ((isRes && !hasSubSave[0]) ? " Subprocess " : " Process ")
       << nameSave << " with code " << codeSave << " is 2 -> "
       << nFinalSave << ".\n";

  // Subprocess name and code for nondiffractive processes.
  if (hasSubSave[0])
    cout << " Subprocess " << nameSubSave[0] << " with code " << codeSubSave[0]
         << " is 2 -> " << nFinalSubSave[0] << ".\n";

  // Process-type-specific kinematics information.
  if ( isRes && nFinalSave == 1)
    cout << " It has sHat = " << setw(10) << sH[0] << ".\n";
  else if ( isRes && nFinalSave == 2)
    cout << " It has sHat = " << setw(10) << sH[0] << ",    tHat = "
         << setw(10) << tH[0] << ",    uHat = " << setw(10) << uH[0] << ",\n"
         << "       pTHat = " << setw(10) << pTH[0] << ",   m3Hat = "
         << setw(10) << m3H[0] << ",   m4Hat = " << setw(10) << m4H[0] << ",\n"
         << "    thetaHat = " << setw(10) << thetaH[0] << ",  phiHat = "
         << setw(10) << phiH[0] << ".\n";
    else if ( nFinalSave == 2)
    cout << " It has s = " << setw(10) << sH[0] << ",    t = " << setw(10)
         << tH[0] << ",    u = " << setw(10) << uH[0] << ",\n"
         << "       pT = " << setw(10) << pTH[0] << ",   m3 = " << setw(10)
         << m3H[0] << ",   m4 = " << setw(10) << m4H[0] << ",\n"
         << "    theta = " << setw(10) << thetaH[0] << ",  phi = " << setw(10)
         << phiH[0] << ".\n";
  else if ( isRes && nFinalSave == 3)
    cout << " It has sHat = " << setw(10) << sH[0] << ", <pTHat> = "
         << setw(10) << pTH[0] << ".\n";
  else if ( nFinalSave == 3)
    cout << " It has s = " << setw(10) << sH[0] << ",  t_A = " << setw(10)
         << tH[0] << ",  t_B = " << setw(10) << uH[0] << ",\n"
         << "     <pT> = " << setw(10) << pTH[0] << ".\n";

  // Couplings.
  if (isRes) cout << "     alphaEM = " << setw(10) << alphaEMSave[0]
    << ",  alphaS = " << setw(10) << alphaSSave[0] << "    at Q2 = "
    << setw(10) << Q2RenSave[0] << ".\n";

  // Diffractive subsystems.
  for (int iDS = 1; iDS < 4; ++iDS) if (id1Save[iDS] != 0) {
    if (iDS == 1) cout << "\n Diffractive system on side A: \n";
    if (iDS == 2) cout << "\n Diffractive system on side B: \n";
    if (iDS == 3) cout << "\n Central diffractive system: \n";
    cout << " In 1: id = " << setw(4) << id1pdfSave[iDS] << ", x = "
         << setw(10) << x1pdfSave[iDS] << ", pdf = " << setw(10)
         << pdf1Save[iDS] << " at Q2 = " << setw(10) << Q2FacSave[iDS]
         << ".\n" << " In 2: id = " << setw(4) << id2pdfSave[iDS]
         << ", x = " << setw(10) << x2pdfSave[iDS] << ", pdf = "
         << setw(10) << pdf2Save[iDS] << " at same Q2.\n";
    cout << " Subprocess " << nameSubSave[iDS] << " with code "
         << codeSubSave[iDS] << " is 2 -> " << nFinalSubSave[iDS] << ".\n";
    if (nFinalSubSave[iDS] == 1) {
      cout << " It has sHat = " << setw(10) << sH[iDS] << ".\n";
    } else if (nFinalSubSave[iDS] == 2) {
      cout << " It has sHat = " << setw(10) << sH[iDS] << ",    tHat = "
           << setw(10) << tH[iDS] << ",    uHat = " << setw(10) << uH[iDS]
           << ",\n" << "       pTHat = " << setw(10) << pTH[iDS]
           << ",   m3Hat = " << setw(10) << m3H[iDS] << ",   m4Hat = "
           << setw(10) << m4H[iDS] << ",\n" << "    thetaHat = " << setw(10)
           << thetaH[iDS] << ",  phiHat = "  << setw(10) << phiH[iDS] << ".\n";
    }
    cout << "     alphaEM = " << setw(10) << alphaEMSave[iDS]
         << ",  alphaS = " << setw(10) << alphaSSave[iDS] << "    at Q2 = "
         << setw(10) << Q2RenSave[iDS] << ".\n";
  }

  // Impact parameter.
  if (bIsSet) cout << "\n Impact parameter b = " << setw(10) << bMPISave
    << " gives enhancement factor = " << setw(10) << enhanceMPISave
    << ".\n";

  // Multiparton interactions and shower evolution.
  if (evolIsSet) cout << " Max pT scale for MPI = " << setw(10) << pTmaxMPISave
    << ", ISR = " << setw(10) << pTmaxISRSave << ", FSR = " << setw(10)
    << pTmaxISRSave << ".\n Number of MPI = " << setw(5) << nMPISave
    << ", ISR = " << setw(5) << nISRSave << ", FSRproc = " << setw(5)
    << nFSRinProcSave << ", FSRreson = " << setw(5) << nFSRinResSave
    << ".\n";

  // Listing finished.
  cout << "\n --------  End PYTHIA Info Listing  --------------------"
       << "----------------" << endl;

}

//--------------------------------------------------------------------------

// Event weights and accumulated weight.

double Info::weight(int iWeight) const {
  double wt = weightContainerPtr->weightNominal;
  if (iWeight < 0 ||
    iWeight >= int(weightContainerPtr->weightsShowerPtr->getWeightsSize()))
    return wt;
  else
    return wt * weightContainerPtr->weightsShowerPtr->getWeightsValue(iWeight);
}

double Info::weightSum() const {
  return (abs(lhaStrategySave) == 4) ? CONVERTMB2PB * wtAccSum : wtAccSum;
}

//--------------------------------------------------------------------------

// List of all hard processes switched on.

vector<int> Info::codesHard() {
  vector<int> codesNow;
  for (map<int, long>::iterator nTryEntry = nTryM.begin();
    nTryEntry != nTryM.end(); ++nTryEntry)
      codesNow.push_back( nTryEntry->first );
  return codesNow;
}

//--------------------------------------------------------------------------

// Print a message the first few times. Insert in database.

void Info::errorMsg(string messageIn, string extraIn, bool showAlways) {

  // Recover number of times message occured. Also inserts new string.
  int times = messages[messageIn];
  ++messages[messageIn];

  // Print message the first time.
  if ((times == 0 || showAlways) && printErrors) cout << " PYTHIA "
    << messageIn << " " << extraIn << endl;

}

//--------------------------------------------------------------------------

// Add all errors from the other Info object to the counts of this object.

void Info::errorCombine(const Info& other) {
  for (pair<string, int> messageEntry : other.messages)
    messages[messageEntry.first] += messageEntry.second;
}

//--------------------------------------------------------------------------

// Provide total number of errors/aborts/warnings experienced to date.

int Info::errorTotalNumber() const {

  int nTot = 0;
  for (pair<string, int> messageEntry : messages)
    nTot += messageEntry.second;
  return nTot;

}

//--------------------------------------------------------------------------

// Print statistics on errors/aborts/warnings.

void Info::errorStatistics() const {

  // Header.
  cout << "\n *-------  PYTHIA Error and Warning Messages Statistics  "
       << "----------------------------------------------------------* \n"
       << " |                                                       "
       << "                                                          | \n"
       << " |  times   message                                      "
       << "                                                          | \n"
       << " |                                                       "
       << "                                                          | \n";

  // Loop over all messages
  map<string, int>::const_iterator messageEntry = messages.begin();
  if (messageEntry == messages.end())
    cout << " |      0   no errors or warnings to report              "
         << "                                                          | \n";
  while (messageEntry != messages.end()) {
    // Message printout.
    string temp = messageEntry->first;
    int len = temp.length();
    temp.insert( len, max(0, 102 - len), ' ');
    cout << " | " << setw(6) << messageEntry->second << "   "
         << temp << " | \n";
    ++messageEntry;
  }

  // Done.
  cout << " |                                                       "
       << "                                                          | \n"
       << " *-------  End PYTHIA Error and Warning Messages Statistics"
       << "  ------------------------------------------------------* "
       << endl;

}

//--------------------------------------------------------------------------

// Return a list of all header key names

vector<string> Info::headerKeys() const {
  vector<string> keys;
  for (pair<string, string> headerEntry : headers)
    keys.push_back(headerEntry.first);
  return keys;
}

//--------------------------------------------------------------------------

// Reset the LHEF3 objects read from the init and header blocks.

void Info::setLHEF3InitInfo() {
  initrwgt     = 0;
  generators   = 0;
  weightgroups = 0;
  init_weights = 0;
  headerBlock  = "";
}

//--------------------------------------------------------------------------

// Set the LHEF3 objects read from the init and header blocks.

void Info::setLHEF3InitInfo( int LHEFversionIn, LHAinitrwgt *initrwgtIn,
  vector<LHAgenerator> *generatorsIn,
  map<string,LHAweightgroup> *weightgroupsIn,
  map<string,LHAweight> *init_weightsIn, string headerBlockIn ) {
  LHEFversionSave = LHEFversionIn;
  initrwgt        = initrwgtIn;
  generators      = generatorsIn;
  weightgroups    = weightgroupsIn;
  init_weights    = init_weightsIn;
  headerBlock     = headerBlockIn;
  weightContainerPtr->weightsLHEF.
    identifyVariationsFromLHAinit( init_weightsIn );
  weightContainerPtr->weightsMerging.setLHEFvariationMapping();
}

//--------------------------------------------------------------------------

// Reset the LHEF3 objects read from the event block.

void Info::setLHEF3EventInfo() {
  eventAttributes    = 0;
  weights_detailed   = 0;
  weights_compressed = 0;
  scales             = 0;
  weights            = 0;
  rwgt               = 0;
  weights_detailed_vector.resize(0);
  eventComments      = "";
  eventWeightLHEF    = 1.0;
  weightContainerPtr->weightsLHEF.clear();
}

//--------------------------------------------------------------------------

// Set the LHEF3 objects read from the event block.

void Info::setLHEF3EventInfo( map<string, string> *eventAttributesIn,
   map<string,double> *weights_detailedIn,
   vector<double> *weights_compressedIn,
   LHAscales *scalesIn, LHAweights *weightsIn,
   LHArwgt *rwgtIn, vector<double> weights_detailed_vecIn,
   vector<string> weights_detailed_name_vecIn,
   string eventCommentsIn, double eventWeightLHEFIn ) {
   eventAttributes    = eventAttributesIn;
   weights_detailed   = weights_detailedIn;
   weights_compressed = weights_compressedIn;
   scales             = scalesIn;
   weights            = weightsIn;
   rwgt               = rwgtIn;
   weights_detailed_vector = weights_detailed_vecIn;
   eventComments      = eventCommentsIn;
   eventWeightLHEF    = eventWeightLHEFIn;
   weightContainerPtr->weightsLHEF.bookVectors(weights_detailed_vecIn,
     weights_detailed_name_vecIn);
}

//--------------------------------------------------------------------------

// Retrieve events tag information.

string Info::getEventAttribute(string key, bool doRemoveWhitespace) const {
  if (!eventAttributes) return "";
  if ( eventAttributes->find(key) != eventAttributes->end() ) {
    string res = (*eventAttributes)[key];
    if (doRemoveWhitespace)
      res.erase (remove (res.begin(), res.end(), ' '), res.end());
    return res;
  }
  return "";
}

//--------------------------------------------------------------------------

// Retrieve initrwgt tag information.

unsigned int Info::getInitrwgtSize() const {
  if (!initrwgt) return 0;
  return initrwgt->weights.size();
}

//--------------------------------------------------------------------------

// Retrieve generator tag information.

unsigned int Info::getGeneratorSize() const {
  if (!generators) return 0;
  return generators->size();
}

string Info::getGeneratorValue(unsigned int n) const {
  if (!generators || generators->size() < n+1) return "";
  return (*generators)[n].contents;
}

string Info::getGeneratorAttribute( unsigned int n, string key,
  bool doRemoveWhitespace) const {
  if (!generators || generators->size() < n+1) return "";
  string res("");
  if ( key == "name") {
    res = (*generators)[n].name;
  } else if ( key == "version") {
    res = (*generators)[n].version;
  } else if ( (*generators)[n].attributes.find(key)
           != (*generators)[n].attributes.end() ) {
    res = (*generators)[n].attributes[key];
  }
  if (doRemoveWhitespace && res != "")
    res.erase (remove (res.begin(), res.end(), ' '), res.end());
  return res;
}

//--------------------------------------------------------------------------

// Retrieve rwgt tag information.

unsigned int Info::getWeightsDetailedSize() const {
  if (!weights_detailed) return 0;
  return weights_detailed->size();
}

double Info::getWeightsDetailedValue(string n) const {
  if (weights_detailed->empty()
    || weights_detailed->find(n) == weights_detailed->end())
    return numeric_limits<double>::quiet_NaN();
  return (*weights_detailed)[n];
}

string Info::getWeightsDetailedAttribute(string n, string key,
  bool doRemoveWhitespace) const {
  if (!rwgt || rwgt->wgts.find(n) == rwgt->wgts.end())
    return "";
  string res("");
  if ( key == "id") {
    res = rwgt->wgts[n].id;
  } else if ( rwgt->wgts[n].attributes.find(key)
           != rwgt->wgts[n].attributes.end() ) {
    res = rwgt->wgts[n].attributes[key];
  }
  if (doRemoveWhitespace && res != "")
    res.erase (remove (res.begin(), res.end(), ' '), res.end());
  return res;
}

//--------------------------------------------------------------------------

// Retrieve weights tag information.

unsigned int Info::getWeightsCompressedSize() const {
  if (!weights_compressed) return 0;
  return weights_compressed->size();
}

double Info::getWeightsCompressedValue(unsigned int n) const {
  if (weights_compressed->empty() || weights_compressed->size() < n+1)
    return numeric_limits<double>::quiet_NaN();
  return (*weights_compressed)[n];
}

string Info::getWeightsCompressedAttribute(string key,
  bool doRemoveWhitespace) const {
  if (!weights || weights->attributes.find(key) == weights->attributes.end())
    return "";
  string res("");
  if ( weights->attributes.find(key)
           != weights->attributes.end() ) {
    res = weights->attributes[key];
  }
  if (doRemoveWhitespace && res != "")
    res.erase (remove (res.begin(), res.end(), ' '), res.end());
  return res;
}

//--------------------------------------------------------------------------

// Retrieve scales tag information.

string Info::getScalesValue(bool doRemoveWhitespace) const {
  if (!scales) return "";
  string res = scales->contents;
  if (doRemoveWhitespace && res != "")
    res.erase (remove (res.begin(), res.end(), ' '), res.end());
  return res;
}

double Info::getScalesAttribute(string key) const {
  if (!scales) return numeric_limits<double>::quiet_NaN();
  double res = numeric_limits<double>::quiet_NaN();
  if ( key == "muf") {
    res = scales->muf;
  } else if ( key == "mur") {
    res = scales->mur;
  } else if ( key == "mups") {
    res = scales->mups;
  } else if ( key == "SCALUP") {
    res = scales->SCALUP;
  } else if ( scales->attributes.find(key)
           != scales->attributes.end() ) {
    res = scales->attributes[key];
  }
  return res;
}

//--------------------------------------------------------------------------

// Move process information to an another diffractive system.

void Info::reassignDiffSystem( int iDSold, int iDSnew) {
  id1Save[iDSnew]       = id1Save[iDSold];       id1Save[iDSold]       = 0;
  id2Save[iDSnew]       = id2Save[iDSold];       id2Save[iDSold]       = 0;
  x1Save[iDSnew]        = x1Save[iDSold];        x1Save[iDSold]        = 0.;
  x2Save[iDSnew]        = x2Save[iDSold];        x2Save[iDSold]        = 0.;
  id1pdfSave[iDSnew]    = id1pdfSave[iDSold];    id1pdfSave[iDSold]    = 0;
  id2pdfSave[iDSnew]    = id2pdfSave[iDSold];    id2pdfSave[iDSold]    = 0;
  x1pdfSave[iDSnew]     = x1pdfSave[iDSold];     x1pdfSave[iDSold]     = 0.;
  x2pdfSave[iDSnew]     = x2pdfSave[iDSold];     x2pdfSave[iDSold]     = 0.;
  pdf1Save[iDSnew]      = pdf1Save[iDSold];      pdf1Save[iDSold]      = 0.;
  pdf2Save[iDSnew]      = pdf2Save[iDSold];      pdf2Save[iDSold]      = 0.;
  Q2RenSave[iDSnew]     = Q2RenSave[iDSold];     Q2RenSave[iDSold]     = 0.;
  Q2FacSave[iDSnew]     = Q2FacSave[iDSold];     Q2FacSave[iDSold]     = 0.;
  alphaEMSave[iDSnew]   = alphaEMSave[iDSold];   alphaEMSave[iDSold]   = 0.;
  alphaSSave[iDSnew]    = alphaSSave[iDSold];    alphaSSave[iDSold]    = 0.;
  scalupSave[iDSnew]    = scalupSave[iDSold];    scalupSave[iDSold]    = 0.;
  sH[iDSnew]            = sH[iDSold];            sH[iDSold]            = 0.;
  tH[iDSnew]            = tH[iDSold];            tH[iDSold]            = 0.;
  uH[iDSnew]            = uH[iDSold];            uH[iDSold]            = 0.;
  pTH[iDSnew]           = pTH[iDSold];           pTH[iDSold]           = 0.;
  m3H[iDSnew]           = m3H[iDSold];           m3H[iDSold]           = 0.;
  m4H[iDSnew]           = m4H[iDSold];           m4H[iDSold]           = 0.;
  thetaH[iDSnew]        = thetaH[iDSold];        thetaH[iDSold]        = 0.;
  phiH[iDSnew]          = phiH[iDSold];          phiH[iDSold]          = 0.;
  hasSubSave[iDSnew]    = hasSubSave[iDSold];    hasSubSave[iDSold]    = false;
  nameSubSave[iDSnew]   = nameSubSave[iDSold];   nameSubSave[iDSold]   = "";
  codeSubSave[iDSnew]   = codeSubSave[iDSold];   codeSubSave[iDSold]   = 0;
  nFinalSubSave[iDSnew] = nFinalSubSave[iDSold]; nFinalSubSave[iDSold] = 0;
}

//==========================================================================

// Class for loading plugin libraries at run time.

//--------------------------------------------------------------------------

// Constructor, with library name and info pointer.

Plugin::Plugin(string nameIn, Info *infoPtrIn) {
  name = nameIn;
  infoPtr = infoPtrIn;
  libPtr = dlopen(nameIn.c_str(), RTLD_LAZY);
  const char* cerror = dlerror();
  string serror(cerror == nullptr ? "" : cerror);
  dlerror();
  if (serror.size()) {
    errorMsg("Error in Plugin::Plugin: " + serror);
    libPtr = nullptr;
  }

}

//--------------------------------------------------------------------------

// Destructor.

Plugin::~Plugin() {
  if (libPtr != nullptr) dlclose(libPtr);
  dlerror();

}

//--------------------------------------------------------------------------

// Access plugin library symbols.

Plugin::Symbol Plugin::symbol(string symName) {
    Symbol sym(0);
    const char* error(0);
    if (libPtr == nullptr) return sym;
    sym = (Symbol)dlsym(libPtr, symName.c_str());
    error = dlerror();
    if (error) errorMsg("Error in Plugin::symbol: " + string(error));
    dlerror();
    return sym;

}

//==========================================================================

} // end namespace Pythia8
