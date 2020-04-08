// -*- C++ -*-
//
// This file is part of LHAPDF
// Copyright (C) 2012-2016 The LHAPDF collaboration (see AUTHORS for details)
//
#include "LHAPDF/PDF.h"
#include "LHAPDF/PDFSet.h"
#include "LHAPDF/PDFIndex.h"
#include "LHAPDF/Factories.h"
#include "LHAPDF/Utils.h"
#include "LHAPDF/Paths.h"
#include "LHAPDF/Version.h"
#include "LHAPDF/LHAGlue.h"
#include <cstring>

using namespace std;


// We have to create and initialise some common blocks here for backwards compatibility
struct w50512 {
  double qcdl4, qcdl5;
};
w50512 w50512_;

struct w50513 {
  double xmin, xmax, q2min, q2max;
};
w50513 w50513_;

struct lhapdfr {
  double qcdlha4, qcdlha5;
  int nfllha;
};
lhapdfr lhapdfr_;



namespace   { //< Unnamed namespace to restrict visibility to this file


  /// @brief PDF object storage here is a smart pointer to ensure deletion of created PDFs
  typedef std::shared_ptr<LHAPDF::PDF> PDFPtr;


  /// @brief A struct for handling the active PDFs for the Fortran interface.
  ///
  /// We operate in a string-based way, since maybe there will be sets with names, but no
  /// index entry in pdfsets.index.
  ///
  /// @todo Can we avoid the strings and just work via the LHAPDF ID and factory construction?
  ///
  /// Smart pointers are used in the native map used for PDF member storage so
  /// that they auto-delete if the PDFSetHandler that holds them goes out of
  /// scope (i.e. is overwritten).
  struct PDFSetHandler {

    /// Default constructor
    ///
    /// It'll be stored in a map so we need one of these...
    PDFSetHandler() : currentmem(0)
    { }

    /// Constructor from a PDF set name
    ///
    /// @note If the set name contains a member specification, i.e. myname/2,
    /// that member rather than the central one will be initialised and made
    /// current.
    PDFSetHandler(const string& name) {
      pair<string, int> set_mem = LHAPDF::lookupPDF(name);
      // First check that the lookup was successful, i.e. it was a valid ID for the LHAPDF6 set collection
      if (set_mem.first.empty() || set_mem.second < 0)
        throw LHAPDF::UserError("Could not find a valid PDF with string = " + name);
      // Try to load this PDF
      setname = set_mem.first;
      loadMember(set_mem.second);
    }

    /// Constructor from a PDF set's LHAPDF ID code
    ///
    /// @note The set member given by the ID (rather than the central one) will
    /// be initialised and made current.
    PDFSetHandler(int lhaid) {
      pair<string,int> set_mem = LHAPDF::lookupPDF(lhaid);
      // First check that the lookup was successful, i.e. it was a valid ID for the LHAPDF6 set collection
      if (set_mem.first.empty() || set_mem.second < 0)
        throw LHAPDF::UserError("Could not find a valid PDF with LHAPDF ID = " + LHAPDF::to_str(lhaid));
      // Try to load this PDF
      setname = set_mem.first;
      loadMember(set_mem.second);
    }

    /// @brief Load a new PDF member, set it to be active
    ///
    /// If it's already loaded, the existing object will not be reloaded.
    void loadMember(int mem) {
      if (mem < 0)
        throw LHAPDF::UserError("Tried to load a negative PDF member ID: " + LHAPDF::to_str(mem) + " in set " + setname);
      if (members.find(mem) == members.end())
        members[mem] = PDFPtr(LHAPDF::mkPDF(setname, mem));
      currentmem = mem;
      //return members[mem];
    }

    /// Actively delete a PDF member to save memory, set the active member to be the next available, or 0
    void unloadMember(int mem) {
      members.erase(mem);
      const int nextmem = (!members.empty()) ? members.begin()->first : 0;
      loadMember(nextmem);
    }

    /// @brief Get a PDF member, making it active
    ///
    /// Non-const because it can secretly load the member. Not that constness
    /// matters in a Fortran interface utility function!
    const PDFPtr member(int mem) {
      loadMember(mem);
      return members.find(mem)->second;
    }

    /// Get the currently active PDF member
    ///
    /// Non-const because it can secretly load the member. Not that constness
    /// matters in a Fortran interface utility function!
    const PDFPtr activeMember() {
      return member(currentmem);
    }

    /// Get the currently active PDF member
    ///
    /// Non-const because it can secretly load the member. Not that constness
    /// matters in a Fortran interface utility function!
    void setActiveMember(int mem) {
      loadMember(mem);
    }

    /// The currently active member in this set
    int currentmem;

    /// Name of this set
    string setname;

    /// Map of pointers to selected member PDFs
    ///
    // /// It's mutable so that a "const" member-getting operation can implicitly
    // /// load a new PDF object. Good idea / bad idea? Disabled for now.
    // mutable map<int, PDFPtr> members;
    map<int, PDFPtr> members;
  };


  /// Collection of active sets
  static map<int, PDFSetHandler> ACTIVESETS;

  /// The currently active set
  int CURRENTSET = 0;

}



string lhaglue_get_current_pdf(int nset) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    return "NONE";
  CURRENTSET = nset;
  return ACTIVESETS[nset].activeMember()->set().name() + " (" +
    LHAPDF::to_str(ACTIVESETS[nset].activeMember()->lhapdfID()) + ")";
}



namespace {


  /// C-string -> Fortran-string converter
  ///
  /// Credit: https://stackoverflow.com/questions/10163485/passing-char-arrays-from-c-to-fortran
  void cstr_to_fstr(const char* cstring, char* fstring, std::size_t fstring_len) {
    std::size_t inlen = std::strlen(cstring);
    std::size_t cpylen = std::min(inlen, fstring_len);
    // TODO: truncation error or warning
    //if (inlen > fstring_len) FOOOOO();
    std::copy(cstring, cstring+cpylen, fstring);
    std::fill(fstring+cpylen, fstring+fstring_len, ' ');
  }


  /// C++-string -> Fortran-string converter
  void ccstr_to_fstr(const string& ccstring, char* fstring, std::size_t fstring_len) {
    const char* cstring = ccstring.c_str();
    cstr_to_fstr(cstring, fstring, fstring_len);
  }


  /// Fortran-string -> C++-string converter
  string fstr_to_ccstr(const char* fstring, const std::size_t fstring_len, bool spcpad=false) {
    // Allocate space for an equivalent C-string (with an extra terminating null byte)
    char* s = new char[fstring_len+1];
    // Copy all characters and add the terminating null byte
    strncpy(s, fstring, fstring_len);
    s[fstring_len] = '\0';
    // Replace all trailing spaces with null bytes unless explicitly stopped
    if (!spcpad) {
      for (int i = fstring_len-1; i >= 0; --i) {
        if (s[i] != ' ') break;
        s[i] = '\0';
      }
    }
    string rtn(s); //< copy the result to a C++ string
    delete[] s; //< clean up the dynamic array
    return rtn;
  }


}


extern "C" {


  // NEW FORTRAN INTERFACE FUNCTIONS

  /// Get the LHAPDF library version as a string
  void lhapdf_getversion_(char* s, size_t len) {
    cstr_to_fstr(LHAPDF_VERSION, s, len);
  }


  /// List of available PDF sets, returned as a space-separated string
  void lhapdf_getpdfsetlist_(char* s, size_t len) {
    string liststr;
    for (const string& setname : LHAPDF::availablePDFSets()) {
      if (!liststr.empty()) liststr += " ";
      liststr += setname;
    }
    ccstr_to_fstr(liststr, s, len);
  }


  /// Get PDF data path (colon-separated if there is more than one element)
  void lhapdf_getdatapath_(char* s, size_t len) {
    string pathstr;
    for (const string& path : LHAPDF::paths()) {
      if (!pathstr.empty()) pathstr += ":";
      pathstr += path;
    }
    ccstr_to_fstr(pathstr, s, len);
  }

  /// Set PDF data path(s)
  void lhapdf_setdatapath_(const char* s, size_t len) {
    LHAPDF::setPaths(fstr_to_ccstr(s, len));
  }

  /// Prepend to PDF data path
  void lhapdf_prependdatapath_(const char* s, size_t len) {
    LHAPDF::pathsPrepend(fstr_to_ccstr(s, len));
  }

  /// Append to PDF data path
  void lhapdf_appenddatapath_(const char* s, size_t len) {
    LHAPDF::pathsAppend(fstr_to_ccstr(s, len));
  }


  //------------------


  void lhapdf_initpdfset_byname_(const int& nset, const char* name, int namelength) {
    const string cname = fstr_to_ccstr(name, namelength);
    ACTIVESETS[nset] = PDFSetHandler(cname);
    CURRENTSET = nset;
  }

  void lhapdf_initpdfset_byid_(const int& nset, const int& lhaid) {
    ACTIVESETS[nset] = PDFSetHandler(lhaid);
    CURRENTSET = nset;
  }

  void lhapdf_delpdfset_(const int& nset) {
    ACTIVESETS.erase(nset);
    CURRENTSET = 0;
  }

  void lhapdf_delpdf_(const int& nset, const int& nmem) {
    CURRENTSET = nset;
    ACTIVESETS[CURRENTSET].unloadMember(nmem);
  }


  //------------------


  void lhapdf_hasflavor(const int& nset, const int& nmem, const int& pid, int& rtn) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use set slot " + LHAPDF::to_str(nset) + " but it is not initialised");
    rtn = ACTIVESETS[nset].member(nmem)->hasFlavor(pid) ? 1 : 0;
    // Update current set focus
    CURRENTSET = nset;
  }


  void lhapdf_xfxq2_(const int& nset, const int& nmem, const int& pid, const double& x, const double& q2, double& xf) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use set slot " + LHAPDF::to_str(nset) + " but it is not initialised");
    try {
      xf = ACTIVESETS[nset].member(nmem)->xfxQ2(pid, x, q2);
    } catch (const exception& e) {
      xf = 0;
    }
    // Update current set focus
    CURRENTSET = nset;
  }

  void lhapdf_xfxq_(const int& nset, const int& nmem, const int& pid, const double& x, const double& q, double& xf) {
    const double q2 = q*q;
    lhapdf_xfxq2_(nset, nmem, pid, x, q2, xf);
  }


  void lhapdf_xfxq2_stdpartons_(const int& nset, const int& nmem, const int& pid, const double& x, const double& q2, double* xfs) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    // Evaluate for the 13 LHAPDF5 standard partons (-6..6)
    for (size_t i = 0; i < 13; ++i) {
      try {
        xfs[i] = ACTIVESETS[nset].member(nmem)->xfxQ2(i-6, x, q2);
      } catch (const exception& e) {
        xfs[i] = 0;
      }
    }
    // Update current set focus
    CURRENTSET = nset;
  }

  void lhapdf_xfxq_stdpartons_(const int& nset, const int& nmem, const int& pid, const double& x, const double& q, double* xfs) {
    const double q2 = q*q;
    lhapdf_xfxq2_stdpartons_(nset, nmem, pid, x, q2, xfs);
  }


  //-----------------


  /// Get the alpha_s order for the set
  void lhapdf_getorderas_(const int& nset, const int& nmem, int& oas) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    oas = ACTIVESETS[nset].member(nmem)->info().get_entry_as<int>("AlphaS_OrderQCD");
    // Update current set focus
    CURRENTSET = nset;
  }

  /// Get the alpha_s(Q2) value for set nset
  void lhapdf_alphasq2_(const int& nset, const int& nmem, const double& q2, double& alphas) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    alphas = ACTIVESETS[nset].member(nmem)->alphasQ2(q2);
    // Update current set focus
    CURRENTSET = nset;
  }

  /// Get the alpha_s(Q) value for set nset
  /// @todo Return value rather than return arg? Can we do that elsewhere, too, e.g. single-value PDF xf functions?
  void lhapdf_alphasq_(const int& nset, const int& nmem, const double& q, double& alphas) {
    const double q2 = q*q;
    lhapdf_alphasq2_(nset, nmem, q2, alphas);
  }


  // Metadata functions

  // /// Get the number of error members in the set (with special treatment for single member sets)
  // void numberpdfm_(const int& nset, int& numpdf) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   // Set equal to the number of members  for the requested set
  //   numpdf=  ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("NumMembers");
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // /// Get the max number of active flavours
  // void getnfm_(const int& nset, int& nf) {
  //   //nf = ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("AlphaS_NumFlavors");
  //   nf = ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("NumFlavors");
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // /// Get nf'th quark mass
  // void getqmassm_(const int& nset, const int& nf, double& mass) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   if      (nf*nf ==  1) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MDown");
  //   else if (nf*nf ==  4) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MUp");
  //   else if (nf*nf ==  9) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MStrange");
  //   else if (nf*nf == 16) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MCharm");
  //   else if (nf*nf == 25) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MBottom");
  //   else if (nf*nf == 36) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MTop");
  //   else throw LHAPDF::UserError("Trying to get quark mass for invalid quark ID #" + LHAPDF::to_str(nf));
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // /// Get the nf'th quark threshold
  // void getthresholdm_(const int& nset, const int& nf, double& Q) {
  //   try {
  //     if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //       throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //     if      (nf*nf ==  1) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdDown");
  //     else if (nf*nf ==  4) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdUp");
  //     else if (nf*nf ==  9) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdStrange");
  //     else if (nf*nf == 16) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdCharm");
  //     else if (nf*nf == 25) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdBottom");
  //     else if (nf*nf == 36) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdTop");
  //     //else throw LHAPDF::UserError("Trying to get quark threshold for invalid quark ID #" + LHAPDF::to_str(nf));
  //   } catch (...) {
  //     getqmassm_(nset, nf, Q);
  //   }
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // void getxminm_(const int& nset, const int& nmem, double& xmin) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const int activemem = ACTIVESETS[nset].currentmem;
  //   ACTIVESETS[nset].loadMember(nmem);
  //   xmin = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMin");
  //   ACTIVESETS[nset].loadMember(activemem);
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // void getxmaxm_(const int& nset, const int& nmem, double& xmax) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const int activemem = ACTIVESETS[nset].currentmem;
  //   ACTIVESETS[nset].loadMember(nmem);
  //   xmax = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMax");
  //   ACTIVESETS[nset].loadMember(activemem);
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // void getq2minm_(const int& nset, const int& nmem, double& q2min) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const int activemem = ACTIVESETS[nset].currentmem;
  //   ACTIVESETS[nset].loadMember(nmem);
  //   q2min = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMin"));
  //   ACTIVESETS[nset].loadMember(activemem);
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // void getq2maxm_(const int& nset, const int& nmem, double& q2max) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const int activemem = ACTIVESETS[nset].currentmem;
  //   ACTIVESETS[nset].loadMember(nmem);
  //   q2max = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMax"));
  //   ACTIVESETS[nset].loadMember(activemem);
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }

  // void getminmaxm_(const int& nset, const int& nmem, double& xmin, double& xmax, double& q2min, double& q2max) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const int activemem = ACTIVESETS[nset].currentmem;
  //   ACTIVESETS[nset].loadMember(nmem);
  //   xmin = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMin");
  //   xmax = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMax");
  //   q2min = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMin"));
  //   q2max = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMax"));
  //   ACTIVESETS[nset].loadMember(activemem);
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }


  // /// Backwards compatibility functions for LHAPDF5 calculations of
  // /// PDF uncertainties and PDF correlations (G. Watt, March 2014).

  // // subroutine GetPDFUncTypeM(nset,lMonteCarlo,lSymmetric)
  // void getpdfunctypem_(const int& nset, int& lmontecarlo, int& lsymmetric) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const string errorType = ACTIVESETS[nset].activeMember()->set().errorType();
  //   if (errorType == "replicas") { // Monte Carlo PDF sets
  //     lmontecarlo = 1;
  //     lsymmetric = 1;
  //   } else if (errorType == "symmhessian") { // symmetric eigenvector PDF sets
  //     lmontecarlo = 0;
  //     lsymmetric = 1;
  //   } else { // default: assume asymmetric Hessian eigenvector PDF sets
  //     lmontecarlo = 0;
  //     lsymmetric = 0;
  //   }
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }
  // // subroutine GetPDFUncType(lMonteCarlo,lSymmetric)
  // void getpdfunctype_(int& lmontecarlo, int& lsymmetric) {
  //   int nset1 = 1;
  //   getpdfunctypem_(nset1, lmontecarlo, lsymmetric);
  // }


  // // subroutine GetPDFuncertaintyM(nset,values,central,errplus,errminus,errsym)
  // void getpdfuncertaintym_(const int& nset, const double* values, double& central, double& errplus, double& errminus, double& errsymm) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const size_t nmem = ACTIVESETS[nset].activeMember()->set().size()-1;
  //   const vector<double> vecvalues(values, values + nmem + 1);
  //   LHAPDF::PDFUncertainty err = ACTIVESETS[nset].activeMember()->set().uncertainty(vecvalues, -1);
  //   central = err.central;
  //   errplus = err.errplus;
  //   errminus = err.errminus;
  //   errsymm = err.errsymm;
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }
  // // subroutine GetPDFuncertainty(values,central,errplus,errminus,errsym)
  // void getpdfuncertainty_(const double* values, double& central, double& errplus, double& errminus, double& errsymm) {
  //   int nset1 = 1;
  //   getpdfuncertaintym_(nset1, values, central, errplus, errminus, errsymm);
  // }


  // // subroutine GetPDFcorrelationM(nset,valuesA,valuesB,correlation)
  // void getpdfcorrelationm_(const int& nset, const double* valuesA, const double* valuesB, double& correlation) {
  //   if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //     throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  //   const size_t nmem = ACTIVESETS[nset].activeMember()->set().size()-1;
  //   const vector<double> vecvaluesA(valuesA, valuesA + nmem + 1);
  //   const vector<double> vecvaluesB(valuesB, valuesB + nmem + 1);
  //   correlation = ACTIVESETS[nset].activeMember()->set().correlation(vecvaluesA,vecvaluesB);
  //   // Update current set focus
  //   CURRENTSET = nset;
  // }
  // // subroutine GetPDFcorrelation(valuesA,valuesB,correlation)
  // void getpdfcorrelation_(const double* valuesA, const double* valuesB, double& correlation) {
  //   int nset1 = 1;
  //   getpdfcorrelationm_(nset1, valuesA, valuesB, correlation);
  // }





  //////////////////

  // LHAPDF5 / PDFLIB COMPATIBILITY INTERFACE FUNCTIONS


  // System-level info

  /// LHAPDF library version
  void getlhapdfversion_(char* s, size_t len) {
    // strncpy(s, LHAPDF_VERSION, len);
    cstr_to_fstr(LHAPDF_VERSION, s, len);
  }


  /// Does nothing, only provided for backward compatibility
  void lhaprint_(int& a) {  }


  /// Set LHAPDF parameters
  ///
  /// @note Only the verbosity parameters have any effect: PDF behaviour is not
  /// controlled globally in LHAPDF6.
  void setlhaparm_(const char* par, int parlength) {
    const string cpar = LHAPDF::to_upper(fstr_to_ccstr(par, parlength));
    if (cpar == "NOSTAT" || cpar == "16") {
      cerr << "WARNING: Fortran call to control LHAPDF statistics collection has no effect" << endl;
    } else if (cpar == "LHAPDF" || cpar == "17") {
      cerr << "WARNING: Fortran call to globally control alpha_s calculation has no effect" << endl;
    } else if (cpar == "EXTRAPOLATE" || cpar == "18") {
      cerr << "WARNING: Fortran call to globally control PDF extrapolation has no effect" << endl;
    } else if (cpar == "SILENT" || cpar == "LOWKEY") {
      LHAPDF::setVerbosity(0);
    } else if (cpar == "19") {
      LHAPDF::setVerbosity(1);
    }
  }
  /// Get LHAPDF parameters -- does nothing in LHAPDF6!
  void getlhaparm_(int dummy, char* par, int parlength) {
    cstr_to_fstr("", par, parlength);
  }


  /// Return a dummy max number of sets (there is no limitation now)
  void getmaxnumsets_(int& nmax) {
    nmax = 1000;
  }


  /// Set PDF data path
  void setpdfpath_(const char* s, size_t len) {
    /// @todo Works? Need to check C-string copying, null termination
    char s2[1024];
    s2[len] = '\0';
    strncpy(s2, s, len);
    LHAPDF::pathsPrepend(s2);
  }

  /// Get PDF data path (colon-separated if there is more than one element)
  void getdatapath_(char* s, size_t len) {
    /// @todo Works? Need to check Fortran string return, string macro treatment, etc.
    string pathstr;
    for (const string& path : LHAPDF::paths()) {
      if (!pathstr.empty()) pathstr += ":";
      pathstr += path;
    }
    // strncpy(s, pathstr.c_str(), len);
    cstr_to_fstr(pathstr.c_str(), s, len);
  }


  // PDF initialisation and focus-switching

  /// Load a PDF set
  ///
  /// @todo Does this version actually take a *path*? What to do?
  void initpdfsetm_(const int& nset, const char* setpath, int setpathlength) {
    // Strip file extension for backward compatibility
    string fullp = string(setpath, setpathlength);
    // Remove trailing whitespace
    fullp.erase( std::remove_if( fullp.begin(), fullp.end(), ::isspace ), fullp.end() );
    // Use only items after the last /
    const string pap = LHAPDF::dirname(fullp);
    const string p = LHAPDF::basename(fullp);
    // Prepend path to search area
    LHAPDF::pathsPrepend(pap);
    // Handle extensions
    string path = LHAPDF::file_extn(p).empty() ? p : LHAPDF::file_stem(p);
    /// @note We correct the misnamed CTEQ6L1/CTEQ6ll set name as a backward compatibility special case.
    if (LHAPDF::to_lower(path) == "cteq6ll") path = "cteq6l1";
    // Create the PDF set with index nset
    // if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    if (path != ACTIVESETS[nset].setname)
      ACTIVESETS[nset] = PDFSetHandler(path); ///< @todo Will be wrong if a structured path is given
    CURRENTSET = nset;
  }
  /// Load a PDF set (non-multiset version)
  void initpdfset_(const char* setpath, int setpathlength) {
    int nset1 = 1;
    initpdfsetm_(nset1, setpath, setpathlength);
  }


  /// Load a PDF set by name
  void initpdfsetbynamem_(const int& nset, const char* setname, int setnamelength) {
    // Truncate input to size setnamelength
    string p = setname;
    p.erase(setnamelength, std::string::npos);
    // Strip file extension for backward compatibility
    string name = LHAPDF::file_extn(p).empty() ? p : LHAPDF::file_stem(p);
    // Remove trailing whitespace
    name.erase( std::remove_if( name.begin(), name.end(), ::isspace ), name.end() );
    /// @note We correct the misnamed CTEQ6L1/CTEQ6ll set name as a backward compatibility special case.
    if (LHAPDF::to_lower(name) == "cteq6ll") name = "cteq6l1";
    // Create the PDF set with index nset
    // if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    if (name != ACTIVESETS[nset].setname)
      ACTIVESETS[nset] = PDFSetHandler(name);
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Load a PDF set by name (non-multiset version)
  void initpdfsetbyname_(const char* setname, int setnamelength) {
    int nset1 = 1;
    initpdfsetbynamem_(nset1, setname, setnamelength);
  }


  /// Load a PDF in current set
  void initpdfm_(const int& nset, const int& nmember) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    ACTIVESETS[nset].loadMember(nmember);
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Load a PDF in current set (non-multiset version)
  void initpdf_(const int& nmember) {
    int nset1 = 1;
    initpdfm_(nset1, nmember);
  }


  /// Get the current set number (i.e. allocation slot index)
  void getnset_(int& nset) {
    nset = CURRENTSET;
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  }

  /// Explicitly set the current set number (i.e. allocation slot index)
  void setnset_(const int& nset) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    CURRENTSET = nset;
  }


  /// Get the current member number in slot nset
  void getnmem_(int& nset, int& nmem) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    nmem = ACTIVESETS[nset].currentmem;
    // Update current set focus
    CURRENTSET = nset;
  }

  /// Set the current member number in slot nset
  void setnmem_(const int& nset, const int& nmem) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" +
                              LHAPDF::to_str(nset) + " but it is not initialised");
    ACTIVESETS[nset].loadMember(nmem);
    // Update current set focus
    CURRENTSET = nset;
  }



  // PDF evolution functions

  // NEW BY MZ to evolve one single parton

  /// Get xf(x) values for common partons from current PDF
  void evolvepartm_(const int& nset, const int& ipart, const double& x, const double& q, double& fxq) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    int ipart_copy; // this is to deal with photons, which are labeled 7 in MG5aMC
    ipart_copy = ipart;
    if (ipart==7) ipart_copy = 22;
    try {
      fxq = ACTIVESETS[nset].activeMember()->xfxQ(ipart_copy, x, q);
    } catch (const exception& e) {
      fxq = 0;
    }
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get xf(x) values for common partons from current PDF (non-multiset version)
  void evolvepart_( const int& ipart, const double& x, const double& q, double& fxq) {
    int nset1 = 1;
    evolvepartm_(nset1, ipart, x, q, fxq);
  }

  /// Get xf(x) values for common partons from current PDF
  void evolvepdfm_(const int& nset, const double& x, const double& q, double* fxq) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    // Evaluate for the 13 LHAPDF5 standard partons (-6..6)
    for (size_t i = 0; i < 13; ++i) {
      try {
        fxq[i] = ACTIVESETS[nset].activeMember()->xfxQ(i-6, x, q);
      } catch (const exception& e) {
        fxq[i] = 0;
      }
    }
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get xf(x) values for common partons from current PDF (non-multiset version)
  void evolvepdf_(const double& x, const double& q, double* fxq) {
    int nset1 = 1;
    evolvepdfm_(nset1, x, q, fxq);
  }


  /// Determine if the current PDF has a photon flavour (historically only MRST2004QED)
  /// @todo Function rather than subroutine?
  /// @note There is no multiset version. has_photon will respect the current set slot.
  bool has_photon_() {
    return ACTIVESETS[CURRENTSET].activeMember()->hasFlavor(22);
  }


  /// Get xfx values from current PDF, including an extra photon flavour
  void evolvepdfphotonm_(const int& nset, const double& x, const double& q, double* fxq, double& photonfxq) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    // First evaluate the "normal" partons
    evolvepdfm_(nset, x, q, fxq);
    // Then evaluate the photon flavor (historically only for MRST2004QED)
    try {
      photonfxq = ACTIVESETS[nset].activeMember()->xfxQ(22, x, q);
    } catch (const exception& e) {
      photonfxq = 0;
    }
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get xfx values from current PDF, including an extra photon flavour (non-multiset version)
  void evolvepdfphoton_(const double& x, const double& q, double* fxq, double& photonfxq) {
    int nset1 = 1;
    evolvepdfphotonm_(nset1, x, q, fxq, photonfxq);
  }


  /// Get xf(x) values for common partons from a photon PDF
  void evolvepdfpm_(const int& nset, const double& x, const double& q, const double& p2, const int& ip2, double& fxq) {
    // Update current set focus
    CURRENTSET = nset;
    throw LHAPDF::NotImplementedError("Photon structure functions are not yet supported in LHAPDF6");
  }
  /// Get xf(x) values for common partons from a photon PDF (non-multiset version)
  void evolvepdfp_(const double& x, const double& q, const double& p2, const int& ip2, double& fxq) {
    int nset1 = 1;
    evolvepdfpm_(nset1, x, q, p2, ip2, fxq);
  }


  // alpha_s evolution

  /// Get the alpha_s order for the set
  void getorderasm_(const int& nset, int& oas) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    // Set equal to the number of members for the requested set
    oas = ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("AlphaS_OrderQCD");
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get the alpha_s order for the set (non-multiset version)
  void getorderas_(int& oas) {
    int nset1 = 1;
    getorderasm_(nset1, oas);
  }


  /// Get the alpha_s(Q) value for set nset
  double alphaspdfm_(const int& nset, const double& Q){
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    return ACTIVESETS[nset].activeMember()->alphasQ(Q);
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get the alpha_s(Q) value for the set (non-multiset version)
  double alphaspdf_(const double& Q){
    int nset1 = 1;
    return alphaspdfm_(nset1, Q);
  }


  // Metadata functions

  /// Get the number of error members in the set
  void numberpdfm_(const int& nset, int& numpdf) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    // Set equal to the number of members  for the requested set
    numpdf=  ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("NumMembers");
    // Reproduce old LHAPDF v5 behaviour, i.e. subtract 1
    numpdf -= 1;
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get the number of error members in the set (non-multiset version)
  void numberpdf_(int& numpdf) {
    int nset1 = 1;
    numberpdfm_(nset1, numpdf);
  }


  /// Get the max number of active flavours
  void getnfm_(const int& nset, int& nf) {
    //nf = ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("AlphaS_NumFlavors");
    nf = ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("NumFlavors");
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get the max number of active flavours (non-multiset version)
  void getnf_(int& nf) {
    int nset1 = 1;
    getnfm_(nset1, nf);
  }


  /// Get nf'th quark mass
  void getqmassm_(const int& nset, const int& nf, double& mass) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    if      (nf*nf ==  1) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MDown");
    else if (nf*nf ==  4) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MUp");
    else if (nf*nf ==  9) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MStrange");
    else if (nf*nf == 16) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MCharm");
    else if (nf*nf == 25) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MBottom");
    else if (nf*nf == 36) mass = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("MTop");
    else throw LHAPDF::UserError("Trying to get quark mass for invalid quark ID #" + LHAPDF::to_str(nf));
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get nf'th quark mass (non-multiset version)
  void getqmass_(const int& nf, double& mass) {
    int nset1 = 1;
    getqmassm_(nset1, nf, mass);
  }


  /// Get the nf'th quark threshold
  void getthresholdm_(const int& nset, const int& nf, double& Q) {
    try {
      if (ACTIVESETS.find(nset) == ACTIVESETS.end())
        throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
      if      (nf*nf ==  1) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdDown");
      else if (nf*nf ==  4) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdUp");
      else if (nf*nf ==  9) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdStrange");
      else if (nf*nf == 16) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdCharm");
      else if (nf*nf == 25) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdBottom");
      else if (nf*nf == 36) Q = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("ThresholdTop");
      //else throw LHAPDF::UserError("Trying to get quark threshold for invalid quark ID #" + LHAPDF::to_str(nf));
    } catch (...) {
      getqmassm_(nset, nf, Q);
    }
    // Update current set focus
    CURRENTSET = nset;
  }
  /// Get the nf'th quark threshold
  void getthreshold_(const int& nf, double& Q) {
    int nset1 = 1;
    getthresholdm_(nset1, nf, Q);
  }


  /// Print PDF set's description to stdout
  void getdescm_(const int& nset) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    cout << ACTIVESETS[nset].activeMember()->description() << endl;
    // Update current set focus
    CURRENTSET = nset;
  }
  void getdesc_() {
    int nset1 = 1;
    getdescm_(nset1);
  }


  void getxminm_(const int& nset, const int& nmem, double& xmin) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const int activemem = ACTIVESETS[nset].currentmem;
    ACTIVESETS[nset].loadMember(nmem);
    xmin = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMin");
    ACTIVESETS[nset].loadMember(activemem);
    // Update current set focus
    CURRENTSET = nset;
  }
  void getxmin_(const int& nmem, double& xmin) {
    int nset1 = 1;
    getxminm_(nset1, nmem, xmin);
  }


  void getxmaxm_(const int& nset, const int& nmem, double& xmax) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const int activemem = ACTIVESETS[nset].currentmem;
    ACTIVESETS[nset].loadMember(nmem);
    xmax = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMax");
    ACTIVESETS[nset].loadMember(activemem);
    // Update current set focus
    CURRENTSET = nset;
  }
  void getxmax_(const int& nmem, double& xmax) {
    int nset1 = 1;
    getxmaxm_(nset1, nmem, xmax);
  }


  void getq2minm_(const int& nset, const int& nmem, double& q2min) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const int activemem = ACTIVESETS[nset].currentmem;
    ACTIVESETS[nset].loadMember(nmem);
    q2min = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMin"));
    ACTIVESETS[nset].loadMember(activemem);
    // Update current set focus
    CURRENTSET = nset;
  }
  void getq2min_(const int& nmem, double& q2min) {
    int nset1 = 1;
    getq2minm_(nset1, nmem, q2min);
  }


  void getq2maxm_(const int& nset, const int& nmem, double& q2max) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const int activemem = ACTIVESETS[nset].currentmem;
    ACTIVESETS[nset].loadMember(nmem);
    q2max = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMax"));
    ACTIVESETS[nset].loadMember(activemem);
    // Update current set focus
    CURRENTSET = nset;
  }
  void getq2max_(const int& nmem, double& q2max) {
    int nset1 = 1;
    getq2maxm_(nset1, nmem, q2max);
  }


  void getminmaxm_(const int& nset, const int& nmem, double& xmin, double& xmax, double& q2min, double& q2max) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const int activemem = ACTIVESETS[nset].currentmem;
    ACTIVESETS[nset].loadMember(nmem);
    xmin = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMin");
    xmax = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMax");
    q2min = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMin"));
    q2max = LHAPDF::sqr(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMax"));
    ACTIVESETS[nset].loadMember(activemem);
    // Update current set focus
    CURRENTSET = nset;
  }
  void getminmax_(const int& nmem, double& xmin, double& xmax, double& q2min, double& q2max) {
    int nset1 = 1;
     getminmaxm_(nset1, nmem, xmin, xmax, q2min, q2max);
   }



  void getlam4m_(const int& nset, const int& nmem, double& qcdl4) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    CURRENTSET = nset;
    ACTIVESETS[nset].loadMember(nmem);
    qcdl4 = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("AlphaS_Lambda4", -1.0);
  }
  void getlam4_(const int& nmem, double& qcdl4) {
    int nset1 = 1;
    getlam4m_(nset1, nmem, qcdl4);
  }


  void getlam5m_(const int& nset, const int& nmem, double& qcdl5) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    CURRENTSET = nset;
    ACTIVESETS[nset].loadMember(nmem);
    qcdl5 = ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("AlphaS_Lambda5", -1.0);
  }
  void getlam5_(const int& nmem, double& qcdl5) {
    int nset1 = 1;
    getlam5m_(nset1, nmem, qcdl5);
  }





  /// Backwards compatibility functions for LHAPDF5 calculations of
  /// PDF uncertainties and PDF correlations (G. Watt, March 2014).

  // subroutine GetPDFUncTypeM(nset,lMonteCarlo,lSymmetric)
  void getpdfunctypem_(const int& nset, int& lmontecarlo, int& lsymmetric) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const string errorType = ACTIVESETS[nset].activeMember()->set().errorType();
    if (LHAPDF::startswith(errorType, "replicas")) { // Monte Carlo PDF sets
      lmontecarlo = 1;
      lsymmetric = 1;
    } else if (LHAPDF::startswith(errorType, "symmhessian")) { // symmetric eigenvector PDF sets
      lmontecarlo = 0;
      lsymmetric = 1;
    } else { // default: assume asymmetric Hessian eigenvector PDF sets
      lmontecarlo = 0;
      lsymmetric = 0;
    }
    // Update current set focus
    CURRENTSET = nset;
  }
  // subroutine GetPDFUncType(lMonteCarlo,lSymmetric)
  void getpdfunctype_(int& lmontecarlo, int& lsymmetric) {
    int nset1 = 1;
    getpdfunctypem_(nset1, lmontecarlo, lsymmetric);
  }


  // subroutine GetPDFuncertaintyM(nset,values,central,errplus,errminus,errsym)
  void getpdfuncertaintym_(const int& nset, const double* values, double& central, double& errplus, double& errminus, double& errsymm) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const size_t nmem = ACTIVESETS[nset].activeMember()->set().size()-1;
    const vector<double> vecvalues(values, values + nmem + 1);
    LHAPDF::PDFUncertainty err = ACTIVESETS[nset].activeMember()->set().uncertainty(vecvalues, -1);
    central = err.central;
    // For a combined set, the PDF and parameter variation uncertainties will be added in quadrature.
    errplus = err.errplus;
    errminus = err.errminus;
    errsymm = err.errsymm;
    // Update current set focus
    CURRENTSET = nset;
  }
  // subroutine GetPDFuncertainty(values,central,errplus,errminus,errsym)
  void getpdfuncertainty_(const double* values, double& central, double& errplus, double& errminus, double& errsymm) {
    int nset1 = 1;
    getpdfuncertaintym_(nset1, values, central, errplus, errminus, errsymm);
  }


  // subroutine GetPDFcorrelationM(nset,valuesA,valuesB,correlation)
  void getpdfcorrelationm_(const int& nset, const double* valuesA, const double* valuesB, double& correlation) {
    if (ACTIVESETS.find(nset) == ACTIVESETS.end())
      throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
    const size_t nmem = ACTIVESETS[nset].activeMember()->set().size()-1;
    const vector<double> vecvaluesA(valuesA, valuesA + nmem + 1);
    const vector<double> vecvaluesB(valuesB, valuesB + nmem + 1);
    correlation = ACTIVESETS[nset].activeMember()->set().correlation(vecvaluesA,vecvaluesB);
    // Update current set focus
    CURRENTSET = nset;
  }
  // subroutine GetPDFcorrelation(valuesA,valuesB,correlation)
  void getpdfcorrelation_(const double* valuesA, const double* valuesB, double& correlation) {
    int nset1 = 1;
    getpdfcorrelationm_(nset1, valuesA, valuesB, correlation);
  }


  ///////////////////////////////////////


  /// REALLY OLD PDFLIB COMPATIBILITY FUNCTIONS

  /// PDFLIB initialisation function
  void pdfset_(const char* par, const double* value, int parlength) {

    string my_par(par), message;
    int id;
    // Identify the calling program (yuck!)
    if (my_par.find("NPTYPE") != string::npos) {
      message = "==== LHAPDF6 USING PYTHIA-TYPE LHAGLUE INTERFACE ====";
      // Take PDF ID from value[2]
      id = value[2]+1000*value[1];
    } else if (my_par.find("HWLHAPDF") != string::npos) {
      message = "==== LHAPDF6 USING HERWIG-TYPE LHAGLUE INTERFACE ====";
      // Take PDF ID from value[0]
      id = value[0];
    } else if (my_par.find("DEFAULT") != string::npos) {
      message = "==== LHAPDF6 USING DEFAULT-TYPE LHAGLUE INTERFACE ====";
      // Take PDF ID from value[0]
      id = value[0];
    } else {
      message = "==== LHAPDF6 USING PDFLIB-TYPE LHAGLUE INTERFACE ====";
      // Take PDF ID from value[2]
      id = value[2]+1000*value[1];
    }
    pair<string, int> set_id = LHAPDF::lookupPDF(id);
    if (set_id.first != ACTIVESETS[1].setname || set_id.second != ACTIVESETS[1].currentmem) {
      if (LHAPDF::verbosity() > 0) cout << message << endl;
      ACTIVESETS[1] = PDFSetHandler(id);
    }

    CURRENTSET = 1;

    // Extract parameters for common blocks (with sensible fallback values)
    PDFPtr pdf = ACTIVESETS[1].activeMember();
    w50513_.xmin = pdf->info().get_entry_as<double>("XMin", 0.0);
    w50513_.xmax = pdf->info().get_entry_as<double>("XMax", 1.0);
    w50513_.q2min = LHAPDF::sqr(pdf->info().get_entry_as<double>("QMin", 1.0));
    w50513_.q2max = LHAPDF::sqr(pdf->info().get_entry_as<double>("QMax", 1.0e5));
    w50512_.qcdl4 = pdf->info().get_entry_as<double>("AlphaS_Lambda4", 0.0);
    w50512_.qcdl5 = pdf->info().get_entry_as<double>("AlphaS_Lambda5", 0.0);
    lhapdfr_.qcdlha4 = pdf->info().get_entry_as<double>("AlphaS_Lambda4", 0.0);
    lhapdfr_.qcdlha5 = pdf->info().get_entry_as<double>("AlphaS_Lambda5", 0.0);
    lhapdfr_.nfllha = 4;
    // Activate legacy/compatibility LHAPDF5-type behaviour re. broken Lambda values
    if (pdf->info().get_entry_as<bool>("Pythia6LambdaV5Compat", true)) {
      w50512_.qcdl4 = 0.192;
      w50512_.qcdl5 = 0.192;
      lhapdfr_.qcdlha4 = 0.192;
      lhapdfr_.qcdlha5 = 0.192;
    }
  }

  /// PDFLIB nucleon structure function querying
  void structm_(const double& x, const double& q,
                double& upv, double& dnv, double& usea, double& dsea,
                double& str, double& chm, double& bot, double& top, double& glu) {
    CURRENTSET = 1;
    /// Fill (partial) parton return variables
    PDFPtr pdf = ACTIVESETS[1].activeMember();
    dsea = pdf->xfxQ(-1, x, q);
    usea = pdf->xfxQ(-2, x, q);
    dnv = pdf->xfxQ(1, x, q) - dsea;
    upv = pdf->xfxQ(2, x, q) - usea;
    str = pdf->xfxQ(3, x, q);
    chm = (pdf->hasFlavor(4)) ? pdf->xfxQ(4, x, q) : 0;
    bot = (pdf->hasFlavor(5)) ? pdf->xfxQ(5, x, q) : 0;
    top = (pdf->hasFlavor(6)) ? pdf->xfxQ(6, x, q) : 0;
    glu = pdf->xfxQ(21, x, q);
  }

  /// PDFLIB photon structure function querying
  void structp_(const double& x, const double& q2, const double& p2, const double& ip2,
                double& upv, double& dnv, double& usea, double& dsea,
                double& str, double& chm, double& bot, double& top, double& glu) {
    throw LHAPDF::NotImplementedError("Photon structure functions are not yet supported");
  }

  /// PDFLIB statistics on PDF under/overflows
  void pdfsta_() {
    /// @note Can't do anything...
  }


}


// LHAPDF namespace C++ compatibility code
#ifdef ENABLE_LHAGLUE_CXX


void LHAPDF::setVerbosity(LHAPDF::Verbosity noiselevel) {
  LHAPDF::setVerbosity((int) noiselevel);
}

void LHAPDF::setPDFPath(const string& path) {
  pathsPrepend(path);
}

string LHAPDF::pdfsetsPath() {
  return paths()[0];
}

int LHAPDF::numberPDF() {
  int nmem;
  numberpdf_(nmem);
  return nmem;
}
int LHAPDF::numberPDF(int nset) {
  int nmem;
  numberpdfm_(nset,nmem);
  return nmem;
}

void LHAPDF::initPDF( int memset) {
  int nset1 = 1;
  initpdfm_(nset1, memset);
}
void LHAPDF::initPDF(int nset, int memset) {
  initpdfm_(nset, memset);
}


double LHAPDF::xfx(double x, double Q, int fl) {
  vector<double> r(13);
  evolvepdf_(x, Q, &r[0]);
  return r[fl+6];
}
double LHAPDF::xfx(int nset, double x, double Q, int fl) {
  vector<double> r(13);
  evolvepdfm_(nset, x, Q, &r[0]);
  return r[fl+6];
}

vector<double> LHAPDF::xfx(double x, double Q) {
  vector<double> r(13);
  evolvepdf_(x, Q, &r[0]);
  return r;
}
vector<double> LHAPDF::xfx(int nset, double x, double Q) {
  vector<double> r(13);
  evolvepdfm_(nset, x, Q, &r[0]);
  return r;
}

void LHAPDF::xfx(double x, double Q, double* results) {
  evolvepdf_(x, Q, results);
}
void LHAPDF::xfx(int nset, double x, double Q, double* results) {
  evolvepdfm_(nset, x, Q, results);
}


vector<double> LHAPDF::xfxphoton(double x, double Q) {
  vector<double> r(13);
  double mphoton;
  evolvepdfphoton_(x, Q, &r[0], mphoton);
  r.push_back(mphoton);
  return r;
}
vector<double> LHAPDF::xfxphoton(int nset, double x, double Q) {
  vector<double> r(13);
  double mphoton;
  evolvepdfphotonm_(nset, x, Q, &r[0], mphoton);
  r.push_back(mphoton);
  return r;
}

void LHAPDF::xfxphoton(double x, double Q, double* results) {
  evolvepdfphoton_(x, Q, results, results[13]);
}
void LHAPDF::xfxphoton(int nset, double x, double Q, double* results) {
  evolvepdfphotonm_(nset, x, Q, results, results[13]);
}

double LHAPDF::xfxphoton(double x, double Q, int fl) {
  vector<double> r(13);
  double mphoton;
  evolvepdfphoton_(x, Q, &r[0], mphoton);
  if (fl == 7) return mphoton;
  return r[fl+6];
}
double LHAPDF::xfxphoton(int nset, double x, double Q, int fl) {
  vector<double> r(13);
  double mphoton;
  evolvepdfphotonm_(nset, x, Q, &r[0], mphoton);
  if ( fl == 7 ) return mphoton;
  return r[fl+6];
}


void LHAPDF::initPDFSet(const string& filename, int nmem) {
  initPDFSet(1,filename, nmem);
}

void LHAPDF::initPDFSet(int nset, const string& filename, int nmem) {
  initPDFSetByName(nset,filename);
  ACTIVESETS[nset].loadMember(nmem);
  CURRENTSET = nset;
}


void LHAPDF::initPDFSet(const string& filename, SetType type, int nmem) {
  // silently ignore type
  initPDFSet(1,filename, nmem);
}

void LHAPDF::initPDFSet(int nset, const string& filename, SetType type, int nmem) {
  // silently ignore type
  initPDFSetByName(nset,filename);
  ACTIVESETS[nset].loadMember(nmem);
  CURRENTSET = nset;
}

void LHAPDF::initPDFSet(int nset, int setid, int nmem) {
  pair<string, int> set_id = LHAPDF::lookupPDF(setid+nmem);
  if (set_id.second != nmem)
    throw LHAPDF::UserError("Inconsistent member numbers: " + LHAPDF::to_str(set_id.second) + " != " + LHAPDF::to_str(nmem));
  if (set_id.first != ACTIVESETS[nset].setname || nmem != ACTIVESETS[nset].currentmem)
      ACTIVESETS[nset] = PDFSetHandler(setid+nmem);
  CURRENTSET = nset;
}

void LHAPDF::initPDFSet(int setid, int nmem) {
  initPDFSet(1,setid,nmem);
}

#define SIZE 999
void LHAPDF::initPDFSetByName(const string& filename) {
  std::cout << "initPDFSetByName: " << filename << std::endl;
  char cfilename[SIZE+1];
  strncpy(cfilename, filename.c_str(), SIZE);
  initpdfsetbyname_(cfilename, filename.length());
}

void LHAPDF::initPDFSetByName(int nset, const string& filename) {
  char cfilename[SIZE+1];
  strncpy(cfilename, filename.c_str(), SIZE);
  initpdfsetbynamem_(nset, cfilename, filename.length());
}

void LHAPDF::initPDFSetByName(const string& filename, SetType type) {
  //silently ignore type
  std::cout << "initPDFSetByName: " << filename << std::endl;
  char cfilename[SIZE+1];
  strncpy(cfilename, filename.c_str(), SIZE);
  initpdfsetbyname_(cfilename, filename.length());
}

void LHAPDF::initPDFSetByName(int nset, const string& filename, SetType type) {
  //silently ignore type
  char cfilename[SIZE+1];
  strncpy(cfilename, filename.c_str(), SIZE);
  initpdfsetbynamem_(nset, cfilename, filename.length());
}


void LHAPDF::getDescription() {
  getDescription(1);
}

void LHAPDF::getDescription(int nset) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  cout << ACTIVESETS[nset].activeMember()->set().description() << endl;
}


double LHAPDF::alphasPDF(double Q) {
  return LHAPDF::alphasPDF(1, Q) ;
}

double LHAPDF::alphasPDF(int nset, double Q) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return alphaS for the requested set
  return ACTIVESETS[nset].activeMember()->alphasQ(Q);
}


bool LHAPDF::hasPhoton(){
  return has_photon_();
}


int LHAPDF::getOrderAlphaS() {
  return LHAPDF::getOrderAlphaS(1) ;
}

int LHAPDF::getOrderAlphaS(int nset) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return alphaS Order for the requested set
  return ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("AlphaS_OrderQCD", -1);
}


int LHAPDF::getOrderPDF() {
  return LHAPDF::getOrderPDF(1) ;
}

int LHAPDF::getOrderPDF(int nset) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return PDF order for the requested set
  return ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("OrderQCD", -1);
}


double LHAPDF::getLam4(int nmem) {
  return LHAPDF::getLam4(1, nmem) ;
}

double LHAPDF::getLam4(int nset, int nmem) {
  // if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //   throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  // CURRENTSET = nset;
  // ACTIVESETS[nset].loadMember(nmem);
  // return ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("AlphaS_Lambda4", -1.0);
  double qcdl4;
  getlam4m_(nset, nmem, qcdl4);
  return qcdl4;
}


double LHAPDF::getLam5(int nmem) {
  return LHAPDF::getLam5(1, nmem) ;
}

double LHAPDF::getLam5(int nset, int nmem) {
  // if (ACTIVESETS.find(nset) == ACTIVESETS.end())
  //   throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  // CURRENTSET = nset;
  // ACTIVESETS[nset].loadMember(nmem);
  // return ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("AlphaS_Lambda5", -1.0);
  double qcdl5;
  getlam5m_(nset, nmem, qcdl5);
  return qcdl5;
}


int LHAPDF::getNf() {
  return LHAPDF::getNf(1) ;
}

int LHAPDF::getNf(int nset) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return alphaS Order for the requested set
  return ACTIVESETS[nset].activeMember()->info().get_entry_as<int>("NumFlavors");
}


double LHAPDF::getXmin(int nmem) {
  return LHAPDF::getXmin(1, nmem) ;
}

double LHAPDF::getXmin(int nset, int nmem) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return alphaS Order for the requested set
  ACTIVESETS[nset].loadMember(nmem);
  return ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMin");
}

double LHAPDF::getXmax(int nmem) {
  return LHAPDF::getXmax(1, nmem) ;
}

double LHAPDF::getXmax(int nset, int nmem) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return alphaS Order for the requested set
  ACTIVESETS[nset].loadMember(nmem);
  return ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("XMax");
}

double LHAPDF::getQ2min(int nmem) {
  return LHAPDF::getQ2min(1, nmem) ;
}

double LHAPDF::getQ2min(int nset, int nmem) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return alphaS Order for the requested set
  ACTIVESETS[nset].loadMember(nmem);
  return pow(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMin"),2);
}

double LHAPDF::getQ2max(int nmem) {
  return LHAPDF::getQ2max(1,nmem) ;
}

double LHAPDF::getQ2max(int nset, int nmem) {
  if (ACTIVESETS.find(nset) == ACTIVESETS.end())
    throw LHAPDF::UserError("Trying to use LHAGLUE set #" + LHAPDF::to_str(nset) + " but it is not initialised");
  CURRENTSET = nset;
  // return alphaS Order for the requested set
  ACTIVESETS[nset].loadMember(nmem);
  return pow(ACTIVESETS[nset].activeMember()->info().get_entry_as<double>("QMax"),2);
}

double LHAPDF::getQMass(int nf) {
  return LHAPDF::getQMass(1, nf) ;
}

double LHAPDF::getQMass(int nset, int nf) {
  double mass;
  getqmassm_(nset, nf, mass);
  return mass;
}

double LHAPDF::getThreshold(int nf) {
  return LHAPDF::getThreshold(1, nf) ;
}

double LHAPDF::getThreshold(int nset, int nf) {
  double thres;
  getthresholdm_(nset, nf, thres);
  return thres;
}

void LHAPDF::usePDFMember(int member) {
  initpdf_(member);
}

void LHAPDF::usePDFMember(int nset, int member) {
  initpdfm_(nset, member);
}

#endif // ENABLE_LHAGLUE_CXX
