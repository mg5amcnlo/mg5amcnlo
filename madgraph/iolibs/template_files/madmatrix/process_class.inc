! Copyright (C) 2010 The MadGraph7 development team and contributors.
! Created by: J. Alwall (Jul 2010) for the MadGraph7 CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: A. Valassi (Sep 2021) for the MadGraph7 CUDACPP plugin.
! Further modified by: D. Massaro, A. Thete, A. Valassi (2021-2025).
! Integrated with the MadGraph7 project in Feb 2026.
!==========================================================================
  //==========================================================================
  // A class for calculating the matrix elements for
  %(process_lines)s
  //--------------------------------------------------------------------------

  class CPPProcess
  {
  public: /* clang-format off */

    // Constructor (from command line arguments)
    CPPProcess( bool verbose = false, bool debug = false );

    // Destructor
    ~CPPProcess();

    // Initialize process (read model parameters from file)
    virtual void initProc( const std::string& param_card_name );

#ifndef MGONGPU_HARDCODE_PARAM
    // Re-synchronize the independent parameters/couplings (cIPD/cIPC/cIPF), the
    // BSM-independent-parameter buffer (bsmIndepParam) and the external masses with the
    // current values in the model parameters of this process. sigmaKin() and friends read
    // these copies (parameters()), not the model parameters themselves, for performance,
    // so updating the model parameters alone (e.g. via modelParameters().setParameterByName())
    // has no effect until this is called again; it is called once automatically by initProc().
    void refreshIndependentParams();

    // The model parameters of this process (not shared with any other CPPProcess)
    Parameters& modelParameters() { return *m_pars; }
#endif

    // The parameters the matrix element of this process is evaluated with: the kernels
    // (sigmaKin and friends) take them as an argument. The good helicities in there are
    // set by the caller (sigmaKin_setGoodHel).
    ProcessParameters& parameters() { return m_params; }
    const ProcessParameters& parameters() const { return m_params; }

    // PDG id for a given flavor combination index and particle index (0-based).
    // Useful for tools that need to enumerate flavor combinations without going through UMAMI.
    static int flavorPDG( int iflavor, int ipar );

    // Other methods of this instance (???)
    const std::vector<fptype>& getMasses() const { return m_masses; }
    //virtual int code() const{ return 1; }
    //void setInitial( int inid1, int inid2 ){ id1 = inid1; id2 = inid2; }
    //int getDim() const { return dim; }
    //int getNIOParticles() const { return nexternal; } // nexternal was nioparticles

    // Accessors (unused so far: add four of them only to fix a clang build warning)
    //bool verbose() const { return m_verbose; }
    bool debug() const { return m_debug; }

  public:

    // Compile-time constants: single source of truth is ProcessData.h
    static constexpr int np4 = ProcessData::np4;
    static constexpr int nw6 = ProcessData::nw6;
    static constexpr int npari = ProcessData::npari;
    static constexpr int nparf = ProcessData::nparf;
    static constexpr int npar = ProcessData::npar;
    static constexpr int ncomb = ProcessData::ncomb;
    static constexpr int ndiagrams = ProcessData::ndiagrams;
    static constexpr int ncolor = ProcessData::ncolor;
    // The color structures the color flow is picked among. Same as ncolor,
    // unless the color sum runs on a smaller basis (see set_color_flow_lines_cpp).
    static constexpr int ncolor_flow = %(ncolor_flow)s;%(split_order_constants)s
    static constexpr int nmaxflavor = ProcessData::nmaxflavor;

  private: /* clang-format on */

    // Store the parameters and couplings in m_params (the calls are generated in initProc
    // and refreshIndependentParams)
    void setIndependentParams( const fptype* tIPD );
    void setIndependentCouplings( const cxtype* tIPC );
    void setFlavorCouplings( const int* tIPF_partner1, const int* tIPF_partner2, const cxtype* tIPF_value );
    void setBsmIndepParam( const double* values, int n );
#ifndef MGONGPU_HARDCODE_PARAM
    // Re-populate a mass vector from the model parameters, as initProc() does for m_masses
    void refreshExternalMasses( std::vector<fptype>& masses );
#endif

    // Command line arguments (constructor)
    bool m_verbose;
    bool m_debug;

    // Physics model parameters to be read from file (initProc function), owned by this process
#ifndef MGONGPU_HARDCODE_PARAM
    std::unique_ptr<Parameters> m_ownedPars;
    Parameters* m_pars;
#endif
    std::vector<fptype> m_masses; // external particle masses
    ProcessParameters m_params{}; // see parameters()

    // Other variables of this instance (???)
    //int id1, id2; // initial particle ids
    //cxtype** amp; // ???
  };
