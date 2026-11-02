! Copyright (C) 2010 The MadGraph7 development team and contributors.
! Created by: J. Alwall (Jul 2010) for the MadGraph7 CPP backend.
!==========================================================================
! Copyright (C) 2020-2026 CERN and UCLouvain.
! Licensed under the GNU Lesser General Public License (version 3 or later).
! Modified originally by: A. Valassi (Sep 2021) for the MadGraph7 CUDACPP plugin.
! Further modified by: J. Teig, A. Valassi, Z. Wettersten (2021-2025).
! Integrated with the MadGraph7 project in Feb 2026.
!==========================================================================
//==========================================================================
// Class member functions for calculating the matrix elements for
%(process_lines)s
//
// sigmaKin and everything it calls (calculate_jamps, good-helicity
// filtering, color/channel selection) are backend-owned, see
// backend/{cpu,simd,gpu}/SigmaKin.cc, which #includes the process-specific
// EvaluateDiagrams.inc and ColorFlows.inc generated alongside this file. This file only
// contains what's genuinely process-specific: the class constructor,
// physics-parameter/coupling setup (initProc), and the flavor accessor.
//==========================================================================

namespace madmatrix
{
  using namespace ProcessData;
  using namespace ProcessTables; // for nMF (flavor-coupling setup in initProc)

  //--------------------------------------------------------------------------

  CPPProcess::CPPProcess( bool verbose,
                          bool debug )
    : m_verbose( verbose )
    , m_debug( debug )
#ifndef MGONGPU_HARDCODE_PARAM
    , m_pars( 0 )
#endif
    , m_masses()
  {
    setHelicitiesAndFlavors( &tHel[0][0], &tFlavors[0][0] );

    // Enable SIGFPE traps for Floating Point Exceptions
#ifdef MGONGPUCPP_DEBUG
    fpeEnable();
#endif
  }

  //--------------------------------------------------------------------------

  CPPProcess::~CPPProcess() {}

  //--------------------------------------------------------------------------

  // PDG id for the given flavor combination and particle index (host side).
  int
  CPPProcess::flavorPDG( int iflavor, int ipar )
  {
    return flavorPDGs[iflavor][ipar];
  }

  //--------------------------------------------------------------------------

#ifndef MGONGPU_HARDCODE_PARAM
  // Initialize process (with parameters read from user cards)
  void
  CPPProcess::initProc( const std::string& param_card_name )
  {
    // Instantiate the model class and set parameters that stay fixed during run
    m_pars = Parameters::getInstance();
    // Parameters is a process-wide singleton (not owned per-instance), so a fresh
    // initProc() must explicitly drop any overrides left by a previous
    // setParameterByName() call -- otherwise re-initializing from a (possibly
    // different) param card would silently carry forward stale overrides instead
    // of giving a clean slate.
    m_pars->resetParameterOverrides();
    SLHAReader slha( param_card_name, m_verbose );
    m_pars->setIndependentParameters( slha );
    m_pars->setIndependentCouplings();
    //m_pars->setDependentParameters(); // now computed event-by-event (running alphas #373)
    //m_pars->setDependentCouplings(); // now computed event-by-event (running alphas #373)
    if( m_verbose )
    {
      m_pars->printIndependentParameters();
      m_pars->printIndependentCouplings();
      //m_pars->printDependentParameters(); // now computed event-by-event (running alphas #373)
      //m_pars->printDependentCouplings(); // now computed event-by-event (running alphas #373)
    }
    %(initProc_lines)s
    createNormalizedColorMatrix(); // no-op on cpu/simd, pushes to device memory on gpu
    refreshIndependentParams();
  }

  // See declaration in CPPProcess.h for why this exists and when it must be called.
  void
  CPPProcess::refreshIndependentParams()
  {
    Parameters* m_pars = Parameters::getInstance();
    // Read physics parameters like masses and couplings from user configuration files,
    // then hand them to the backend-owned storage via the setters (see SigmaKin.cc).
    %(cipdassign)s
    %(cipcassign)s
    %(cipfassign)s
    %(bsmassign)s
  }

  // See declaration in CPPProcess.h for why this exists and when it must be called.
  void
  CPPProcess::refreshExternalMasses( std::vector<fptype>& masses )
  {
    Parameters* m_pars = Parameters::getInstance();
    masses.clear();
%(refreshExternalMasses_lines)s
  }
#else
  // Initialize process (with hardcoded parameters)
  void
  CPPProcess::initProc( const std::string& /*param_card_name*/ )
  {
    // Use hardcoded physics parameters
    if( m_verbose )
    {
      Parameters::printIndependentParameters();
      Parameters::printIndependentCouplings();
      //Parameters::printDependentParameters(); // now computed event-by-event (running alphas #373)
      //Parameters::printDependentCouplings(); // now computed event-by-event (running alphas #373)
    }
    %(hardcoded_initProc_lines)s
    createNormalizedColorMatrix(); // no-op on cpu/simd, pushes to device memory on gpu
    %(cipdhrdassign)s
    %(cipchrdassign)s
    %(cipfhrdassign)s
    %(bsmhrdassign)s
  }
#endif

  //--------------------------------------------------------------------------
} // end namespace
