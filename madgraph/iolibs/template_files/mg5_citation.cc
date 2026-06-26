//==========================================================================
// Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
//
// This file is a part of the MadGraph5_aMC@NLO project, an application which
// automatically generates Feynman diagrams and matrix elements for arbitrary
// high-energy processes in the Standard Model and beyond.
//
// It is subject to the MadGraph5_aMC@NLO license which should accompany this
// distribution.
//
// For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
//==========================================================================
#include "mg5_citation.h"

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <unistd.h>

namespace mg5
{
  void cite( const std::string& key, const std::string& context )
  {
    if ( key.empty() ) return;

    // enabled only when MG5_CITATION_DIR points somewhere
    const char* dir = std::getenv( "MG5_CITATION_DIR" );
    if ( dir == nullptr || dir[0] == '\0' ) return;

    const std::string record = key + '\t' + context;

    // guard the shared state and the file against threads of this process
    static std::mutex mtx;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lock( mtx );

    if ( !seen.insert( record ).second ) return; // already recorded

    // build  <dir>/cite.<host>.<pid>.log
    char host[256];
    if ( gethostname( host, sizeof( host ) ) != 0 ) host[0] = '\0';
    host[sizeof( host ) - 1] = '\0';
    std::string fname = std::string( dir ) + "/cite." + host + "."
                        + std::to_string( static_cast<long>( getpid() ) )
                        + ".log";

    // append the record, swallowing any failure
    try
    {
      std::ofstream out( fname.c_str(), std::ios::app );
      if ( out.is_open() ) out << record << '\n';
    }
    catch ( ... )
    {
      // citation tracking must never abort a run
    }
  }
}
