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
#ifndef MG5_CITATION_H
#define MG5_CITATION_H

#include <string>

namespace mg5
{
  // Record that the reference identified by the INSPIRE texkey `key` was used
  // by this run, optionally for the purpose described by the free-text
  // `context`.
  //
  // Each call appends a single line "key<TAB>context" to the per-process file
  //     $MG5_CITATION_DIR/cite.<host>.<pid>.log
  // (de-duplicated within the process).  The orchestrating Python layer
  // collects every such file at the end of the run and turns them into a
  // ready-to-use citations.bib together with a human-readable summary.
  //
  // Per-process file names mean there is never a cross-process write race, on
  // any filesystem.  When MG5_CITATION_DIR is unset the call is a silent no-op,
  // so it is safe to call unconditionally.  Any I/O failure is swallowed:
  // citation tracking must never abort a run.
  void cite( const std::string& key, const std::string& context = "" );
}

#endif // MG5_CITATION_H
