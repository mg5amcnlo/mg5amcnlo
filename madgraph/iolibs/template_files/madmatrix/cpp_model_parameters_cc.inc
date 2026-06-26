// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024) for the MG5aMC CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#include "Parameters.h"

#include <iomanip>
#include <iostream>

#ifdef MGONGPUCPP_GPUIMPL
using namespace mg5amcGpu;
#else
using namespace mg5amcCpu;
#endif

#ifndef MGONGPU_HARDCODE_PARAM

// Initialize static instance
Parameters* Parameters::instance = 0;

// Function to get static instance - only one instance per program
Parameters*
Parameters::getInstance()
{
  if( instance == 0 )
    instance = new Parameters();
  return instance;
}

void
Parameters::setIndependentParameters( SLHAReader& slha )
{
  zero = 0;                         // define "zero"
  ZERO = 0;                         // define "zero"
  std::vector<int> indices( 2, 0 ); // prepare a vector for indices
  %(set_independent_parameters)s
}

void
Parameters::setIndependentCouplings()
{
  %(set_independent_couplings)s
  %(set_flv_couplings)s
}

/*
void
Parameters::setDependentParameters() // now computed event-by-event (running alphas #373)
{
  %(set_dependent_parameters)s
}

void
Parameters::setDependentCouplings() // now computed event-by-event (running alphas #373)
{
  %(set_dependent_couplings)s
}
*/

#endif

// Routines for printing out parameters
void
Parameters::printIndependentParameters()
{
  std::cout << "model parameters independent of event kinematics:" << std::endl;
  std::cout << "(Warning: aS in the runcard is ignored because event-by-event Gs are hardcoded or retrieved from Fortran)" << std::endl;
  %(print_independent_parameters)s
}

void
Parameters::printIndependentCouplings()
{
  std::cout << "model couplings independent of event kinematics:" << std::endl;
  %(print_independent_couplings)s
}

/*
void
Parameters::printDependentParameters() // now computed event-by-event (running alphas #373)
{
  std::cout << "model parameters dependent on event kinematics:" << std::endl;
  %(print_dependent_parameters)s
}

void
Parameters::printDependentCouplings() // now computed event-by-event (running alphas #373)
{
  std::cout << "model couplings dependent on event kinematics:" << std::endl;
  %(print_dependent_couplings)s
}
*/
