// Copyright (C) 2010 The MadGraph7 development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MadGraph7 CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: A. Valassi (Sep 2021) for the MadGraph7 CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2021-2024) for the MadGraph7 CUDACPP plugin.
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
// This file has been automatically generated for CUDA/C++ standalone by
%(info_lines)s
//==========================================================================

#include "Parameters.h"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace madmatrix;

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
  m_slha = slha;
  registerParameters();
  %(set_independent_parameters)s
}

void
Parameters::setIndependentCouplings()
{
  %(set_independent_couplings)s
  %(set_flv_couplings)s
}

void
Parameters::registerParameters()
{
  %(register_parameters)s
  %(register_slha_keys)s
}

bool
Parameters::resolveSlhaKey( const std::string& name, std::string& resolved ) const
{
  std::istringstream iss( name );
  std::string block;
  if( !( iss >> block ) ) return false;
  std::transform( block.begin(), block.end(), block.begin(),
                  []( unsigned char c ) { return std::tolower( c ); } );
  std::string key = block;
  bool any_index = false;
  int index;
  while( iss >> index )
  {
    key += " " + std::to_string( index );
    any_index = true;
  }
  if( !any_index ) return false;
  std::map<std::string, std::string>::const_iterator it = m_slha_key_to_name.find( key );
  if( it == m_slha_key_to_name.end() ) return false;
  resolved = it->second;
  return true;
}

bool
Parameters::setParameterByName( std::string name, double real, double imag )
{
  (void)imag; // settable (SLHA-card-level) parameters are always real
  std::string resolved;
  if( resolveSlhaKey( name, resolved ) ) name = resolved;
  if( m_settable_params.find( name ) == m_settable_params.end() )
    return false;
  m_param_overrides[name] = real;
  setIndependentParameters( m_slha );
  setIndependentCouplings();
  return true;
}

bool
Parameters::getParameterByName( std::string name, double* real, double* imag )
{
  std::string resolved;
  if( resolveSlhaKey( name, resolved ) ) name = resolved;
  std::map<std::string, double*>::iterator real_it = m_real_params.find( name );
  if( real_it != m_real_params.end() )
  {
    *real = *( real_it->second );
    if( imag ) *imag = 0.;
    return true;
  }
  // cxtype (not hardcoded cxsmpl<double>): must match m_complex_params'
  // own value_type (see its declaration's own comment) -- real()/imag()
  // below still convert safely into *real/*imag (always plain double*
  // here) whatever cxtype's own underlying precision is.
  std::map<std::string, cxtype*>::iterator complex_it = m_complex_params.find( name );
  if( complex_it != m_complex_params.end() )
  {
    *real = complex_it->second->real();
    if( imag ) *imag = complex_it->second->imag();
    return true;
  }
  // Independent model parameters (as opposed to derived couplings just
  // above) -- see m_complex_model_params' own declaration comment for why
  // this is a second, separate map/type rather than sharing
  // m_complex_params.
  std::map<std::string, cxsmpl<double>*>::iterator complex_model_it =
      m_complex_model_params.find( name );
  if( complex_model_it != m_complex_model_params.end() )
  {
    *real = complex_model_it->second->real();
    if( imag ) *imag = complex_model_it->second->imag();
    return true;
  }
  return false;
}

void
Parameters::resetParameterOverrides()
{
  m_param_overrides.clear();
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
