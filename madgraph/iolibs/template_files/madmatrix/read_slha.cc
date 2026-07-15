// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Sep 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: O. Mattelaer, S. Roiser, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.
// ==========================================================================

#include "read_slha.h"

#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
//#ifdef __HIPCC__
//#include <experimental/filesystem> // see https://rocm.docs.amd.com/en/docs-5.4.3/CHANGELOG.html#id79
//#else
//#include <filesystem> // bypass this completely to ease portability on LUMI #803
//#endif
#include <fstream>
#include <iostream>

void
SLHABlock::set_entry( std::vector<int> indices, double value )
{
  if( _entries.size() == 0 )
    _indices = indices.size();
  else if( indices.size() != _indices )
    throw "Wrong number of indices in set_entry";

  _entries[indices] = value;
}

double
SLHABlock::get_entry( std::vector<int> indices, double def_val )
{
  if( _entries.find( indices ) == _entries.end() )
  {
    std::cout << "Warning: No such entry in " << _name << ", using default value "
              << def_val << std::endl;
    return def_val;
  }
  return _entries[indices];
}

std::string
SLHAReader::get_exe_path() // see https://cplusplus.com/forum/general/11104
{
  char result[PATH_MAX];
  ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
  return std::string( result, ( count > 0 ) ? count : 0 );
}

void
SLHAReader::read_slha_file( std::string file_name, bool verbose )
{
  const char envpath[] = "MG5AMC_CARD_PATH";
  std::ifstream param_card;
  param_card.open( file_name.c_str(), std::ifstream::in );
  if( param_card.good() )
  {
    if( verbose ) std::cout << "Opened slha file " << file_name << " for reading" << std::endl;
  }
  else if( getenv( envpath ) )
  {
    std::cout << "WARNING! Card file '" << file_name << "' does not exist:"
              << " look for the file in directory $" << envpath << "='" << getenv( envpath ) << "'" << std::endl;
    /*
#ifdef __HIPCC__
    const std::string file_name2 = std::experimental::filesystem::path( getenv( envpath ) ) / std::experimental::filesystem::path( file_name ).filename();
#else
    const std::string file_name2 = std::filesystem::path( getenv( envpath ) ) / std::filesystem::path( file_name ).filename();
#endif
    */
    const std::size_t foundsep = file_name.find_last_of( "/" ); // strip dirname from file_name #923
    const std::string base_name = ( foundsep != std::string::npos ? file_name.substr( foundsep + 1 ) : file_name );
    const std::string file_name2 = std::string( getenv( envpath ) ) + "/" + base_name; // bypass std::filesystem #803
    param_card.open( file_name2.c_str(), std::ifstream::in );
    if( param_card.good() )
    {
      std::cout << "Opened slha file " << file_name2 << " for reading" << std::endl;
    }
    else
    {
      std::cout << "ERROR! Card file '" << file_name2 << "' does not exist" << std::endl;
      throw "Error while opening param card";
    }
  }
  else
  {
    const std::string exepath = SLHAReader::get_exe_path();
    if( exepath != "" )
    {
      std::cout << "WARNING! Card file '" << file_name << "' does not exist:"
                << " and environment variable '" << envpath << "' is not set:"
                << " look for the file in a path relative to the executable path '" << exepath << "'" << std::endl;

      const std::size_t foundsep = exepath.find_last_of( "/" );
      const std::string exedir = ( foundsep != std::string::npos ? exepath.substr( 0, foundsep ) : std::string( "." ) );
      const std::string file_name2 = exedir + "/" + file_name;
      param_card.open( file_name2.c_str(), std::ifstream::in );
      if( param_card.good() )
      {
        std::cout << "Opened slha file " << file_name2 << " for reading" << std::endl;
      }
      else
      {
        std::cout << "WARNING! Card file '" << file_name2 << "' does not exist: try one level higher" << std::endl;
      }
      const std::string file_name3 = exedir + "/../" + file_name;
      param_card.open( file_name3.c_str(), std::ifstream::in );
      if( param_card.good() )
      {
        std::cout << "Opened slha file " << file_name3 << " for reading" << std::endl;
      }
      else
      {
        std::cout << "ERROR! Card file '" << file_name3 << "' does not exist" << std::endl;
        throw "Error while opening param card";
      }
    }
    else
    {
      std::cout << "ERROR! Card file '" << file_name << "' does not exist"
                << " and environment variable '" << envpath << "' is not set"
                << " and executable path cannot be determined" << std::endl;
      throw "Error while opening param card";
    }
  }
  char buf[200];
  std::string line;
  std::string block( "" );
  while( param_card.good() )
  {
    param_card.getline( buf, 200 );
    line = buf;
    // Change to lowercase
    transform( line.begin(), line.end(), line.begin(), (int ( * )( int ))tolower );
    if( line != "" && line[0] != '#' )
    {
      if( block != "" )
      {
        // Look for double index blocks
        double dindex1, dindex2;
        double value;
        std::stringstream linestr2( line );
        if( linestr2 >> dindex1 >> dindex2 >> value &&
            dindex1 == int( dindex1 ) and dindex2 == int( dindex2 ) )
        {
          std::vector<int> indices;
          indices.push_back( int( dindex1 ) );
          indices.push_back( int( dindex2 ) );
          set_block_entry( block, indices, value );
          // Done with this line, read next
          continue;
        }
        std::stringstream linestr1( line );
        // Look for single index blocks
        if( linestr1 >> dindex1 >> value && dindex1 == int( dindex1 ) )
        {
          std::vector<int> indices;
          indices.push_back( int( dindex1 ) );
          set_block_entry( block, indices, value );
          // Done with this line, read next
          continue;
        }
      }
      // Look for block
      if( line.find( "block " ) != line.npos )
      {
        line = line.substr( 6 );
        // Get rid of spaces between block and block name
        while( line[0] == ' ' )
          line = line.substr( 1 );
        // Now find end of block name
        size_t space_pos = line.find( ' ' );
        if( space_pos != std::string::npos )
          line = line.substr( 0, space_pos );
        block = line;
        continue;
      }
      // Look for decay
      if( line.find( "decay " ) == 0 )
      {
        line = line.substr( 6 );
        block = "";
        std::stringstream linestr( line );
        int pdg_code;
        double value;
        if( linestr >> pdg_code >> value )
          set_block_entry( "decay", pdg_code, value );
        else
          std::cout << "Warning: Wrong format for decay block " << line << std::endl;
        continue;
      }
    }
  }
  if( _blocks.size() == 0 )
    throw "No information read from SLHA card";

  param_card.close();
}

double
SLHAReader::get_block_entry( std::string block_name, std::vector<int> indices, double def_val )
{
  if( _blocks.find( block_name ) == _blocks.end() )
  {
    std::cout << "No such block " << block_name << ", using default value "
              << def_val << std::endl;
    return def_val;
  }
  return _blocks[block_name].get_entry( indices );
}

double
SLHAReader::get_block_entry( std::string block_name, int index, double def_val )
{
  std::vector<int> indices;
  indices.push_back( index );
  return get_block_entry( block_name, indices, def_val );
}

void
SLHAReader::set_block_entry( std::string block_name, std::vector<int> indices, double value )
{
  if( _blocks.find( block_name ) == _blocks.end() )
  {
    SLHABlock block( block_name );
    _blocks[block_name] = block;
  }
  _blocks[block_name].set_entry( indices, value );
  /*
  cout << "Set block " << block_name << " entry ";
  for (int i=0;i < indices.size();i++)
    cout << indices[i] << " ";
  cout << "to " << _blocks[block_name].get_entry(indices) << endl;
  */
}

void
SLHAReader::set_block_entry( std::string block_name, int index, double value )
{
  std::vector<int> indices;
  indices.push_back( index );
  set_block_entry( block_name, indices, value );
}
