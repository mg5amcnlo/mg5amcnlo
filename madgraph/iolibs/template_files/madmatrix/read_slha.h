// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Sep 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: O. Mattelaer, S. Roiser, A. Valassi (2020-2024).
// Integrated with the MadGraph7 project in Feb 2026.
// ==========================================================================

#ifndef READ_SLHA_H
#define READ_SLHA_H 1

#include <map>
#include <sstream>
#include <string>
#include <vector>

class SLHABlock
{
public:
  SLHABlock( std::string name = "" ) { _name = name; }
  ~SLHABlock() {}
  void set_entry( std::vector<int> indices, double value );
  double get_entry( std::vector<int> indices, double def_val = 0 );
  void set_name( std::string name ) { _name = name; }
  std::string get_name() { return _name; }
  unsigned int get_indices() { return _indices; }
private:
  std::string _name;
  std::map<std::vector<int>, double> _entries;
  unsigned int _indices;
};

class SLHAReader
{
public:
  SLHAReader( std::string file_name = "", bool verbose = true )
  {
    if( file_name != "" ) read_slha_file( file_name, verbose );
  }
  static std::string get_exe_path();
  void read_slha_file( std::string file_name, bool verbose );
  double get_block_entry( std::string block_name, std::vector<int> indices, double def_val = 0 );
  double get_block_entry( std::string block_name, int index, double def_val = 0 );
  void set_block_entry( std::string block_name, std::vector<int> indices, double value );
  void set_block_entry( std::string block_name, int index, double value );
private:
  std::map<std::string, SLHABlock> _blocks;
};

#endif
