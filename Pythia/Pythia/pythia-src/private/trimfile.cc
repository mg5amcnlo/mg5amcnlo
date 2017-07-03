// trimfile.cc is a part of the PYTHIA event generator.
// Copyright (C) 2017 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Execute with "private/trimfile filename" from main directory.

// Perform/prepare a number of cleanup operations on a file:
// -- remove unwanted characters at the end of each line
//    (keep one blank for xml files to simplify reformatting);
// -- remove empty lines at the end of the file;
// -- warn if a line is too long
//    (but leave reformatting to manual control);
// -- warn if a line contains tab characters
//    (can be fixed in emacs by "ctrl-x h" followed by "cmnd-x untabify");
// -- perform some string replacements, notably copyright year 
//    (&copy; in Frontpage.xml to be done by hand; also update convertXML).

// By default trimfile only works on .h, .cc, .cmnd and .xml files,
// but command-line option "-f" forces it for any file type.

// With the command-line option "-d" you can give a directory name as input,
// and all the files in it will be processed. Does not handle subdirectories,
// and cannot be combined with "-f". 

//==========================================================================

// Stdlib header files.
#include <vector>
#include <cctype>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <dirent.h>

// Used Stdlib elements.
using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::istringstream;

//--------------------------------------------------------------------------

// Recommended max line length.
int maxLen = 79;

// Possibility to replace strings inside file.
string replaceOld[3] = { "Copyright (C) 2016", "\"true\"", "\"false\"" }; 
string replaceNew[3] = { "Copyright (C) 2017", "\"on\"",   "\"off\""   };
int    replaceLen[3] = { 18, 6, 7 };

//--------------------------------------------------------------------------

// Forward reference.
bool trimFile(string fileName, bool doPad);

//--------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  // Command-line arguments.
  string argv1 = (argc > 1) ? argv[1] : "";
  string argv2 = (argc > 2) ? argv[2] : "";

  // Check if trimming to be forced for all file types.
  bool doForce = (argc == 3 && (argv1 == "-f" || argv2 == "-f"));

  // Check if whole directory of files to be processes.
  bool doDir = (argc == 3 && (argv1 == "-d" || argv2 == "-d"));

  // Check that correct number of command-line arguments.
  if ( argc != 2 && !doForce && !doDir ) {
    cout << "\n Error: wrong number of command-line arguments" << endl;
    return 1;
  }

  // Vector of files to be read. Only one if not directory.
  vector<string> files;
  string fileOrDir = argv1;
  if (fileOrDir == "-f" || fileOrDir == "-d") fileOrDir = argv2;
  if (!doDir) files.push_back( fileOrDir);

  // Open a directory of files. Check that it worked.
  else {
    DIR* directory;
    struct dirent* direntry;
    directory = opendir( fileOrDir.c_str() );
    if (!directory) {
      cout << " Error: failed to open directory " << fileOrDir << endl;
      return 1;
    }

    // Read names of files in directory. Skip hidden files.
    while ( (direntry = readdir(directory)) ) {
      string fileTmp = direntry->d_name;
      if (fileTmp[0] != '.') files.push_back( fileOrDir + "/" + fileTmp);
    }
    closedir(directory);
  }

  // Loop over files in vector.
  for (unsigned int i = 0; i < files.size(); ++i) {
    string fileName = files[i];

    // Identify file type.
    int nameLen = fileName.size();
    bool isCode = (nameLen > 2 && fileName.substr(nameLen - 2, 2) == ".h")
               || (nameLen > 3 && fileName.substr(nameLen - 3, 3) == ".cc")
               || (nameLen > 5 && fileName.substr(nameLen - 5, 5) == ".cmnd");
    bool isXML  = (nameLen > 4 && fileName.substr(nameLen - 4, 4) == ".xml");

    // Skip FJcore and PythonWrapper files; they give plenty of warnings 
    // not to be addressed.
    if (fileName.find("FJcore") != string::npos) isCode = false;
    if (fileName.find("PythonWrapper") != string::npos) isCode = false;

    // Process code and xml files; also others with -f option.
    if (isCode || isXML || doForce) {
      cout << "\n Begin trimming file " << fileName << endl;
      trimFile( fileName, isXML);
    } else {
      cout << "\n File " << fileName << " is not processed" << endl;
    }
  }

  // Done.
  return 0;
}

//--------------------------------------------------------------------------

bool trimFile(string fileName, bool doPad) {

  // Open input file.
  ifstream is( fileName.c_str() );
  if (!is) {
    cout << " Error: input file " << fileName << " not found" << endl;
    return false;
  }

  // Read in input file.
  vector<string> lines;
  string line;
  while ( getline(is, line) ) lines.push_back(line);
  is.close();

  // How many kinds of replacements? Warn for long lines?
  int  kindRep  = doPad ? 3 : 1;
  bool warnLong = (fileName.find("ParticleData.xml") == string::npos); 

  // Counters for number of trims, removals and replacements.
  int nTrim    = 0;
  int nRemove  = lines.size();
  int nReplace = 0;

  // Loop over all lines in the file.
  for (unsigned int i = 0; i < lines.size(); ++i) {
    line = lines[i];

    // Trim the line to have zero or one blank at the end, but no more.
    size_t posEnd = line.find_last_not_of(" \n\t\v\b\r\f\a");
    if (posEnd == string::npos) posEnd = -1;
    line.erase(posEnd + 1);
    if (doPad) line += " ";
    if (line != lines[i]) ++nTrim;

    // Do string replacements inside line where required.
    for (int j = 0; j < kindRep; ++j)
    if (line.find(replaceOld[j]) != string::npos) {
      line.replace( line.find(replaceOld[j]), replaceLen[j], replaceNew[j]);
      ++nReplace;
    }

    // Detect long lines or lines with tab characters.
    if (warnLong && line.size() > maxLen) cout << "    Warning: line " 
      << i + 1 << " is " << line.size() << " characters long" << endl;
    if (line.find("\t") != string::npos) cout << "    Warning: line "
      << i + 1 << " contains a tab character " << endl;

    // Replace processed line.
    lines[i] = line;
  }

  // Remove empty lines at the end of the file.
  if (doPad) {while (lines.back().length() == 1) lines.pop_back();}
  else       {while (lines.back().length() == 0) lines.pop_back();}
  nRemove -= lines.size();

  // Open output file.
  ofstream os( fileName.c_str() );
  if (!os) {
    cout << " Error: output file " << fileName << " not found" << endl;
    return false;
  }

  // Write out all lines.
  for (unsigned int i = 0; i < lines.size(); ++i) os << lines[i] << endl;
  cout << " Successfully processed " << lines.size() << " lines" << endl;
  if (nTrim > 0) cout << "    with " << nTrim << " line trims" << endl;
  if (nRemove > 0) cout << "    with " << nRemove << " line removals" << endl;
  if (nReplace > 0) cout << "    with " << nReplace << " replacements" << endl;

  // Done.
  return true;
}

//==========================================================================
