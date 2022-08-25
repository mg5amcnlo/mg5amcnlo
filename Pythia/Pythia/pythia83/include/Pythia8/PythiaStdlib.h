// PythiaStdlib.h is a part of the PYTHIA event generator.
// Copyright (C) 2022 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Header file for functionality pulled in from Stdlib,
// plus a few useful utilities (small powers; positive square root,
// convert strings to lowercase, Gamma function).

#ifndef Pythia8_PythiaStdlib_H
#define Pythia8_PythiaStdlib_H

// Stdlib header files for mathematics.
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <memory>
#include <functional>
#include <limits>
#include <utility>

// Stdlib header files for strings and containers.
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <deque>
#include <queue>
#include <set>
#include <list>
#include <functional>

// Stdlib header file for dynamic library loading.
#include <dlfcn.h>

// Stdlib header file for input and output.
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

// Thread header files.
#include <mutex>
#include <atomic>
#include <thread>

// Define pi if not yet done.
#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

// Set floating point exceptions from the gcc compiler for debug
// purposes. Use the compilation flag -DGCCFPDEBUG to enable.
#ifdef GCCFPDEBUG
#ifndef __ENABLE_FP_DEBUG__
#define __ENABLE_FP_DEBUG__
#include <fenv.h>
static void __attribute__((constructor)) raisefpe() {
   feenableexcept (FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
}
#endif
#endif

// By this declaration you do not need to use std:: qualifier everywhere.
// using namespace std;

// Alternatively you can specify exactly which std:: methods will be used.
// Now made default so std does not spill outside namespace Pythia8.
namespace Pythia8 {

// Generic utilities and mathematical functions.
using std::move;
using std::swap;
using std::max;
using std::min;
using std::abs;
using std::sort;
using std::function;
using std::isnan;
using std::isinf;
using std::isfinite;
using std::numeric_limits;

// Strings and containers.
using std::pair;
using std::make_pair;
using std::string;
using std::to_string;
using std::vector;
using std::array;
using std::map;
using std::multimap;
using std::unordered_map;
using std::deque;
using std::priority_queue;
using std::set;
using std::multiset;
using std::list;
using std::tuple;
using std::function;

// Input/output streams.
using std::cin;
using std::cout;
using std::cerr;
using std::istream;
using std::ostream;
using std::fstream;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::ios;

// Input/output formatting.
using std::endl;
using std::fixed;
using std::scientific;
using std::left;
using std::right;
using std::setw;
using std::setprecision;

// Pointers
using std::shared_ptr;
using std::weak_ptr;
using std::unique_ptr;
using std::dynamic_pointer_cast;
using std::make_shared;

// Threading.
using std::queue;
using std::mutex;
using std::thread;
using std::atomic;

} // end namespace Pythia8

namespace Pythia8 {

// Define conversion hbar * c = 0.2 GeV * fm = 1 and related.
constexpr double HBARC     = 0.19732698;
constexpr double GEV2FMINV = 1. / HBARC;
constexpr double GEVINV2FM = HBARC;
constexpr double FM2GEVINV = 1./HBARC;
constexpr double FMINV2GEV = HBARC;

// Define conversion (hbar * c)^2 = 0.4 GeV^2 * mb = 1 and related.
constexpr double HBARCSQ     = 0.38937937;
constexpr double GEVSQ2MBINV = 1. / HBARCSQ;
constexpr double GEVSQINV2MB = HBARCSQ;
constexpr double MB2GEVSQINV = 1. / HBARCSQ;
constexpr double MBINV2GEVSQ = HBARCSQ;

// Define conversion between fm and mm, in both directions.
constexpr double FM2MM   = 1e-12;
constexpr double MM2FM   = 1e12;

// Define conversion between mb and pb or fb, in both directions.
constexpr double MB2PB   = 1e9;
constexpr double PB2MB   = 1e-9;
constexpr double MB2FB   = 1e12;
constexpr double FB2MB   = 1e-12;

// Define conversion between fm^2 and mb, in both directions.
constexpr double FMSQ2MB = 10.;
constexpr double MB2FMSQ = 0.1;

// Powers of small integers - for balance speed/code clarity.
constexpr double pow2(const double& x) {return x*x;}
constexpr double pow3(const double& x) {return x*x*x;}
constexpr double pow4(const double& x) {return x*x*x*x;}
constexpr double pow5(const double& x) {return x*x*x*x*x;}
constexpr double pow6(const double& x) {return x*x*x*x*x*x;}
constexpr double pow7(const double& x) {return x*x*x*x*x*x*x;}
constexpr double pow8(const double& x) {return x*x*x*x*x*x*x*x;}

// Avoid problem with negative square root argument (from roundoff).
inline double sqrtpos(const double& x) {return sqrt( max( 0., x));}

// Restrinct value to lie in specified range.
inline double clamp(const double& x, const double& xmin, const double& xmax) {
  return (x < xmin) ? xmin : (x > xmax) ? xmax : x; }

// Convert a string to lowercase for case-insensitive comparisons.
// By default remove any initial and trailing blanks or escape characters.
string toLower(const string& name, bool trim = true);

// Variant of above, with in-place replacement.
inline void toLowerRep(string& name, bool trim = true) {
  name = toLower( name, trim);}

//==========================================================================

} // end namespace Pythia8

// Define the hash for a pair.
namespace std {
  template <class T1, class T2> struct hash<pair<T1, T2> > {
  public:
    size_t operator()(const pair<T1, T2>& p) const {
      return hash<T1>{}(p.first) ^ hash<T2>{}(p.second);
    }
  };

//==========================================================================

} // end namespace std

#endif // Pythia8_PythiaStdlib_H
