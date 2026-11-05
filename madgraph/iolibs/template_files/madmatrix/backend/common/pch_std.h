// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Standard headers precompiled by 'make PCH=1' (see madmatrix.mk).
//
// Every translation unit of a subprocess starts by parsing the same standard
// headers, which with clang is ~0.3 s of the ~0.4 s a small process needs per
// file (and a subprocess has six files). The union of what the backends include
// is listed here; a header a given file does not use costs nothing.

#include <algorithm>
#include <array>
#include <cassert>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
