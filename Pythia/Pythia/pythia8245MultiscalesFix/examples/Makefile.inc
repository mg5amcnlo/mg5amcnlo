# PYTHIA configuration file.
# Generated on Tue Mar 22 15:59:51 CET 2022 with the user supplied options:
# --prefix=/Users/rikkert/Documents/Physics/MG5_aMC/PY8meetsMG5aMC_release/Pythia/Pythia/pythia8245MultiscalesFix
# --with-hepmc2=/usr/local/
# --cxx-common=-ldl -fPIC -lstdc++ --std=c++14

# Install directory prefixes.
PREFIX_BIN=/Users/rikkert/Documents/Physics/MG5_aMC/PY8meetsMG5aMC_release/Pythia/Pythia/pythia8245MultiscalesFix/bin
PREFIX_INCLUDE=/Users/rikkert/Documents/Physics/MG5_aMC/PY8meetsMG5aMC_release/Pythia/Pythia/pythia8245MultiscalesFix/include
PREFIX_LIB=/Users/rikkert/Documents/Physics/MG5_aMC/PY8meetsMG5aMC_release/Pythia/Pythia/pythia8245MultiscalesFix/lib
PREFIX_SHARE=/Users/rikkert/Documents/Physics/MG5_aMC/PY8meetsMG5aMC_release/Pythia/Pythia/pythia8245MultiscalesFix/share/Pythia8

# Compilation flags (see ./configure --help for further documentation).
ENABLE_SHARED=false
CXX=g++
CXX_COMMON=-ldl -fPIC -lstdc++ --std=c++14
CXX_SHARED=-dynamiclib
CXX_SONAME=-Wl,-dylib_install_name,@rpath/
LIB_SUFFIX=.dylib

# EVTGEN configuration.
EVTGEN_USE=false
EVTGEN_BIN=
EVTGEN_INCLUDE=
EVTGEN_LIB=

# FASTJET3 configuration.
FASTJET3_USE=false
FASTJET3_BIN=
FASTJET3_INCLUDE=
FASTJET3_LIB=

# HEPMC2 configuration.
HEPMC2_USE=true
HEPMC2_BIN=/usr/local//bin
HEPMC2_INCLUDE=/usr/local//include
HEPMC2_LIB=/usr/local//lib

# HEPMC3 configuration.
HEPMC3_USE=false
HEPMC3_BIN=
HEPMC3_INCLUDE=
HEPMC3_LIB=

# LHAPDF5 configuration.
LHAPDF5_USE=false
LHAPDF5_BIN=
LHAPDF5_INCLUDE=
LHAPDF5_LIB=

# LHAPDF6 configuration.
LHAPDF6_USE=false
LHAPDF6_BIN=
LHAPDF6_INCLUDE=
LHAPDF6_LIB=

# POWHEG configuration.
POWHEG_USE=false
POWHEG_BIN=
POWHEG_INCLUDE=
POWHEG_LIB=

# PROMC configuration.
PROMC_USE=false
PROMC_BIN=
PROMC_INCLUDE=
PROMC_LIB=

# ROOT configuration.
ROOT_USE=false
ROOT_BIN=
ROOT_LIBS=
CXX_ROOT=
ROOT_INCLUDE=
ROOT_LIB=

# GZIP configuration.
GZIP_USE=false
GZIP_BIN=
GZIP_INCLUDE=
GZIP_LIB=

# BOOST configuration.
BOOST_USE=false
BOOST_BIN=
BOOST_INCLUDE=
BOOST_LIB=

# PYTHON configuration.
PYTHON_USE=false
PYTHON_BIN=
PYTHON_INCLUDE=
PYTHON_LIB=
