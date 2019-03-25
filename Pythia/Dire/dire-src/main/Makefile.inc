# DIRE configuration file.
# Generated on Mon Mar 25 10:21:24 CET 2019 with the user supplied options:
# --prefix=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Dire/dire-build
# --with-pythia8=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src
# Configure string:
# ./configure --prefix=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Dire/dire-build --with-pythia8=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src

# Install directory prefixes.
PREFIX_BIN=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Dire/dire-build/bin
PREFIX_INCLUDE=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Dire/dire-build/include
PREFIX_LIB=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Dire/dire-build/lib
PREFIX_SRC=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Dire/dire-src/src
PREFIX_SHARE=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Dire/dire-build/share/Dire

# Compilation flags (see ./configure --help for further documentation).
ENABLE_SHARED=false
CXX=g++
CXX_COMMON=-O2 -std=c++98 -pedantic -W -Wall -Wshadow -fPIC
CXX_SHARED=-shared
CXX_SONAME=-Wl,-soname,
LIB_SUFFIX=.so

# PYTHIA8 configuration.
PYTHIA8_USE=true
PYTHIA8_BIN=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src/bin/
PYTHIA8_INCLUDE=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src/include
PYTHIA8_LIB=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src/lib
PYTHIA8_PREFIX=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src
PYTHIA8_SHARE=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src/share
PYTHIA8_EXAMPLES=/home/oates/rikkert/space/MG5_aMC/PY8meetsMG5aMC/Pythia/Pythia/pythia-src/share/Pythia8/examples

# MG5MES configuration.
MG5MES_USE=false
MG5MES_BIN=
MG5MES_INCLUDE=./
MG5MES_LIB=./

# OPENMP configuration.
OPENMP_USE=false
OPENMP_BIN=
OPENMP_INCLUDE=./
OPENMP_LIB=./