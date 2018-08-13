# DIRE configuration file.
# Generated on Wed Jul 25 15:36:01 CDT 2018 with the user supplied options:
# --prefix=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Dire/dire-build
# --with-pythia8=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build
# Configure string:
# ./configure --prefix=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Dire/dire-build --with-pythia8=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build

# Install directory prefixes.
PREFIX_BIN=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Dire/dire-build/bin
PREFIX_INCLUDE=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Dire/dire-build/include
PREFIX_LIB=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Dire/dire-build/lib
PREFIX_SRC=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Dire/dire-src/src
PREFIX_SHARE=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Dire/dire-build/share/Dire

# Compilation flags (see ./configure --help for further documentation).
ENABLE_SHARED=false
CXX=g++
CXX_COMMON=-O2 -std=c++98 -pedantic -W -Wall -Wshadow -fPIC
CXX_SHARED=-shared
CXX_SONAME=-Wl,-soname,
LIB_SUFFIX=.so

# PYTHIA8 configuration.
PYTHIA8_USE=true
PYTHIA8_BIN=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build/bin/
PYTHIA8_INCLUDE=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build/include
PYTHIA8_LIB=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build/lib
PYTHIA8_PREFIX=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build
PYTHIA8_SHARE=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build/share
PYTHIA8_EXAMPLES=/home/prestel/work/2018/MG5/PY8meetsMG5aMC/Pythia/Pythia/pythia-build/share/Pythia8/examples

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
