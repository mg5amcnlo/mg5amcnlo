# Build of the Pythia8 driver for MC@NLO showers.
#
# MG5_aMC runs this Makefile in a build directory of its own
# (MCatNLO/build/PYTHIA8_<hash>), passing the configuration on the command
# line; see madgraph/various/mcatnlo_shower.py. By hand, for example:
#
#   mkdir -p build && cd build
#   make -f ../pythia8/Makefile PYTHIA8_CONFIG=/path/to/bin/pythia8-config \
#        HEPMC=3 HEPMC_FLAGS="$(/path/to/bin/pythia8-config --hepmc3)"
#
# Variables:
#   PYTHIA8_CONFIG  pythia8-config of the Pythia8 installation
#   CXX, FC         C++ and Fortran compilers
#   HEPMC           0 (no event file), 2 or 3 (HepMC library used to write
#                   HepMC2 ascii events)
#   HEPMC_FLAGS     include and link flags for that HepMC library
#   ANALYSIS        objects of a Fortran analysis (HwU.o first), built from
#                   ../PY8Analyzer and ../srcCommon; empty: no analysis
#   EXTRA_CXXFLAGS, EXTRA_LDFLAGS, EXTRA_FFLAGS
#                   additional flags (include paths, libraries)

SRCDIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MCATNLO := $(abspath $(SRCDIR)/..)

PYTHIA8_CONFIG ?= pythia8-config
CXX     ?= c++
FC      ?= gfortran
CXXSTD  ?= -std=c++17
HEPMC   ?= 0
HEPMC_FLAGS ?=
ANALYSIS ?=
EXTRA_CXXFLAGS ?=
EXTRA_LDFLAGS ?=
EXTRA_FFLAGS ?=
FFLAGS  ?= -O -w
# Required by the analyses and by the linking with C++, whatever FFLAGS is.
FFLAGS_REQUIRED := -fPIC -ffixed-line-length-132

PY8_CXXFLAGS := $(shell $(PYTHIA8_CONFIG) --cxxflags)
PY8_LIBS     := $(shell $(PYTHIA8_CONFIG) --libs)

ifeq ($(shell uname -s),Darwin)
SOEXT := dylib
else
SOEXT := so
endif

DEFS := -DMG5_HEPMC=$(HEPMC)
FORTRAN_LIBS :=
ifneq ($(strip $(ANALYSIS)),)
DEFS += -DMG5_ANALYSIS
# Link the Fortran run-time library with the C++ compiler.
GFORTRAN_LIB := $(shell $(FC) -print-file-name=libgfortran.$(SOEXT))
ifneq ($(dir $(GFORTRAN_LIB)),./)
FORTRAN_LIBS := -L$(dir $(GFORTRAN_LIB)) -Wl,-rpath,$(dir $(GFORTRAN_LIB))
endif
FORTRAN_LIBS += -lgfortran
endif

vpath %.f  $(MCATNLO)/PY8Analyzer $(MCATNLO)/srcCommon
vpath %.cc $(MCATNLO)/PY8Analyzer $(MCATNLO)/srcCommon

.PHONY: all clean
all: Pythia8.exe

%.o: %.f
	$(FC) $(FFLAGS) $(FFLAGS_REQUIRED) $(EXTRA_FFLAGS) -I$(MCATNLO)/include -c $< -o $@

# The analyses use the modules of HwU.f: build it first (also with make -j).
ifneq ($(filter HwU.o,$(ANALYSIS)),)
$(filter-out HwU.o,$(ANALYSIS)): HwU.o
endif

%.o: %.cc
	$(CXX) $(PY8_CXXFLAGS) $(CXXSTD) $(EXTRA_CXXFLAGS) -I$(MCATNLO)/include -c $< -o $@

mcatnlo_pythia8.o: $(SRCDIR)/mcatnlo_pythia8.cc
	$(CXX) $(PY8_CXXFLAGS) $(CXXSTD) $(DEFS) $(HEPMC_FLAGS) $(EXTRA_CXXFLAGS) \
	  -I$(MCATNLO)/include -c $< -o $@

Pythia8.exe: $(ANALYSIS) mcatnlo_pythia8.o
	$(CXX) $(PY8_CXXFLAGS) $(CXXSTD) -o $@ mcatnlo_pythia8.o $(ANALYSIS) \
	  $(PY8_LIBS) $(HEPMC_FLAGS) $(EXTRA_LDFLAGS) $(FORTRAN_LIBS)

clean:
	rm -f *.o *.mod Pythia8.exe
