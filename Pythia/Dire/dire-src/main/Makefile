# Makefile is a part of the DIRE parton shower.
# Copyright (C) 2016 Stefan Prestel.
# Author: Stefan Prestel (heavily relying on the PYTHIA examples Makefile
# by Philip Ilten)
#
# Example usage is:
#     make dire00

################################################################################
# VARIABLES: Definition of the relevant variables from the configuration script.
################################################################################

# Include the PYTHIA (!) and DIRE configuration files (note that Dire conf. is
# included first, to get the paths to Pythia, and then again, to overwrite
# dangerous Pythia variables)
PYTHIA_LIB_SUFFIX=.a
CXX_ADDITIONAL=
-include Makefile.inc
ifneq ("$(wildcard $(PYTHIA8_PREFIX)/Makefile.inc)","")
-include $(PYTHIA8_PREFIX)/Makefile.inc
ifeq ($(ENABLE_SHARED),true)
  PYTHIA_LIB_SUFFIX=$(LIB_SUFFIX)
  CXX_ADDITIONAL+=-L$(PYTHIA8_PREFIX)/lib -Wl,-rpath $(PYTHIA8_PREFIX)/lib -lpythia8
  ifeq ($(LHAPDF6_USE),true)
    PYTHIA_LIB_SUFFIX=lhapdf6.so
  endif
endif
-include Makefile.inc
else
-include $(PYTHIA8_PREFIX)/share/Pythia8/examples/Makefile.inc
ifeq ($(ENABLE_SHARED),true)
  PYTHIA_LIB_SUFFIX=$(LIB_SUFFIX)
  CXX_ADDITIONAL+=-L$(PYTHIA8_PREFIX)/lib -Wl,-rpath $(PYTHIA8_PREFIX)/lib -lpythia8
  ifeq ($(LHAPDF6_USE),true)
    PYTHIA_LIB_SUFFIX=lhapdf6.so
  endif
endif
-include Makefile.inc
endif

# Handle GZIP support.
GZIP_COMMON=
ifeq ($(GZIP_USE),true)
  GZIP_COMMON+= -DGZIPSUPPORT -I$(GZIP_INCLUDE)
  GZIP_COMMON+= -L$(GZIP_LIB) -Wl,-rpath $(GZIP_LIB) -lz
  #CXX_COMMON+= -DGZIPSUPPORT -I$(GZIP_INCLUDE)
  #CXX_COMMON+= -L$(GZIP_LIB) -Wl,-rpath $(GZIP_LIB) -lz
endif

# Handle OPENMP support.
ifeq ($(OPENMP_USE),true)
  CXX_COMMON+= -fopenmp -DOPENMP
endif

CXX_COMMON:=-I$(PYTHIA8_INCLUDE) $(CXX_COMMON) -O2 -ansi -pedantic -W -Wall -Wshadow -fPIC -Wl,-rpath $(PYTHIA8_LIB) -ldl

DIRE_INCLUDE=$(PREFIX_INCLUDE)
DIRE_SRC=$(PREFIX_SRC)
DIRE_LIB=$(PREFIX_LIB)
DIRE_CONFIG_BIN=$(PREFIX_BIN)/dire-config
DIRE_CONFIG=`$(DIRE_CONFIG_BIN) --all-libs`
DIRE_CONFIG_NOOPEMMP=$(shell $(DIRE_CONFIG_BIN) --all-libs | sed 's,-fopenmp,,g' | sed "s,-DOPENMP,,g")

DIRE_LIB_SUFFIX=.a
ifeq ($(ENABLE_SHARED),true)
  DIRE_LIB_SUFFIX=$(LIB_SUFFIX)
  CXX_ADDITIONAL+=-L$(DIRE_LIB) -Wl,-rpath $(DIRE_LIB) -ldire
endif

ROOT_DIR=/nfs/farm/g/theory/qcdsim/sp/hepsoft/pythia/Pythia8/pythia8/pythia82/branches/sp-dev/share/Pythia8/examples/VINCIA/software/root/build/root-5.34.13
ROOT_USE=true
ROOT_LIB=$(ROOT_DIR)/lib/root
ROOT_BIN=$(ROOT_DIR)/bin/root-config
ROOT_INCLUDE=$(ROOT_DIR)/include/root
CXX_ROOT=`$(ROOT_BIN) --cflags`

################################################################################
# RULES: Definition of the rules used to build the PYTHIA examples.
################################################################################

# Rules without physical targets (secondary expansion for specific rules).
.SECONDEXPANSION:
.PHONY: all clean

# PYTHIA libraries.
$(PYTHIA8_LIB)/libpythia8$(PYTHIA_LIB_SUFFIX):
	$(error Error: PYTHIA must be built, please run "make"\
                in the top PYTHIA directory $(PYTHIA8_LIB))

# Try to make DIRE libraries again, if necessary.
$(DIRE_LIB)/libdire$(DIRE_LIB_SUFFIX): $(DIRE_SRC)/*.cc \
	$(wildcard $(DIRE_INCLUDE)/*/*.h)
ifeq ("$(wildcard $(PWD)/../Makefile)","") 
	@echo "Dire source has been changed. Please recompile source."
else
	@echo "Dire source has been changed. Recompiling in source directory."
	cd ..; $(MAKE); cd main
endif

# Examples without external dependencies.
dire : $$@.cc
	@echo "Default compilation with ''$(CXX) $^ -o $@ $(DIRE_CONFIG)''"
	$(CXX) $^ -o $@ $(DIRE_CONFIG)
dire% : $$@.cc
	@echo "Default compilation with ''$(CXX) $^ -o $@ $(DIRE_CONFIG)''"
	$(CXX) $^ -o $@ $(DIRE_CONFIG)
direOMP : $$@.cc
ifeq ($(OPENMP_USE),true)
	@echo "Default compilation with ''$(CXX) $^ -o $@ $(DIRE_CONFIG)''"
	$(CXX) $^ -o direOMP-openmp $(DIRE_CONFIG)
	@echo "Additional compilation without OpenMP"
	$(CXX) $^ -o direOMP $(DIRE_CONFIG_NOOPEMMP)
else
	@echo "Default compilation with ''$(CXX) $^ -o $@ $(DIRE_CONFIG)''"
	$(CXX) $^ -o $@ $(DIRE_CONFIG)
endif

# Make DIRE main programs with HepMC dependence.
dire01 dire02 dire03: $$@.cc $(DIRE_LIB)/libdire$(DIRE_LIB_SUFFIX)\
	$(PYTHIA8_LIB)/libpythia8$(PYTHIA_LIB_SUFFIX)
ifeq ($(HEPMC2_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(HEPMC2_INCLUDE) $(CXX_COMMON)\
	 $(CXX_ADDITIONAL) -L$(HEPMC2_LIB) -Wl,-rpath $(HEPMC2_LIB) -lHepMC $(GZIP_COMMON)
else
	@echo "Error: $@ requires HEPMC2"
endif

# Make DIRE main programs for deuterium runs with HepMC dependence.
dire-deuterium: $$@.cc $(DIRE_LIB)/libdire$(DIRE_LIB_SUFFIX)\
	$(PYTHIA8_LIB)/libpythia8$(PYTHIA_LIB_SUFFIX)
ifeq ($(HEPMC2_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(HEPMC2_INCLUDE) $(CXX_COMMON)\
	 -L$(HEPMC2_LIB) -Wl,-rpath $(HEPMC2_LIB) -lHepMC $(GZIP_COMMON)
else
	@echo "Error: $@ requires HEPMC2"
endif

# Make DIRE main programs for deuterium runs with ROOT dependence.
dire-deuterium-root: $$@.cc $(DIRE_LIB)/libdire$(DIRE_LIB_SUFFIX)\
	$(PYTHIA8_LIB)/libpythia8$(PYTHIA_LIB_SUFFIX)
ifeq ($(ROOT_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(ROOT_INCLUDE) $(CXX_COMMON) $(CXX_ROOT)\
	-Wl,-rpath -L$(ROOT_LIB) `$(ROOT_BIN) --libs` $(GZIP_COMMON)
else
	@echo "Error: $@ requires ROOT"
endif

# PROMC.
dire04: $$@.cc $(DIRE_LIB)/libdire$(DIRE_LIB_SUFFIX) $(PYTHIA8_LIB)/libpythia8$(PYTHIA_LIB_SUFFIX)
ifeq ($(PROMC_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(PROMC_INCLUDE)/src -I$(PROMC_INCLUDE)/include\
	 $(CXX_COMMON) $(CXX_ADDITIONAL) -DPROMC=\"$(PROMC_INCLUDE)\" -Wno-long-long\
	 -L$(PROMC_LIB) -Wl,-rpath,$(PROMC_LIB) -lpromc -lprotoc -lprotobuf\
	 -lprotobuf-lite -lcbook $(GZIP_COMMON)
else
	@echo "Error: $@ requires PROMC"
endif


# Clean.
clean:
	@rm -f dire;\
	 rm -f dire[0-9][0-9];
