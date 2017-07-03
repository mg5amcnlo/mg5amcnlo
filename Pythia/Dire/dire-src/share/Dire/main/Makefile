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
-include Makefile.inc
-include $(PYTHIA8_PREFIX)/Makefile.inc Makefile.inc

# Handle GZIP support.
ifeq ($(GZIP_USE),true)
  CXX_COMMON+= -DGZIPSUPPORT -I$(GZIP_INCLUDE)
  CXX_COMMON+= -L$(GZIP_LIB) -Wl,-rpath $(GZIP_LIB) -lz
endif

CXX_COMMON:=-I$(PYTHIA8_INCLUDE) $(CXX_COMMON) -O2 -ansi -pedantic -W -Wall -Wshadow -fPIC -Wl,-rpath $(PYTHIA8_LIB) -ldl

DIRE_INCLUDE=$(PREFIX_INCLUDE)
DIRE_SRC=$(PREFIX_SRC)
DIRE_LIB=$(PREFIX_LIB)

################################################################################
# RULES: Definition of the rules used to build the PYTHIA examples.
################################################################################

# Rules without physical targets (secondary expansion for specific rules).
.SECONDEXPANSION:
.PHONY: all clean

# PYTHIA libraries.
$(PYTHIA8_LIB)/libpythia8.a :
	$(error Error: PYTHIA must be built, please run "make"\
                in the top PYTHIA directory $(PYTHIA8_LIB))

# Make DIRE libraries again, if necessary.
$(DIRE_LIB)/libdire.a : $(DIRE_SRC)/*.cc \
	$(wildcard $(DIRE_INCLUDE)/*/*.h)
	cd ..; $(MAKE) ; cd main

# Examples without external dependencies.
dire% : $$@.cc $(DIRE_LIB)/libdire.a $(PYTHIA8_LIB)/libpythia8.a
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) $(CXX_COMMON)

# Make DIRE main programs with HepMC dependence.
dire01 dire02 dire03 pythia03: $$@.cc $(DIRE_LIB)/libdire.a\
	$(PYTHIA8_LIB)/libpythia8.a
ifeq ($(HEPMC2_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(HEPMC2_INCLUDE) $(CXX_COMMON)\
	 -L$(HEPMC2_LIB) -Wl,-rpath $(HEPMC2_LIB) -lHepMC
else
	@echo "Error: $@ requires HEPMC2"
endif

# FASTJET3 with modified Mass-Drop Tagger.
dire-fj: $$@.cc $(DIRE_LIB)/libdire.a $(PYTHIA8_LIB)/libpythia8.a
ifeq ($(FASTJET3_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(FASTJET3_INCLUDE) $(CXX_COMMON)\
	 -L$(FASTJET3_LIB) -Wl,-rpath,$(FASTJET3_LIB) -lfastjet -lRecursiveTools -lClusteringVetoPlugin -lConstituentSubtractor -lEnergyCorrelator -lGenericSubtractor -lJetCleanser -lJetFFMoments -lJetsWithoutJets -lNsubjettiness -lQCDAwarePlugin -lScJet -lSoftKiller -lSubjetCounting -lValenciaPlugin -lVariableR -lfastjetplugins -lfastjettools -lsiscone -lsiscone_spherical
else
	@echo "Error: $@ requires FASTJET3"
endif

# PROMC.
dire04: $$@.cc $(DIRE_LIB)/libdire.a $(PYTHIA8_LIB)/libpythia8.a
ifeq ($(PROMC_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(PROMC_INCLUDE)/src -I$(PROMC_INCLUDE)/include\
	 $(CXX_COMMON) -DPROMC=\"$(PROMC_INCLUDE)\" -Wno-long-long\
	 -L$(PROMC_LIB) -Wl,-rpath,$(PROMC_LIB) -lpromc -lprotoc -lprotobuf\
	 -lprotobuf-lite -lcbook
else
	@echo "Error: $@ requires PROMC"
endif


# Make DIRE main programs with HepMC dependence.
dire-mess dire-mecs dire-mops: $$@.cc $(DIRE_LIB)/libdire.a\
	$(PYTHIA8_LIB)/libpythia8.a
ifeq ($(HEPMC2_USE),true)
	$(CXX) $^ -o $@ -I$(DIRE_INCLUDE) -I$(HEPMC2_INCLUDE) \
	$(CXX_COMMON)\
	 -L$(HEPMC2_LIB) -Wl,-rpath $(HEPMC2_LIB) -lHepMC
else
	@echo "Error: $@ requires HEPMC2"
endif

# Clean.
clean:
	@rm -f dire[0-9][0-9];\
	 rm -f tmp/*.o; rm -f lib/libdire.a
