# Makefile is a part of the PYTHIA event generator.
# Copyright (C) 2021 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, September 2014.
#
# This is is the Makefile used to build PYTHIA examples on POSIX systems.
# Example usage is:
#     make test001
# For help using the make command please consult the local system documentation,
# i.e. "man make" or "make --help".

################################################################################
# VARIABLES: Definition of the relevant variables from the configuration script.
################################################################################

# Set the shell.
SHELL=/usr/bin/env bash

# Include the configuration.
-include ../../Makefile.inc

# Handle GZIP support.
GZIP_INC=
GZIP_FLAGS=
ifeq ($(GZIP_USE),true)
  GZIP_INC+= -DGZIPSUPPORT -I$(GZIP_INCLUDE)
  GZIP_FLAGS+= -L$(GZIP_LIB) -Wl,-rpath,$(GZIP_LIB) -lz
endif

# Check distribution (use local version first, then installed version).
ifneq ("$(wildcard ../../lib/libpythia8.*)","")
  PREFIX_LIB=../../lib
  PREFIX_INCLUDE=../../include
endif
CXX_COMMON:=-I$(PREFIX_INCLUDE) $(CXX_COMMON)
CXX_COMMON+= -L$(PREFIX_LIB) -Wl,-rpath,$(PREFIX_LIB) -lpythia8 -ldl 
PYTHIA=$(PREFIX_LIB)/libpythia8$(LIB_SUFFIX)


################################################################################
# RULES: Definition of the rules used to build the PYTHIA examples.
################################################################################

# Rules without physical targets (secondary expansion for specific rules).
.SECONDEXPANSION:
.PHONY: all clean

# All targets (no default behavior).
all:
	@echo "Usage: make check"

# The Makefile configuration.
../../Makefile.inc:
	$(error Error: PYTHIA must be configured, please run "./configure"\
                in the top PYTHIA directory)

# PYTHIA libraries.
$(PYTHIA) :
	$(error Error: PYTHIA must be built, please run "make"\
                in the top PYTHIA directory)

# Examples without external dependencies.
test% : $$@.cc $(PYTHIA)
	$(CXX) $< -o $@ $(CXX_COMMON) $(GZIP_INC) $(GZIP_FLAGS)

# MixMax.
test023: $$@.cc $(PYTHIA)
	$(CXX) $< -o $@ -w $(CXX_COMMON) $(GZIP_INC) $(GZIP_FLAGS)

# HEPMC2 or HEPMC3 (use HEPMC3 if both).
test089 test090: $$@.cc $(PYTHIA)
ifeq ($(HEPMC3_USE),true)
	$(CXX) $@.cc -o $@ $(CXX_COMMON) $(HEPMC3_INCLUDE) $(HEPMC3_LIB)\
	 -DHEPMC3
else ifeq ($(HEPMC2_USE),true)
	$(CXX) $@.cc -o $@ $(CXX_COMMON) $(HEPMC2_INCLUDE) $(HEPMC2_LIB)\
	 -DHEPMC2
else
	$(error Error: $@ requires HEPMC2 or HEPMC3)
endif

CHECKOUTS = out203 out202 out200 out112 out110 out101 out090 out089 out068 out063 out061 out029 out028 out027 out026 out025 out024 out023 out022 out021 out020 out019 out018 out017 out016 out015 out014 out013 out012 out011 out010 out009 out008 out007 out006 out005 out004 out003 out002 out001

CHECKMAINS = test203 test202 test200 test112 test110 test101 test090 test089 test068 test063 test061 test029 test028 test027 test026 test025 test024 test023 test022 test021 test020 test019 test018 test017 test016 test015 test014 test013 test012 test011 test010 test009 test008 test007 test006 test005 test004 test003 test002 test001

all: $(CHECKMAINS)

CHECKDIR = cmp

out%: test%
	@echo started $<
	@PYTHIA8DATA=../../share/Pythia8/xmldoc time ./$< &> $@
	@echo $< done

check%: test% out%
	./testsum -d $(CHECKDIR)

check : $(CHECKMAINS) $(CHECKOUTS)
	./testsum -d $(CHECKDIR)

drycheck : $(CHECKMAINS) $(CHECKOUTS)
	rm -rf $(CHECKDIR)
	mkdir -p $(CHECKDIR)
	cp test[0-9][0-9][0-9].dat out[0-9][0-9][0-9] $(CHECKDIR)


checkdir :
	mkdir -p  $(CHECKDIR)
	cp test[0-9][0-9][0-9].dat out[0-9][0-9][0-9] $(CHECKDIR)

# Clean.
clean:
	@rm -f test[0-9][0-9][0-9]; rm -f out[0-9][0-9][0-9];\
	rm -f *.dat;\
	rm -f weakbosons.lhe; rm -f hist.root;\
	rm -f *~; rm -f \#*; rm -f core*; rm -f *Dct.*; rm -f *.so;
