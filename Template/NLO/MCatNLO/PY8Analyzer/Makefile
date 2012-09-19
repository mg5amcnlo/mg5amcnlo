# -*- Makefile -*- (for emacs)
# src/Makefile-UserModules.  Generated from Makefile-UserModules.in by configure.
#
# Pythia-8.165
#
# This Makefile received very little testing, 
# any bug reports are very welcome!
#

#
# C++ flags
# 

# Comment out the following variables to eliminate HepMC by brute force
HEPMC  = $(HEPMCPATH)
HEPMCINCLUDE = -I$(HEPMC)/include
HEPMCLIB =   -L$(HEPMC)/lib -lHepMC 
HEPMCFIO =   -L$(HEPMC)/lib -lHepMCfio 

CXX = g++
F77 = g77 -fno-automatic

CXXFLAGS = -g -W -Wall -O3

# add additional package header directories here
CXXFLAGS += $(shell $(PY8PATH)/bin/herwig-config --cppflags )

ALLCCFILES=$(shell echo *.cc)
all : $(ALLCCFILES:.cc=.so)

%.o: ../PYAnalyzer/%.f
	$(F77) -c $< -fPIC -o $@

%.o: ../srcPythia/%.f
	$(F77) -c $< -fPIC -o $@

%.o: %.f
	$(F77) -c $< -fPIC -o $@

%.so : %.cc %.h 
	$(CXX) -shared -fPIC $(CXXFLAGS) $(LDFLAGS) $(EXTRAPATHS) $(EXTRALIBS) $(INCLOPTION) $(HEPMCINCLUDE) $(HEPMCLIB) $(HEPMCFIO) $< -o $@ $(FORFILE)

clean:
	rm -f $(ALLCCFILES:.cc=.so)
