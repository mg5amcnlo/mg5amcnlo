# -*- Makefile -*- (for emacs)
# src/Makefile-UserModules.  Generated from Makefile-UserModules.in by configure.
#
# Herwig++ 2.3.2
#
# This Makefile is intended for compiling Herwig++ plugins
# You can find example plugins in the Contrib directory
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
TOPDIR = ../
INCDIR = include
CXX = g++
F77 = gfortran -w -fno-automatic -ffixed-line-length-132

CXXFLAGS = -g -W -Wall -O3

# add additional package header directories here
CXXFLAGS += $(shell $(HWPPPATH)/bin/herwig-config --cppflags )

ALLCCFILES=$(shell echo *.cc)
all : HepMCFortran.so

%.o: ../HWAnalyzer/%.f
	$(F77) -c $< -fPIC -I$(TOPDIR)/$(INCDIR) -o $@

%.o: ../srcHerwig/%.f
	$(F77) -c $< -fPIC -I$(TOPDIR)/$(INCDIR) -o $@

%.o: %.f
	$(F77) -c $< -fPIC -I$(TOPDIR)/$(INCDIR) -o $@

%.o: %.cc
	$(CXX) -c $< -fPIC $(INCLOPTION) -I$(TOPDIR)/$(INCDIR) -o $@

HepMCFortran.so : HepMCFortran.cc HepMCFortran.h $(FORFILE)
	$(CXX) -shared -fPIC $(CXXFLAGS) $(LDFLAGS) $(EXTRAPATHS) $(EXTRALIBS) $(INCLOPTION) $(HEPMCINCLUDE) $(HEPMCLIB) $(HEPMCFIO) $< -o $@ $(FORFILE)

clean:
	rm -f $(ALLCCFILES:.cc=.so)
