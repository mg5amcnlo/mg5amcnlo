SHELL = /bin/sh

-include $(PYTHIA8LOCATION)/config.mk
-include ../Source/make_opts

F77=$(FC)
DEBUG=
FF=$(F77) $(DEBUG)

# Location of directories.
TOPDIR=$(shell \pwd)
INCDIR=include
SRCDIR=src
LIBDIR=lib
LIBDIRARCH=lib/archive

# Match -lpythia8: accept shared libraries as well as static archives.
PYTHIA8_LIB = $(firstword $(wildcard \
    $(PYTHIA8LOCATION)/$(LIBDIR)/libpythia8.so \
    $(PYTHIA8LOCATION)/$(LIBDIR)/libpythia8.dylib \
    $(PYTHIA8LOCATION)/$(LIBDIR)/libpythia8.a) \
    $(PYTHIA8LOCATION)/$(LIBDIR)/libpythia8.a)

# Libraries to include if GZIP support is enabled
ifeq (x$(ENABLEGZIP),xyes)
#LIBGZIP=-L$(BOOSTLIBLOCATION) -lboost_iostreams -L$(ZLIBLOCATION) -lz
LIBGZIP=-L$(ZLIBLOCATION) -lz
endif

%.o: ./PY8Analyzer/%.f
	$(F77) -c $< -fPIC -ffixed-line-length-132 -I$(TOPDIR)/$(INCDIR) -o $@

%.o: ./srcCommon/%.f
	$(F77) -c $< -fPIC -ffixed-line-length-132 -I$(TOPDIR)/$(INCDIR) -o $@

%.o: %.f
	$(F77) -c $< -fPIC -ffixed-line-length-132 -I$(TOPDIR)/$(INCDIR) -o $@

%.o: ./PY8Analyzer/%.cc
	$(CXX) -c $< -fPIC $(INCLOPTION) -o $@

%.o: ./srcCommon/%.cc
	$(CXX) -c $< -fPIC $(INCLOPTION) -o $@

%.o: %.cc
	$(CXX) -c $< -fPIC $(INCLOPTION) -o $@

Pythia8: $(FORFILE) $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a \
	$(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8tohepmc.a
	$(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) \
	-I$(HEPMCLOCATION)/$(INCDIR) -I$(TOPDIR)/$(INCDIR) \
	$(EXTRAPATHS) $(EXTRALIBS) $(INCLOPTION) $@.cc -o Pythia8.exe $(FORFILE) \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 $(LIBLHAPDF) $(LIBGZIP) \
	-lpythia8tohepmc -L$(HEPMCLOCATION)/$(LIBDIR) -lHepMC -lHepMCfio $(LIBGFORTRAN) \

Pythia82: $(FORFILE) $(PYTHIA8_LIB)
	  $(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) \
	  -I$(TOPDIR)/$(INCDIR) $(INCLOPTION) $@.cc -o Pythia8.exe $(FORFILE) \
	  -L$(PYTHIA8LOCATION)/$(LIBDIR) -lpythia8 $(LIBGZIP) \
	  -lHepMCfio $(LIBGFORTRAN) $(HEPMCINCLIB) $(EXTRAPATHS) $(EXTRALIBS) \

Pythia83: $(FORFILE) $(PYTHIA8_LIB)
	  $(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) \
	  -I$(TOPDIR)/$(INCDIR) $(INCLOPTION) $@.cc -o Pythia8.exe $(FORFILE) \
	  -L$(PYTHIA8LOCATION)/$(LIBDIR) -lpythia8 $(LIBGZIP) \
	  -lHepMCfio $(LIBGFORTRAN) $(HEPMCINCLIB) $(EXTRAPATHS) $(EXTRALIBS) \

# Standalone stock-Pythia first-emission driver; no provider, HepMC or Fortran analysis.
MacNLOPS: MacNLOPS.cc MacNLOPS.h $(PYTHIA8_LIB)
	$(CXX) $(CXXFLAGS) -std=c++11 -I$(PYTHIA8LOCATION)/$(INCDIR) \
	  $< -o MacNLOPS.exe -L$(PYTHIA8LOCATION)/$(LIBDIR) \
	  -Wl,-rpath,$(PYTHIA8LOCATION)/$(LIBDIR) -lpythia8 $(LIBGZIP)

# In-process per-channel matching; each SubProcesses/P* directory links it
# with its own densities (make macnlops_onestep).
MacNLOPSMatch.o: MacNLOPSMatch.cc MacNLOPS.h $(PYTHIA8_LIB)
	$(CXX) $(CXXFLAGS) -std=c++11 -I$(PYTHIA8LOCATION)/$(INCDIR) -c $< -o $@

.PHONY: clean MacNLOPS
clean:
	rm -rf *.exe
	rm -f *~; rm -f \#*; rm -f core*
