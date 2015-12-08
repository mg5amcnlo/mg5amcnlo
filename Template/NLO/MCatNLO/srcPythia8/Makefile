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

# Libraries to include if GZIP support is enabled
ifeq (x$(ENABLEGZIP),xyes)
LIBGZIP=-L$(BOOSTLIBLOCATION) -lboost_iostreams -L$(ZLIBLOCATION) -lz
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

Pythia82: $(FORFILE) $(PYTHIA8LOCATION)/$(LIBDIR)/libpythia8.a
	  $(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) \
	  -I$(TOPDIR)/$(INCDIR) $(INCLOPTION) $@.cc -o Pythia8.exe $(FORFILE) \
	  -L$(PYTHIA8LOCATION)/$(LIBDIR) -lpythia8 $(LIBGZIP) \
	  -lHepMCfio $(LIBGFORTRAN) $(HEPMCINCLIB) $(EXTRAPATHS) $(EXTRALIBS) \


.PHONY: clean
clean:
	rm -rf *.exe
	rm -f *~; rm -f \#*; rm -f core*
