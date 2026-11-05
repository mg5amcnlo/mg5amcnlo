SHELL = /bin/sh

-include $(PYTHIA8LOCATION)/config.mk
-include ../Source/make_opts

# $(CXX) may be a C driver: MadGraph auto-detects the compiler and writes it to
# Source/make_opts as DEFAULT_CPP_COMPILER, and plain 'clang'/'gcc' compile C++
# sources but do NOT link the C++ runtime. STDLIB is set per machine in
# Source/make_opts (-lc++ / -lstdc++) and is appended to the link lines below;
# this only fills it in should that file be missing.
ifeq ($(strip $(STDLIB)),)
  ifeq ($(shell uname -s),Darwin)
    STDLIB = -lc++
  else
    STDLIB = -lstdc++
  endif
endif


F77=$(FC)
DEBUG=
# HepMC2 Fortran I/O library, not needed (nor available) with HepMC3 (-DHEPMC3)
HEPMCFIO=-lHepMCfio
# The C++ compiler links the Fortran analysis: ask the Fortran compiler where
# its runtime library is, as it is not always on the search path of the C++
# one (e.g. clang with the gfortran of Homebrew gcc on macOS).
GFORTRAN_LIB:=$(shell $(FC) -print-file-name=libgfortran$(if $(filter Darwin,$(shell uname -s)),.dylib,.so) 2>/dev/null)
LIBGFORTRAN=$(if $(GFORTRAN_LIB),-L$(dir $(GFORTRAN_LIB)) )-lgfortran
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
	-lpythia8tohepmc -L$(HEPMCLOCATION)/$(LIBDIR) -lHepMC -lHepMCfio $(LIBGFORTRAN) $(STDLIB) \

Pythia82: $(FORFILE) $(PYTHIA8_LIB)
	  $(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) \
	  -I$(TOPDIR)/$(INCDIR) $(INCLOPTION) $@.cc -o Pythia8.exe $(FORFILE) \
	  -L$(PYTHIA8LOCATION)/$(LIBDIR) -lpythia8 $(LIBGZIP) \
	  -lHepMCfio $(LIBGFORTRAN) $(HEPMCINCLIB) $(EXTRAPATHS) $(EXTRALIBS) $(STDLIB) \

Pythia83: $(FORFILE) $(PYTHIA8_LIB)
	  $(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) \
	  -I$(TOPDIR)/$(INCDIR) $(INCLOPTION) $@.cc -o Pythia8.exe $(FORFILE) \
	  -L$(PYTHIA8LOCATION)/$(LIBDIR) -lpythia8 $(LIBGZIP) \
	  $(HEPMCFIO) $(LIBGFORTRAN) $(HEPMCINCLIB) $(EXTRAPATHS) $(EXTRALIBS) $(STDLIB) \

.PHONY: clean
clean:
	rm -rf *.exe
	rm -f *~; rm -f \#*; rm -f core*
