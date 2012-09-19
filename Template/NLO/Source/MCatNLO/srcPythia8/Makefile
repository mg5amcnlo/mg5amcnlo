#
# Examples Makefile.
#
#                  M. Kirsanov 07.04.2006
#                     Modified 18.11.2006
#                     26.03.2008 CLHEP dependency removed

SHELL = /bin/sh

-include config.mk
ifeq (x$(PYTHIA8LOCATION),x)
 PYTHIA8LOCATION=..
endif
-include $(PYTHIA8LOCATION)/config.mk

# Location of directories.
TOPDIR=$(shell \pwd)
INCDIR=include
SRCDIR=src
LIBDIR=lib
LIBDIRARCH=lib/archive
BINDIR=bin

# Libraries to include if GZIP support is enabled
ifeq (x$(ENABLEGZIP),xyes)
LIBGZIP=-L$(BOOSTLIBLOCATION) -lboost_iostreams -L$(ZLIBLOCATION) -lz
endif

Pythia8: \
	$(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) $@.cc -o $(BINDIR)/$@.exe \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 -llhapdfdummy $(LIBGZIP)
	@ln -fs $(BINDIR)/$@.exe $@.exe

# Create an executable linked to HepMC (if all goes well).
# Owing to excessive warning output -Wshadow is not used for HepMC.
ifneq (x$(HEPMCLOCATION),x)
  main41 main42: \
	$(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libhepmcinterface.a
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -Wno-shadow -I$(PYTHIA8LOCATION)/$(INCDIR) -I$(HEPMCLOCATION)/include \
	$@.cc -o $(BINDIR)/$@.exe \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 -llhapdfdummy $(LIBGZIP) \
	-lhepmcinterface \
	-L$(HEPMCLOCATION)/lib -lHepMC
	@ln -fs $(BINDIR)/$@.exe $@.exe
else
  main41 main42:
	@echo ERROR, this target needs HepMC, variable HEPMCLOCATION
endif

# Create an executable that links to LHAPDF
main51 main52 main53: $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) $@.cc -o $(BINDIR)/$@.exe \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 $(LIBGZIP) \
	-L$(LHAPDFLOCATION) $(LHAPDFLIBNAME) \
	$(FLIBS)
	@ln -fs $(BINDIR)/$@.exe $@.exe

# Create an executable that links to LHAPDF and HepMC
main61 main62: \
	$(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libhepmcinterface.a
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) -Wno-shadow -I$(PYTHIA8LOCATION)/$(INCDIR) -I$(HEPMCLOCATION)/include \
	$@.cc -o $(BINDIR)/$@.exe \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 -lhepmcinterface $(LIBGZIP) \
	-L$(LHAPDFLOCATION) $(LHAPDFLIBNAME) \
	-L$(HEPMCLOCATION)/lib -lHepMC \
	$(FLIBS)
	@ln -fs $(BINDIR)/$@.exe $@.exe

# Create an executable that links to Fastjet
# Owing to excessive warning output -Wshadow is not used for Fastjet.
# (Fixed as of Fastjet 3.0.1, so will be modified eventually.)
ifneq (x$(FASTJETLOCATION),x)
  main71 main72: $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a
	@mkdir -p $(BINDIR)
    # Note: $(CXXFLAGS) is after Fastjet flags as Fastjet includes
    #       optimisation/debug flags which may be unwanted (e.g. -g -O2)
	$(CXX) -I$(PYTHIA8LOCATION)/$(INCDIR) $@.cc \
	`$(FASTJETLOCATION)/bin/fastjet-config --cxxflags --plugins` \
	$(CXXFLAGS) -Wno-shadow \
	-o $(BINDIR)/$@.exe \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 -llhapdfdummy $(LIBGZIP) \
	-L$(FASTJETLOCATION)/lib \
	`$(FASTJETLOCATION)/bin/fastjet-config --libs --plugins`
	@ln -fs $(BINDIR)/$@.exe $@.exe
	@rm -f $@.o
else
  main71 main72:
	@echo ERROR, this target needs Fastjet, variable FASTJETLOCATION
endif

# Create an executable that links to Fastjet, HepMC and LHApdf
# Owing to excessive warning output -Wshadow is not used for Fastjet.
# (Fixed as of Fastjet 3.0.1, so will be modified eventually.)
ifneq (x$(FASTJETLOCATION),x)
  main81 main82 main83 main84: \
	$(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a  $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libhepmcinterface.a
	@mkdir -p $(BINDIR)
    # Note: $(CXXFLAGS) is after Fastjet flags as Fastjet includes
    #       optimisation/debug flags which may be unwanted (e.g. -g -O2)
	$(CXX) -I$(PYTHIA8LOCATION)/$(INCDIR) $@.cc \
	`$(FASTJETLOCATION)/bin/fastjet-config --cxxflags --plugins` \
	$(CXXFLAGS) -Wno-shadow \
	-I$(PYTHIA8LOCATION)/$(INCDIR) -I$(HEPMCLOCATION)/include \
	-o $(BINDIR)/$@.exe \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 \
	-L$(LHAPDFLOCATION) $(LHAPDFLIBNAME) \
	-lhepmcinterface \
	-L$(HEPMCLOCATION)/lib -lHepMC \
	-L$(FASTJETLOCATION)/lib \
	-L$(LHAPDFLOCATION)/lib \
	`$(FASTJETLOCATION)/bin/fastjet-config --libs --plugins`
	@ln -fs $(BINDIR)/$@.exe $@.exe
	@rm -f $@.o
else
  main81 main82 main83 main84:
	@echo ERROR, this target needs Fastjet, variable FASTJETLOCATION
endif

# Create an executable that links to Pythia6. This part is obsolete and 
# only retained, in a watered-down version, for debug purposes.
# Library created with "gfortran -c pythia64xx.f" (xx = current subversion),
# followed by "ar cru libpythia6.a pythia64xx.o" (on Linux).
# You would need to uncomment below and correct pythia6 library path.
#main91: $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a
#	@mkdir -p $(BINDIR)
#	$(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) $@.cc -o $(BINDIR)/$@.exe \
#	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 -llhapdfdummy $(LIBGZIP) \
#	-L/directoryoflibpythia6.a -lpythia6 \
#	$(FLIBS)
#	@ln -fs $(BINDIR)/$@.exe $@.exe

# Clean up: remove executables and outdated files.
.PHONY: clean
clean:
	rm -rf $(BINDIR)
	rm -rf *.exe
	rm -f *~; rm -f \#*; rm -f core*
