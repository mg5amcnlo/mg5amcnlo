SHELL = /bin/sh

-include $(PYTHIA8LOCATION)/config.mk

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

Pythia8: \
	$(PYTHIA8LOCATION)/$(LIBDIRARCH)/libpythia8.a $(PYTHIA8LOCATION)/$(LIBDIRARCH)/libhepmcinterface.a
	$(CXX) $(CXXFLAGS) -I$(PYTHIA8LOCATION)/$(INCDIR) -I$(HEPMCLOCATION)/include \
        $@.cc -o $@.exe \
	-L$(PYTHIA8LOCATION)/$(LIBDIRARCH) -lpythia8 -llhapdfdummy $(LIBGZIP) \
	-lhepmcinterface \
	-L$(HEPMCLOCATION)/lib -lHepMC

.PHONY: clean
clean:
	rm -rf *.exe
	rm -f *~; rm -f \#*; rm -f core*
