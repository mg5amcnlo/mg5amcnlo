#==========================================================================
# This file has been automatically generated for Pythia 8 by
%(info_lines)s
#==========================================================================

SHELL = /bin/sh

-include ../config.mk

# flags:
#
#FFLAGSSHARED = -fPIC
CFLAGSSHARED = -fPIC
CXXFLAGSSHARED = -fPIC

# Location of directories.
TMPDIR=tmp
INCDIR=../include
SRCDIR=.
LIBDIR=../lib
LIBDIRARCH=../lib/archive

# Location of libraries to be built.
ifeq ($(SHAREDLIBS),yes)
  targets=$(LIBDIRARCH)/libSigma_%(model)s.a
  targets+=$(LIBDIR)/libSigma_%(model)s.$(SHAREDSUFFIX)
else
  targets=$(LIBDIRARCH)/libSigma_%(model)s.a
endif

all: $(targets) ../config.mk

config.mk: ../configure
	cd ..; ./configure

CPPFLAGS= -I$(INCDIR)
LIBS=-L$(LIBDIRARCH) -lpythia8 -llhapdfdummy -lSigma_%(model)s

all: %(main_file)s

%(main_file)s: %(main_file)s.o ../lib/archive/libSigma_%(model)s.a
	g++ -o %(main_file)s %(main_file)s.o $(LIBS)

$(LIBDIRARCH)/libSigma_%(model)s.a:
	cd ../Processes_%(model)s; make
