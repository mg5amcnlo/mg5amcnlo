#==========================================================================
# This file has been automatically generated for Pythia 8 by
%(info_lines)s
#==========================================================================

SHELL = /bin/sh

-include Makefile.inc

# flags:
#
#FFLAGSSHARED = -fPIC
CFLAGSSHARED = -fPIC
CXXFLAGSSHARED = -fPIC

# Location of directories.
TMPDIR=../tmp
INCDIR1=include
INCDIR2=%(include_dir)s
SRCDIR=.
LIBDIR=lib
LIBDIRARCH=lib/archive
PROCDIR=%(process_dir)s

all: $(targets) 

CPPFLAGS= -I../$(INCDIR1) -I../$(INCDIR2)
LIBS=-L../$(LIBDIRARCH) -L../lib -lpythia8 -l$(PROCDIR)

all: %(main_file)s

%(main_file)s: %(main_file)s.o ../$(LIBDIRARCH)/lib$(PROCDIR).a
	$(CXX) -o %(main_file)s %(main_file)s.o $(LIBS)

../$(LIBDIRARCH)/lib$(PROCDIR).a: ../$(PROCDIR)/Makefile
	cd ../$(PROCDIR); make
