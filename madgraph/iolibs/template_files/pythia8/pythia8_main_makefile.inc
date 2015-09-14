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
TMPDIR=../tmp
INCDIR1=include
INCDIR2=%(include_dir)s
SRCDIR=.
LIBDIR=lib
LIBDIRARCH=lib/archive
PROCDIR=%(process_dir)s

all: $(targets) ../config.mk

config.mk: ../configure
	cd ..; ./configure

CPPFLAGS= -I../$(INCDIR1) -I../$(INCDIR2)
LIBS=-L../$(LIBDIRARCH) -lpythia8 -llhapdfdummy -l$(PROCDIR)

all: %(main_file)s

%(main_file)s: %(main_file)s.o ../$(LIBDIRARCH)/lib$(PROCDIR).a
	$(CXX) -o %(main_file)s %(main_file)s.o $(LIBS)

../$(LIBDIRARCH)/lib$(PROCDIR).a: ../$(PROCDIR)/Makefile
	cd ../$(PROCDIR); make
