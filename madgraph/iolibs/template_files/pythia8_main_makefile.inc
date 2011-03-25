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

all: $(targets) ../config.mk

config.mk: ../configure
	cd ..; ./configure

CPPFLAGS= -I$(INCDIR)
LIBS=-L$(LIBDIRARCH) -lpythia8 -llhapdfdummy -lProcesses_%(model)s

all: %(main_file)s

%(main_file)s: %(main_file)s.o ../lib/archive/libProcesses_%(model)s.a
	g++ -o %(main_file)s %(main_file)s.o $(LIBS)

$(LIBDIRARCH)/libProcesses_%(model)s.a: ../Processes_%(model)s/Makefile
	cd ../Processes_%(model)s; make
