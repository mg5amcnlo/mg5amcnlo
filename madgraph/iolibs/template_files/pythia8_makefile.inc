#==========================================================================
# This file has been automatically generated for Pythia 8 by
%(info_lines)s
# Makefile in compliance with the Pythia 8 1.4.9 makefile
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
  targets=$(LIBDIRARCH)/libProcesses_%(model)s.a
  targets+=$(LIBDIR)/libProcesses_%(model)s.$(SHAREDSUFFIX)
else
  targets=$(LIBDIRARCH)/libProcesses_%(model)s.a
endif

all: $(targets) ../config.mk

config.mk: ../configure
	cd ..; ./configure

# Main part: build library. 

$(TMPDIR)/%%.o : $(SRCDIR)/%%.cc
	@mkdir -p $(TMPDIR)
	$(CXX) $(CXXFLAGS) $(CXXFLAGSSHARED) -c -I$(INCDIR) -I$(SRCDIR) $< -o $@

$(TMPDIR)/archive/%%.o : $(SRCDIR)/%%.cc
	@mkdir -p $(TMPDIR)/archive
	$(CXX) $(CXXFLAGS) -c -I$(INCDIR) -I$(SRCDIR) $< -o $@

# Creating the dependency files *.d
# The compiler with option -M is used to build the dependency strings. They
# are further edited with sed (stream editor). The first sed command adds the
# dependency for the *.d files themselves, the second one is needed because
# object files are put in the directory different from src. The last line
# removes empty *.d files produced in case of error.

ifeq ($(SHAREDLIBS),yes)
  $(TMPDIR)/%%.d : $(SRCDIR)/%%.cc
	@echo Making dependency for file $<; \
	mkdir -p $(TMPDIR); \
	$(CC) -M -I$(INCDIR) -I$(SRCDIR) $< | \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' | \
	sed 's/$*\.o/$(TMPDIR)\/$*.o/' > $@; \
	[ -s $@ ] || rm -f $@
endif

$(TMPDIR)/archive/%%.d : $(SRCDIR)/%%.cc
	@echo Making dependency for file $<; \
	mkdir -p $(TMPDIR)/archive; \
	$(CC) -M -I$(INCDIR) -I$(SRCDIR) $< | \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' | \
	sed 's/$*\.o/$(TMPDIR)\/archive\/$*.o/' > $@; \
	[ -s $@ ] || rm -f $@

objects := $(patsubst $(SRCDIR)/%%.cc,$(TMPDIR)/%%.o,$(wildcard $(SRCDIR)/*.cc))
objectsarch := $(patsubst $(SRCDIR)/%%.cc,$(TMPDIR)/archive/%%.o,$(wildcard $(SRCDIR)/*.cc))

$(LIBDIR)/libProcesses_%(model)s.$(SHAREDSUFFIX): $(objects)
	@mkdir -p $(LIBDIR)
	$(CXX) $(LDFLAGSSHARED) -o $@ $(objects) $(LDFLAGLIBNAME),$(notdir $@)

$(LIBDIRARCH)/libProcesses_%(model)s.a: $(objectsarch)
	@mkdir -p $(LIBDIRARCH)
	ar cru $@ $(objectsarch)

deps := $(patsubst $(SRCDIR)/%%.cc,$(TMPDIR)/%%.d,$(wildcard $(SRCDIR)/*.cc))
depsarch := $(patsubst $(SRCDIR)/%%.cc,$(TMPDIR)/archive/%%.d,$(wildcard $(SRCDIR)/*.cc))


# The "if" below is needed in order to avoid producing the dependency files
# when you want to just clean

ifeq (,$(findstring clean, $(MAKECMDGOALS)))
-include $(deps)
-include $(depsarch)
endif

# Clean up: remove (almost?) everything that cannot be recreated.

.PHONY: clean distclean

clean:
	rm -rf $(TMPDIR)
	rm -rf $(LIBDIR)

distclean: clean
	rm -f config.mk
	rm -f *~; rm -f \#*;
	cd $(SRCDIR); rm -f *~; rm -f \#*; cd -
	cd $(INCDIR); rm -f *~; rm -f \#*; cd -
