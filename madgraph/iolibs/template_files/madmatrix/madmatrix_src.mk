# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
# Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2024).
# Integrated with the MadGraph7 project in Feb 2026.

#-------------------------------------------------------------------------------

#=== Use bash in the Makefile (https://www.gnu.org/software/make/manual/html_node/Choosing-the-Shell.html)
SHELL := /bin/bash

#-------------------------------------------------------------------------------

#=== Configure common compiler flags for CUDA and C++
INCFLAGS = -I.

#-------------------------------------------------------------------------------

#=== Configure the C++ compiler (note: CXXFLAGS has been exported from the subprocess Makefile)
# Note: AR, CXX and FC are implicitly defined if not set externally
# See https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html
###RANLIB = ranlib

# Add -mmacosx-version-min=11.3 to avoid "ld: warning: object file was built for newer macOS version than being linked"
LDFLAGS =
ifneq ($(shell $(CXX) --version | egrep '^Apple clang'),)
  LDFLAGS += -mmacosx-version-min=11.3
endif

#-------------------------------------------------------------------------------

#=== Configure ccache for C++ builds (note: GPUCC has been exported from the subprocess Makefile including ccache)

# Enable ccache if USECCACHE=1
ifeq ($(USECCACHE)$(shell echo $(CXX) | grep ccache),1)
  override CXX:=ccache $(CXX)
endif

#-------------------------------------------------------------------------------

#=== Configure build directories and build lockfiles ===

# Use the build directory exported from the subprocess Makefile
###$(info MADMATRIX_BUILDDIR=$(MADMATRIX_BUILDDIR))

# Use the build lockfile "full" tag exported from the subprocess Makefile
###$(info TAG=$(TAG))

# Build directory for object files: current directory by default, or build.<BACKEND> if USEBUILDDIR==1
###$(info Current directory is $(shell pwd))
override BUILDDIR = $(MADMATRIX_BUILDDIR)

# LIBDIR: absolute path for library output.
# Normally exported (as absolute path) from SubProcesses/Makefile.
# When building standalone, defaults to ../lib (one level up from src/).
UNAME_S := $(shell uname -s)
LIBDIR ?= ../lib
override LIBDIR := $(abspath $(LIBDIR))
ifeq ($(UNAME_S),Darwin)
  $(shell mkdir -p $(LIBDIR))
endif
###$(info LIBDIR=$(LIBDIR))

#===============================================================================
#=== Makefile TARGETS and build rules below
#===============================================================================

# The common library name carries the full BACKEND suffix so each vectorisation/GPU variant is distinct.
MADMATRIX_COMMONLIB = madmatrix_common_$(BACKEND)

# Explicitly define the default goal (this is not necessary as it is the first target, which is implicitly the default goal)
.DEFAULT_GOAL := all.$(TAG)

# First target (default goal)
all.$(TAG): $(BUILDDIR)/.build.$(TAG) $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so

# Target: tag-specific build lockfile (only in BUILDDIR; LIBDIR is shared across backends)
override oldtagsb=`if [ -d $(BUILDDIR) ]; then find $(BUILDDIR) -maxdepth 1 -name '.build.*' ! -name '.build.$(TAG)' -exec echo $(shell pwd)/{} \; ; fi`

$(BUILDDIR)/.build.$(TAG):
	@if [ "$(oldtagsb)" != "" ]; then echo -e "Cannot build for tag=$(TAG) as old builds exist in $(BUILDDIR) for other tags:\n$(oldtagsb)\nPlease run 'make clean' first\nIf 'make clean' is not enough: run 'make clean USEBUILDDIR=1 BACKEND=$(BACKEND) FPTYPE=$(FPTYPE)' or 'make cleanall'"; exit 1; fi
	@if [ ! -d $(LIBDIR) ]; then echo "mkdir -p $(LIBDIR)"; mkdir -p $(LIBDIR); fi
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	@touch $(BUILDDIR)/.build.$(TAG)

#-------------------------------------------------------------------------------

# Generic target and build rules: objects from C++ or CUDA/HIP compilation.
# Plain .o suffix — the BUILDDIR (e.g. build.cppavx2/) provides backend separation.
# Use USEBUILDDIR=1 to build for multiple backends simultaneously without cleaning.
ifeq ($(GPUCC),)
$(BUILDDIR)/%%.o : %%.cc *.h $(BUILDDIR)/.build.$(TAG)
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	$(CXX) $(CPPFLAGS) $(INCFLAGS) $(CXXFLAGS) -c $< -o $@
else
$(BUILDDIR)/%%.o : %%.cc *.h $(BUILDDIR)/.build.$(TAG)
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	$(GPUCC) $(CPPFLAGS) $(INCFLAGS) $(GPUFLAGS) -c -x $(GPULANGUAGE) $< -o $@
endif

#-------------------------------------------------------------------------------

objects=$(addprefix $(BUILDDIR)/, read_slha.o Parameters.o)

# Target (and build rules): common (src) library
ifeq ($(GPUCC),)
$(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so : $(objects)
	@if [ ! -d $(LIBDIR) ]; then echo "mkdir -p $(LIBDIR)"; mkdir -p $(LIBDIR); fi
	$(CXX) -shared -o $@ $(objects) $(LDFLAGS)
else
$(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so : $(objects)
	@if [ ! -d $(LIBDIR) ]; then echo "mkdir -p $(LIBDIR)"; mkdir -p $(LIBDIR); fi
	$(GPUCC) -shared -o $@ $(objects) $(LDFLAGS)
endif

#-------------------------------------------------------------------------------

# Target: clean the builds
.PHONY: clean cleanall

# clean: remove objects and the common library for the selected BACKEND only.
clean:
ifeq ($(USEBUILDDIR),1)
	rm -rf $(BUILDDIR)
else
	rm -f $(BUILDDIR)/.build.* $(BUILDDIR)/*.o
endif
	rm -f $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so

# cleanall: remove objects and common libraries for ALL backends.
cleanall:
	rm -rf build.*
	rm -f ./.build.* ./*.o
	rm -f $(LIBDIR)/libmadmatrix_common_*.so

#-------------------------------------------------------------------------------
