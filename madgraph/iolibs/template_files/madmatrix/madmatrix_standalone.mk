# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Integrated with the MadGraph7 project in Feb 2026.
#
# Standalone (standalone_mg7) Makefile.
# Extends the non-standalone (madmatrix) Makefile by including madmatrix.mk and
# adding a recipe for check_sa.exe (the standalone driver). Running `make` here
# will build both the process library AND the standalone executable.

# Pull in the non-standalone Makefile (variables, %%.o pattern rule, library
# targets, clean/cleanall/bldall/etc. all come from there). It also sets its
# own .DEFAULT_GOAL := all.$(TAG), which we override below.
include madmatrix.mk

# Override the default goal so that running `make` in this folder builds both
# the process library and the standalone driver.
.DEFAULT_GOAL := standalone_all

#=== Standalone driver (check_sa.exe) configuration

# Standalone-only object files (compiled via the generic %%.o pattern rule from
# madmatrix.mk).
override standalone_objects = $(BUILDDIR)/RamboSamplingKernels.o \
                              $(BUILDDIR)/CommonRandomNumberKernel.o \
                              $(BUILDDIR)/check_sa.o

# Top-level standalone goal: process lib + standalone driver.
.PHONY: standalone_all
standalone_all: all.$(TAG) check_sa.exe

# Linker flags for the standalone driver. The process library brings in
# CPPProcess, umami, the matrix-element kernels and the cross-section helpers;
# the common library carries Parameters.o etc.
# For CUDA/HIP builds we must link with GPUCC because check_sa.o contains device
# code (the AOSOA->SoA transposition kernel).
ifeq ($(GPUCC),)
check_sa.exe: $(standalone_objects) $(LIBDIR)/lib$(MADMATRIX_LIB).so $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so
	$(CXX) -o $@ $(standalone_objects) $(CXXLIBFLAGSRPATH) -L$(LIBDIR) -l$(MADMATRIX_LIB) -l$(MADMATRIX_COMMONLIB) $(BLASLIBFLAGS)
else
check_sa.exe: $(standalone_objects) $(LIBDIR)/lib$(MADMATRIX_LIB).so $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so
	$(GPUCC) -o $@ $(standalone_objects) $(GPULIBFLAGSRPATH) -L$(LIBDIR) -l$(MADMATRIX_LIB) -l$(MADMATRIX_COMMONLIB) $(BLASLIBFLAGS)
endif

#=== Standalone-aware clean

# Append a hook to the existing 'clean' target from madmatrix.mk so that the
# standalone executable and its objects are removed alongside the process lib.
clean: clean_standalone
.PHONY: clean_standalone
clean_standalone:
	rm -f $(BUILDDIR)/RamboSamplingKernels.o $(BUILDDIR)/CommonRandomNumberKernel.o $(BUILDDIR)/check_sa.o
	rm -f check_sa.exe

# 'cleanall' from madmatrix.mk also wipes build.* directories, which already
# covers our standalone objects when USEBUILDDIR=1. We only need to take care
# of the executable itself.
cleanall: cleanall_standalone
.PHONY: cleanall_standalone
cleanall_standalone:
	rm -f check_sa.exe
