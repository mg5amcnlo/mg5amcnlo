# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: D. Massaro (Sep 2025) for the MG5aMC CUDACPP plugin.
# Based on code originally written by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2024)
# Integrated with the MadGraph7 project in Feb 2026.

# To be used after the project makefile
SHELL := /bin/bash

# Determine CUDACPP_BUILDDIR based on the user-defined choices of BACKEND, FPTYPE, HELINL, HRDCOD and USEBUILDDIR (#829)
# Stop with an error if BACKEND=cuda and nvcc is missing or if BACKEND=hip and hipcc is missing
include ../../src/cudacpp_config.mk
ifeq ($(CUDACPP_BUILDDIR),)
  $(error CUDACPP_BUILDDIR='$(CUDACPP_BUILDDIR)' should not be empty!)
endif

# Basic uname helpers (if not already set)
UNAME_S ?= $(shell uname -s)
UNAME_P ?= $(shell uname -p)
UNAME_M ?= $(shell uname -m)

# Enable the C preprocessor https://gcc.gnu.org/onlinedocs/gfortran/Preprocessing-Options.html
FFLAGS+= -cpp

# Compile counters with -O3 as in the cudacpp makefile (avoid being "unfair" to Fortran #740)
CXXFLAGS = -O3 -Wall -Wshadow -Wextra

# Add -std=c++17 explicitly to avoid build errors on macOS
# Add -mmacosx-version-min=11.3 to avoid "ld: warning: object file was built for newer macOS version than being linked"
ifneq ($(shell $(CXX) --version | egrep '^Apple clang'),)
	CXXFLAGS += -std=c++17 -mmacosx-version-min=11.3
endif

# Enable ccache for C++ if USECCACHE=1 (do not enable it for Fortran since it is not supported for Fortran)
ifeq ($(USECCACHE)$(shell echo $(CXX) | grep ccache),1)
	override CXX := ccache $(CXX)
endif

# ----------------------------------------------------------------------
# Backend library names and process id
# ----------------------------------------------------------------------
CUDACPP_MAKEFILE := cudacpp.mk
processid_short  := $(shell basename $(CURDIR) | awk -F_ '{print $$(NF-1)"_"$$NF}')

ifeq ($(BACKEND),cuda)
	CUDACPP_COMMONLIB := mg5amc_common_cuda
	CUDACPP_BACKENDLIB := mg5amc_$(processid_short)_cuda
else ifeq ($(BACKEND),hip)
	CUDACPP_COMMONLIB := mg5amc_common_hip
	CUDACPP_BACKENDLIB := mg5amc_$(processid_short)_hip
else
	CUDACPP_COMMONLIB := mg5amc_common_cpp
	CUDACPP_BACKENDLIB := mg5amc_$(processid_short)_cpp
endif

# ----------------------------------------------------------------------
# Libraries and link line adjustments
# ----------------------------------------------------------------------
# Prefer LIBDIR everywhere; base makefile already defines LIBDIR.
LINKLIBS := $(LINK_MADLOOP_LIB) $(LINK_LOOP_LIBS) -L$(LIBDIR) \
            -ldhelas -ldsample -lmodel -lgeneric -lpdf -lcernlib $(llhapdf) -lbias

# OpenMP: enable only if requested, USEOPENMP=1 (#758)
ifeq ($(USEOPENMP),1)
  ifneq ($(shell $(CXX) --version | egrep '^Intel'),)
    override OMPFLAGS = -fopenmp
    LINKLIBS += -liomp5 # see #578
    LIBKLIBS += -lintlc # undefined reference to '_intel_fast_memcpy'
  else ifneq ($(shell $(CXX) --version | egrep '^clang'),)
    override OMPFLAGS = -fopenmp
    # For the *cpp* binary with clang, ensure libomp is found
    $(CUDACPP_BUILDDIR)/$(PROG)_cpp: LINKLIBS += -L $(shell dirname $(shell $(CXX) -print-file-name=libc++.so)) -lomp # see #604
  else ifneq ($(shell $(CXX) --version | egrep '^Apple clang'),)
    override OMPFLAGS = # OMP is not supported yet by cudacpp for Apple clang
  else
    override OMPFLAGS = -fopenmp
  endif
endif

# ----------------------------------------------------------------------
# Objects & targets
# ----------------------------------------------------------------------
# Keep driver* separate from PROCESS; we form DSIG groups below.
PROCESS := myamp.o genps.o unwgt.o setcuts.o get_color.o \
           cuts.o cluster.o reweight.o initcluster.o addmothers.o setscales.o \
           idenparts.o dummy_fct.o

DSIG := driver.o $(patsubst %.f, %.o, $(filter-out auto_dsig.f, $(wildcard auto_dsig*.f)))
DSIG_cudacpp := driver_cudacpp.o $(patsubst %.f, %_cudacpp.o, $(filter-out auto_dsig.f, $(wildcard auto_dsig*.f)))

SYMMETRY := symmetry.o idenparts.o

# Binaries

ifeq ($(UNAME),Darwin)
  LDFLAGS += -lc++ -mmacosx-version-min=11.3
else
  LDFLAGS += -Wl,--no-relax
endif

# Explicitly define the default goal (this is not necessary as it is the first target, which is implicitly the default goal)
.DEFAULT_GOAL := all
ifeq ($(BACKEND),cuda)
  all: $(PROG)_fortran $(CUDACPP_BUILDDIR)/$(PROG)_cuda
else ifeq ($(BACKEND),hip)
  all: $(PROG)_fortran $(CUDACPP_BUILDDIR)/$(PROG)_hip
else
  all: $(PROG)_fortran $(CUDACPP_BUILDDIR)/$(PROG)_cpp
endif

# Library build stamps
$(LIBS): .libs

.libs: ../../Cards/param_card.dat ../../Cards/run_card.dat
	$(MAKE) -C ../../Source
	touch $@

$(CUDACPP_BUILDDIR)/.cudacpplibs:
	$(MAKE) VERBOSE=1 -f $(CUDACPP_MAKEFILE)
	touch $@

# Remove per-library recipes from makefile to avoid duplicate sub-makes
# under ../../Source running in parallel otherwise we can have race condition
# Build the libs only via the single .libs stamp.

# Ensure these targets are satisfied by building Source once
$(LIBDIR)libmodel.$(libext)     : | .libs
$(LIBDIR)libgeneric.$(libext)   : | .libs
$(LIBDIR)libpdf.$(libext)       : | .libs
$(LIBDIR)libgammaUPC.$(libext)  : | .libs

# Override the recipes from makefile_orig with empty recipes
# (GNU Make will use the last recipe it reads.)
$(LIBDIR)libmodel.$(libext)     : ; @:
$(LIBDIR)libgeneric.$(libext)   : ; @:
$(LIBDIR)libpdf.$(libext)       : ; @:
$(LIBDIR)libgammaUPC.$(libext)  : ; @:

# On Linux, set rpath to LIBDIR to make it unnecessary to use LD_LIBRARY_PATH
# Use relative paths with respect to the executables ($ORIGIN on Linux)
# On Darwin, building libraries with absolute paths in LIBDIR makes this unnecessary
ifeq ($(UNAME_S),Darwin)
  override LIBFLAGSRPATH :=
else ifeq ($(USEBUILDDIR),1)
  override LIBFLAGSRPATH := -Wl,-rpath,'$$ORIGIN/../$(LIBDIR)/$(CUDACPP_BUILDDIR)'
else
  override LIBFLAGSRPATH := -Wl,-rpath,'$$ORIGIN/$(LIBDIR)'
endif

# Final link steps
$(PROG)_fortran: $(PROCESS) $(DSIG) auto_dsig.o $(LIBS) $(MATRIX) counters.o ompnumthreads.o
	$(FC) -o $@ $(PROCESS) $(DSIG) auto_dsig.o $(MATRIX) $(LINKLIBS) $(BIASDEPENDENCIES) $(OMPFLAGS) counters.o ompnumthreads.o $(LDFLAGS)

# Building $(PROG)_cpp no longer builds $(PROG)_cuda if CUDACPP_BACKENDLIB for cuda exists (this was the case in the past to allow cpp-only builds #503)
$(CUDACPP_BUILDDIR)/$(PROG)_cpp: $(PROCESS) $(DSIG_cudacpp) auto_dsig.o $(LIBS) $(MATRIX) counters.o ompnumthreads.o $(CUDACPP_BUILDDIR)/.cudacpplibs
	$(FC) -o $@ $(PROCESS) $(DSIG_cudacpp) auto_dsig.o $(MATRIX) $(LINKLIBS) $(BIASDEPENDENCIES) $(OMPFLAGS) counters.o ompnumthreads.o -L$(LIBDIR)/$(CUDACPP_BUILDDIR) -l$(CUDACPP_COMMONLIB) -l$(CUDACPP_BACKENDLIB) $(LIBFLAGSRPATH) $(LDFLAGS)

# Building $(PROG)_cuda now uses its own rule
$(CUDACPP_BUILDDIR)/$(PROG)_cuda: $(PROCESS) $(DSIG_cudacpp) auto_dsig.o $(LIBS) $(MATRIX) counters.o ompnumthreads.o $(CUDACPP_BUILDDIR)/.cudacpplibs
	$(FC) -o $@ $(PROCESS) $(DSIG_cudacpp) auto_dsig.o $(MATRIX) $(LINKLIBS) $(BIASDEPENDENCIES) $(OMPFLAGS) counters.o ompnumthreads.o -L$(LIBDIR)/$(CUDACPP_BUILDDIR) -l$(CUDACPP_COMMONLIB) -l$(CUDACPP_BACKENDLIB) $(LIBFLAGSRPATH) $(LDFLAGS)

# Building $(PROG)_hip also uses its own rule
$(CUDACPP_BUILDDIR)/$(PROG)_hip: $(PROCESS) $(DSIG_cudacpp) auto_dsig.o $(LIBS) $(MATRIX) counters.o ompnumthreads.o $(CUDACPP_BUILDDIR)/.cudacpplibs
	$(FC) -o $@ $(PROCESS) $(DSIG_cudacpp) auto_dsig.o $(MATRIX) $(LINKLIBS) $(BIASDEPENDENCIES) $(OMPFLAGS) counters.o ompnumthreads.o -L$(LIBDIR)/$(CUDACPP_BUILDDIR) -l$(CUDACPP_COMMONLIB) -l$(CUDACPP_BACKENDLIB) $(LIBFLAGSRPATH) $(LDFLAGS)

# Helpers compiled with C++
counters.o: counters.cc timer.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

ompnumthreads.o: ompnumthreads.cc ompnumthreads.h
	$(CXX) -I. $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

# Alternate binaries (kept for parity)
$(PROG)_forhel: $(PROCESS) auto_dsig.o $(LIBS) $(MATRIX_HEL)
	$(FC) -o $@ $(PROCESS) $(MATRIX_HEL) $(LINKLIBS) $(LDFLAGS) $(BIASDEPENDENCIES) $(OMPFLAGS)

gensym: $(SYMMETRY) configs.inc $(LIBS)
	$(FC) -o $@ $(SYMMETRY) -L$(LIBDIR) $(LINKLIBS) $(LDFLAGS)

# Compile rules (override base ones)
$(MATRIX): %.o: %.f
	$(FC) $(FFLAGS) $(MATRIX_FLAG) -c $< -I../../Source/ -I../../Source/PDF/gammaUPC

%.o: %.f
	$(FC) $(FFLAGS) -c $< -I../../Source/ -I../../Source/PDF/gammaUPC

%_cudacpp.o: %.f
	$(FC) $(FFLAGS) -c -DMG5AMC_MEEXPORTER_CUDACPP $< -I../../Source/ $(OMPFLAGS) -o $@

# Extra dependencies on discretesampler.mod
auto_dsig.o: .libs
driver.o: .libs
driver_cudacpp.o: .libs
$(MATRIX): .libs
genps.o: .libs

# Convenience link targets to switch $(PROG) symlink
.PHONY: madevent_fortran_link madevent_cuda_link madevent_hip_link madevent_cpp_link
madevent_fortran_link: $(PROG)_fortran
	rm -f $(PROG)
	ln -s $(PROG)_fortran $(PROG)

madevent_cuda_link:
	$(MAKE) USEGTEST=0 BACKEND=cuda $(CUDACPP_BUILDDIR)/$(PROG)_cuda
	rm -f $(PROG)
	ln -s $(CUDACPP_BUILDDIR)/$(PROG)_cuda $(PROG)

madevent_hip_link:
	$(MAKE) USEGTEST=0 BACKEND=hip $(CUDACPP_BUILDDIR)/$(PROG)_hip
	rm -f $(PROG)
	ln -s $(CUDACPP_BUILDDIR)/$(PROG)_hip $(PROG)

madevent_cpp_link:
	$(MAKE) USEGTEST=0 BACKEND=cppauto $(CUDACPP_BUILDDIR)/$(PROG)_cpp
	rm -f $(PROG)
	ln -s $(CUDACPP_BUILDDIR)/$(PROG)_cpp $(PROG)

# Variant AVX builds for cpp backend
override SUPPORTED_AVXS := cppnone cppsse4 cppavx2 cpp512y cpp512z cppauto
madevent_%_link:
	@if [ '$(words $(filter $*, $(SUPPORTED_AVXS)))' != '1' ]; then \
	  echo "ERROR! Invalid target '$@' (supported: $(foreach avx,$(SUPPORTED_AVXS),madevent_$(avx)_link))"; exit 1; fi
	$(MAKE) USEGTEST=0 BACKEND=$* $(CUDACPP_BUILDDIR)/$(PROG)_cpp
	rm -f $(PROG)
	ln -s $(CUDACPP_BUILDDIR)/$(PROG)_cpp $(PROG)

# Cudacpp bldall targets
ifeq ($(UNAME_P),ppc64le)
  bldavxs: bldnone bldsse4
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
  bldavxs: bldnone bldsse4
else
  bldavxs: bldnone bldsse4 bldavx2 bld512y bld512z
endif

ifneq ($(shell which hipcc 2>/dev/null),)
  ifneq ($(shell which nvcc 2>/dev/null),)
    bldall: bldhip bldcuda bldavxs
  else
    bldall: bldhip bldavxs
  endif
else
  ifneq ($(shell which nvcc 2>/dev/null),)
    bldall: bldcuda bldavxs
  else
    bldall: bldavxs
  endif
endif

bldcuda: $(PROG)_fortran $(DSIG_cudacpp)
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cuda

bldhip: $(PROG)_fortran $(DSIG_cudacpp)
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=hip

bldnone: $(PROG)_fortran $(DSIG_cudacpp)
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cppnone

bldsse4: $(PROG)_fortran $(DSIG_cudacpp)
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cppsse4

bldavx2: $(PROG)_fortran $(DSIG_cudacpp)
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cppavx2

bld512y: $(PROG)_fortran $(DSIG_cudacpp)
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cpp512y

bld512z: $(PROG)_fortran $(DSIG_cudacpp)
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cpp512z

# Clean (NB: 'make clean' in Source calls 'make clean' in all P*)
clean: # Clean builds: fortran in this Pn; cudacpp executables for one AVX in this Pn
	$(RM) *.o gensym $(PROG) $(PROG)_fortran $(PROG)_forhel \
	       $(CUDACPP_BUILDDIR)/$(PROG)_cpp \
	       $(CUDACPP_BUILDDIR)/$(PROG)_cuda \
	       $(CUDACPP_BUILDDIR)/$(PROG)_hip

cleanavxs: clean # Clean builds: fortran in this Pn; cudacpp for all AVX in this Pn and in src
	$(MAKE) -f $(CUDACPP_MAKEFILE) cleanall
	rm -f $(CUDACPP_BUILDDIR)/.cudacpplibs
	rm -f .libs

cleanall: # Clean builds: fortran in all P* and in Source; cudacpp for all AVX in all P* and in src
	$(MAKE) -C ../../Source cleanall
	rm -rf $(LIBDIR)libbias.$(libext)
	rm -f ../../Source/*.mod ../../Source/*/*.mod

distclean: cleanall # Clean all fortran and cudacpp builds as well as the googletest installation
	$(MAKE) -f $(CUDACPP_MAKEFILE) distclean

