# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: S. Hageboeck (Dec 2020) for the MG5aMC CUDACPP plugin.
# Further modified by: S. Roiser, A. Valassi (2020-2025).
# Integrated with the MadGraph7 project in Feb 2026.

THISDIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Host detection
UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
UNAME_M := $(shell uname -m)

# Only add AVX2/FMA on non-mac and non-ARM hosts
ifeq ($(UNAME_S),Darwin)
  GTEST_CMAKE_FLAGS :=
else ifeq ($(UNAME_M),aarch64)
  GTEST_CMAKE_FLAGS :=
else
  GTEST_CMAKE_FLAGS := -DCMAKE_CXX_FLAGS="-mavx2 -mfma"
endif

# Compiler-specific googletest build directory (#125 and #738)
# In epochX, CXXNAMESUFFIX=_$(CXXNAME) is exported from cudacpp.mk
# In epoch1/epoch2, CXXNAMESUFFIX is undefined
$(info CXXNAMESUFFIX=$(CXXNAMESUFFIX))
BUILDDIR = build$(CXXNAMESUFFIX)
###$(info BUILDDIR=$(BUILDDIR))
INSTALLDIR = install$(CXXNAMESUFFIX)
###$(info INSTALLDIR=$(INSTALLDIR))

CXXFLAGS += -Igoogletest/googletest/include/ -std=c++11

all: googletest/$(INSTALLDIR)/lib64/libgtest.a

googletest/CMakeLists.txt:
	git clone https://github.com/google/googletest.git -b v1.17.0 googletest

googletest/$(BUILDDIR)/Makefile: googletest/CMakeLists.txt
	mkdir -p googletest/$(BUILDDIR)
	cd googletest/$(BUILDDIR) && cmake -DCMAKE_INSTALL_PREFIX:PATH=$(THISDIR)/googletest/install $(GTEST_CMAKE_FLAGS) -DBUILD_GMOCK=OFF ../

googletest/$(BUILDDIR)/lib/libgtest.a: googletest/$(BUILDDIR)/Makefile
	$(MAKE) -C googletest/$(BUILDDIR)

# NB 'make install' is no longer supported in googletest (issue 328)
# NB keep 'lib64' instead of 'lib' as in LCG cvmfs installations
googletest/$(INSTALLDIR)/lib64/libgtest.a: googletest/$(BUILDDIR)/lib/libgtest.a
	mkdir -p googletest/$(INSTALLDIR)/lib64
	cp googletest/$(BUILDDIR)/lib/lib*.a googletest/$(INSTALLDIR)/lib64/
	mkdir -p googletest/$(INSTALLDIR)/include
	cp -r googletest/googletest/include/gtest googletest/$(INSTALLDIR)/include/

clean:
	rm -rf googletest
