# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Integrated with the MadGraph7 project in Feb 2026.
#
# SubProcesses/makefile: dispatcher over the P* subprocess directories.
#
# This file contains no build rule of its own: the rules live in madmatrix.mk,
# which every P* directory links as its own 'makefile'. All this one does is fan
# out over the subprocess directories with a plain recursive make, so that GNU
# make's jobserver is shared by all the sub-makes. Running
#
#     make -j 18
#
# here therefore spreads 18 concurrent compilations over ALL the subprocess
# directories at once: no directory is built at a time with 18 jobs, and no
# directory is limited to 18/N jobs either. Whenever one directory runs out of
# work its slots are immediately picked up by the others.
#
# Every variable understood by madmatrix.mk (BACKEND, FPTYPE, USEBUILDDIR,
# HELINL, HRDCOD, LIBDIR, DEBUG, PROFILE, ...) can be given on the command line
# as usual; make forwards command-line variables to the sub-makes by itself.

SHELL := /bin/bash

# Group the output of each sub-make instead of interleaving it line by line
# (GNU make >= 4.0; silently skipped on older versions, e.g. the 3.81 shipped
# with macOS, where .FEATURES is undefined).
ifneq ($(filter output-sync,$(.FEATURES)),)
  MAKEFLAGS += --output-sync=target
endif

#-------------------------------------------------------------------------------

#=== The generated subprocess directories

# (the trailing '/.' in the glob restricts the match to directories)
SUBDIRS := $(sort $(patsubst %%/,%%,$(dir $(wildcard P[0-9]*_*/.))))

ifeq ($(SUBDIRS),)
  $(error No subprocess directory (P*) found in $(CURDIR))
endif

# The common library (built from ../src) is shared by every subprocess, so it is
# built once here, before the fan-out. MADMATRIX_COMMONLIB_EXTERNAL=1 then tells
# each P* makefile that this makefile owns the common library, and that it must
# not recurse into ../src on its own -- which is what N sub-makes running in
# parallel would otherwise all do at the same time, racing on the same objects.
#
# The common library is built *through* one representative subprocess directory
# rather than directly: madmatrix.mk already resolves BACKEND (including the
# 'cpu' detection), FPTYPE, the compiler flags and the paths of src/ and
# lib/, and none of that has to be duplicated here.
COMMONDIR := $(firstword $(SUBDIRS))

SUBMAKE = $(MAKE) --no-print-directory MADMATRIX_COMMONLIB_EXTERNAL=1

#-------------------------------------------------------------------------------

#=== Makefile TARGETS
#
# Each action gets one phony target per subprocess directory ('<action>@<dir>').
# Depending on those targets, rather than looping over the directories inside a
# single recipe, is what lets make schedule the directories concurrently.

build_targets    := $(addprefix build@,$(SUBDIRS))
bldall_targets   := $(addprefix bldall@,$(SUBDIRS))
clean_targets    := $(addprefix clean@,$(SUBDIRS))
cleanall_targets := $(addprefix cleanall@,$(SUBDIRS))

.PHONY: all bldall clean cleanall commonlib bldcommonlib cleancommon cleanallcommon
.PHONY: $(SUBDIRS) $(build_targets) $(bldall_targets) $(clean_targets) $(cleanall_targets)

.DEFAULT_GOAL := all

#-------------------------------------------------------------------------------

# Target: build the library of every subprocess (this is the default goal).
# PGOAL is the goal handed to each of them, their default one if empty: 'make
# PGOAL=lib' builds the libraries without the standalone check_sa.exe.
all: $(build_targets)

# The common library must exist before any subprocess links against it: making
# it a prerequisite of every 'build@<dir>' target has make build it exactly once
# and wait for it before starting the fan-out.
$(build_targets): build@%%: commonlib
	+$(SUBMAKE) -C $* $(PGOAL)

# Target: build a single subprocess ("make P1_gg_ttx")
$(SUBDIRS): %%: build@%%

# Target: the library shared by all subprocesses (built from ../src)
commonlib:
	+$(MAKE) --no-print-directory -C $(COMMONDIR) commonlib

#-------------------------------------------------------------------------------

# Target: build every subprocess in all BACKEND modes (each in its own build
# directory). The list of backends worth building on this machine is decided by
# madmatrix.mk, so that the platform logic lives in one place only.
bldall: $(bldall_targets)

$(bldall_targets): bldall@%%: bldcommonlib
	+$(SUBMAKE) -C $* bldall

# Target: the common library, in all BACKEND modes
bldcommonlib:
	+$(MAKE) --no-print-directory -C $(COMMONDIR) bldcommonlib

#-------------------------------------------------------------------------------

# Target: clean the builds of the selected BACKEND, in all subprocesses.
# The common library is cleaned last, once the subprocesses are done with it.
clean: $(clean_targets)
	+$(MAKE) --no-print-directory -C $(COMMONDIR) cleancommon

$(clean_targets): clean@%%:
	+$(SUBMAKE) -C $* clean

# Target: clean the builds of ALL backends, in all subprocesses
cleanall: $(cleanall_targets)
	+$(MAKE) --no-print-directory -C $(COMMONDIR) cleanallcommon

$(cleanall_targets): cleanall@%%:
	+$(SUBMAKE) -C $* cleanall

#-------------------------------------------------------------------------------
