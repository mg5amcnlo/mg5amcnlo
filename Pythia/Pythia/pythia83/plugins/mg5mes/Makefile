# Makefile is a part of the PYTHIA event generator.
# Copyright (C) 2022 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, October 2019.
#
# This is is the Makefile used to build the PYTHIA MG5 matrix element plugins
# on POSIX systems. Example usage is:
#     make -j2
# For help using the make command please consult the local system documentation,
# i.e. "man make" or "make --help".

################################################################################
# VARIABLES: Definition of the relevant variables from the configuration script
# and the distribution structure.
################################################################################

# Set the shell.
SHELL=/usr/bin/env bash

# Local variables.
-include ../../Makefile.inc
TOP_LIB=../../lib
TOP_INCLUDE=../../include
LOCAL_TMP=tmp
LOCAL_INCLUDE=include
LOCAL_DIRS:=$(subst $(LOCAL_TMP)/,,$(wildcard */))
LOCAL_MKDIRS:=$(shell mkdir -p $(LOCAL_TMP) $(TOP_LIB))
LOCAL_MKDIRS:=$(foreach DIR,$(LOCAL_DIRS),$(shell mkdir -p $(LOCAL_TMP)/$(DIR)))
OBJ_COMMON=-MD -DPYTHIA8 $(CXX_COMMON) -I$(TOP_INCLUDE) -w -fpermissive
LIB_COMMON=-Wl,-rpath,$(TOP_LIB) -ldl $(GZIP_LIB) -Wl,-undefined,dynamic_lookup
PYTHIA=$(TOP_LIB)/libpythia8$(LIB_SUFFIX)

# Set the wrapper, determine the objects and targets.
WRAPPER=ExternalMEsMadgraph
OBJECTS=$(addprefix $(LOCAL_TMP)/,$(patsubst %.cc,%.o,$(wildcard */*.cc)))
TARGETS=$(patsubst %/,$(TOP_LIB)/libpythia8mg5%.so,$(LOCAL_DIRS))

################################################################################
# RULES: Definition of the rules used to build the Python interface.
################################################################################

# Rules without physical targets.
.PHONY: all clean

# All targets.
all: $(TARGETS)

# PYTHIA library.
$(PYTHIA):
	$(error Error: PYTHIA must be built, please run "make"\
                in the top PYTHIA directory)

# Auto-dependencies.
-include $(patsubst %.o,%.d,$(OBJECTS))

# Build the objects.
$(LOCAL_TMP)/%/$(WRAPPER).o: $(TOP_INCLUDE)/Pythia8Plugins/$(WRAPPER).h
	PARS=`basename $*/Parameters_*.h`;\
	 $(CXX) -x c++ $< -o $@ -c -DPARS=$${PARS%.h} $(OBJ_COMMON) -I$*
$(LOCAL_TMP)/%.o: %.cc
	$(CXX) $< -o $@ -c $(OBJ_COMMON)

# Build the libraries.
define LIB_RULE
../../lib/libpythia8mg5$(1).so: $(2) $(LOCAL_TMP)/$(1)/$(WRAPPER).o
	$(CXX) $$^ -o $$@ $(CXX_COMMON) $(CXX_SHARED)\
	 $(CXX_SONAME)$(notdir $$@) $(LIB_COMMON)

endef
$(foreach DIR, $(LOCAL_DIRS),$(eval $(call LIB_RULE,$(subst /,,$(DIR)),\
$(addprefix $(LOCAL_TMP)/,$(patsubst %.cc,%.o,$(wildcard $(DIR)*.cc))))))

# Clean.
clean:
	rm -rf $(LOCAL_TMP) $(TARGETS)
