# Makefile is a part of the PYTHIA event generator.
# Copyright (C) 2022 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, October 2019.
#
# This is is the Makefile used to build the PYTHIA Python interface on
# POSIX systems. Example usage is:
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
LOCAL_SRC=src
LOCAL_TMP=tmp
LOCAL_INCLUDE=include
LOCAL_MKDIRS:=$(shell mkdir -p $(LOCAL_TMP) $(TOP_LIB))
OBJ_COMMON=-MD $(CXX_COMMON) -Iinclude $(PYTHON_INCLUDE) -w -fpermissive
LIB_COMMON=-Wl,-rpath,$(TOP_LIB) -ldl $(GZIP_LIB)
PYTHIA=$(TOP_LIB)/libpythia8$(LIB_SUFFIX)

# Determine the headers.
HEADERS=$(patsubst ../../include/Pythia8%.h,$(LOCAL_INCLUDE)/Pythia8%.h,\
	$(sort $(wildcard ../../include/Pythia8*/*.h)))

# Determine the objects.
OBJECTS=$(patsubst $(LOCAL_SRC)/%.cpp,$(LOCAL_TMP)/%.o,\
	$(sort $(wildcard $(LOCAL_SRC)/*.cpp)))

################################################################################
# RULES: Definition of the rules used to build the Python interface.
################################################################################

# Rules without physical targets.
.PHONY: all clean

# All targets.
all: $(TOP_LIB)/pythia8.so

# PYTHIA library.
$(PYTHIA):
	$(error Error: PYTHIA must be built, please run "make"\
                in the top PYTHIA directory)

# Auto-dependencies.
-include $(LOCAL_TMP)/*.d

# Build the headers.
$(LOCAL_INCLUDE)/Pythia8%.h: $(TOP_INCLUDE)/Pythia8%.h
	@mkdir -p $(dir $@)
	@sed "s/protected:/public:/g" $< |\
	 sed "s/\(const  *Info\& *info  *=  *infoPrivate;\)/\1\n  "\
	"Info infoPython() {return Info(infoPrivate);}/g" > $@

# Build the objects.
$(LOCAL_TMP)/%.o: $(LOCAL_SRC)/%.cpp
	$(CXX) $< -o $@ -c $(OBJ_COMMON) 

# Build the library.
$(TOP_LIB)/pythia8.so: $(PYTHIA) $(HEADERS) $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(CXX_COMMON) $(CXX_SHARED)\
	 $(CXX_SONAME)$(notdir $@) $(LIB_COMMON) -L$(TOP_LIB)\
	 -Wl,-rpath,$(PREFIX_LIB) -lpythia8 -Wl,-undefined,dynamic_lookup

# Clean.
clean:
	rm -rf $(LOCAL_TMP) $(TOP_LIB)/pythia8.so
