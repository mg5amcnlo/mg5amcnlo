# Makefile is a part of the DIRE parton shower.
# Copyright (C) 2016 Stefan Prestel.
# Author: Stefan Prestel (heavily relying on the PYTHIA examples Makefile
# by Philip Ilten)
#
# Example usage is:
#     make -j2

################################################################################

# Include the PYTHIA (!) and DIRE configuration files (note that Dire conf. is
# included first, to get the paths to Pythia, and then again, to overwrite
# dangerous Pythia variables)
#-include $(PYTHIA8_EXAMPLES)/Makefile.inc
-include Makefile.inc
-include $(PYTHIA8_PREFIX)/Makefile.inc Makefile.inc

CXX_COMMON:=-I$(PYTHIA8_INCLUDE) $(CXX_COMMON) -O2 -ansi -pedantic -W -Wall -Wshadow -fPIC

# Handle GZIP support.
LIB_COMMON=-Wl,-rpath,$(PREFIX_LIB) -ldl
ifeq ($(GZIP_USE),true)
  CXX_COMMON+= -DGZIPSUPPORT -I$(GZIP_INCLUDE)
  LIB_COMMON+= -L$(GZIP_LIB) -Wl,-rpath,$(GZIP_LIB) -lz
endif

################################################################################

## Now include DIRE configuration - possibly overwriting some of the paths!
#-include Makefile.inc

LOCAL_BIN=bin
LOCAL_EXAMPLES=main
LOCAL_INCLUDE=include
LOCAL_LIB=lib
LOCAL_SRC=src
LOCAL_TMP=tmp

DIRE_INCLUDE=$(LOCAL_INCLUDE)
DIRE_LIB=$(LOCAL_LIB)
DIRE_SRC=$(LOCAL_SRC)
DIRE_TMP=$(LOCAL_TMP)
DIRE_MKDIRS:=$(shell mkdir -p $(DIRE_TMP) $(DIRE_LIB))
CXX_COMMON:=-I$(DIRE_INCLUDE) $(CXX_COMMON)

# DIRE
OBJECTS=$(patsubst $(DIRE_SRC)/%.cc,$(DIRE_TMP)/%.o,\
	$(wildcard $(DIRE_SRC)/*.cc))
TARGETS=$(DIRE_LIB)/libdire.a
ifeq ($(ENABLE_SHARED),true)
  TARGETS+=$(DIRE_LIB)/libdire$(LIB_SUFFIX)
endif

INC_OBJECTS=$(wildcard $(DIRE_INCLUDE)/*/*.h)

################################################################################

# Includes for ME corrections. Not used yet.

ME_SRC=$(DIRE_INCLUDE)/DirePlugins/MyCollectionOfSMProcesses/Processes_sm
ME_INCLUDE=$(DIRE_INCLUDE)/DirePlugins/MyCollectionOfSMProcesses/Processes_sm
ME_TMP=$(DIRE_INCLUDE)/DirePlugins/MyCollectionOfSMProcesses/Processes_sm

OBJECTS+=$(ME_TMP)/HelAmps_sm.o $(ME_TMP)/Parameters_sm.o\
	$(ME_TMP)/PY8MEs.o $(ME_TMP)/rambo.o $(ME_TMP)/read_slha.o\
	$(patsubst $(ME_TMP)/%.cc,$(ME_TMP)/%.o,$(wildcard $(ME_TMP)/PY8ME*.cc))

INC_OBJECTS+=$(wildcard $(ME_INCLUDE)/PY8ME*.h)

################################################################################
# RULES: Definition of the rules used to build DIRE.
################################################################################

# Rules without physical targets (secondary expansion for documentation).
.SECONDEXPANSION:
.PHONY: all install clean distclean

# All targets.
all: $(TARGETS)
	rsync -a Makefile.inc $(LOCAL_EXAMPLES)

# Auto-generated (with -MD flag) dependencies.
-include $(DIRE_TMP)/*.d

# PYTHIA.
$(DIRE_TMP)/Dire.o: $(DIRE_SRC)/Dire.cc
	$(CXX) $< -o $@ -c $(CXX_COMMON)
$(DIRE_TMP)/%.o: $(DIRE_SRC)/%.cc
	$(CXX) $< -o $@ -c $(CXX_COMMON)
$(DIRE_LIB)/libdire.a: $(OBJECTS) $(INC_OBJECTS)
	ar cru $@ $^
$(DIRE_LIB)/libdire$(LIB_SUFFIX): $(OBJECTS) $(wildcard $(DIRE_INCLUDE)/*/*.h)
	$(CXX) $^ -o $@ -c $(CXX_COMMON) $(CXX_SHARED) $(CXX_SONAME),$(notdir $@)\
	  $(LIB_COMMON)

# Install (rsync is used for finer control).
install: all
	mkdir -p $(PREFIX_BIN) $(PREFIX_INCLUDE) $(PREFIX_LIB) $(PREFIX_SHARE)
	rsync -a $(LOCAL_BIN)/* $(PREFIX_BIN) --exclude .svn
	rsync -a $(LOCAL_INCLUDE)/* $(PREFIX_INCLUDE) --exclude .svn
	rsync -a $(LOCAL_LIB)/* $(PREFIX_LIB) --exclude .svn
	rsync -a $(LOCAL_EXAMPLES) $(PREFIX_SHARE) --exclude .svn

# Clean.
clean:
	rm -rf $(DIRE_TMP) $(DIRE_LIB)

# Clean all temporary and generated files.
distclean: clean
	find . -type f -name Makefile.inc -print0 | xargs -0 rm -f
	find . -type f -name "*.o" -print0 | xargs -0 rm -f
