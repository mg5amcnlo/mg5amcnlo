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
# dangerous Pythia variables). Note: Only include files if not performing
# clean or distclean
ifeq (,$(findstring clean, $(MAKECMDGOALS)))
-include Makefile.inc
ifneq ("$(wildcard $(PYTHIA8_PREFIX)/Makefile.inc)","")
-include $(PYTHIA8_PREFIX)/Makefile.inc Makefile.inc
else
-include $(PYTHIA8_PREFIX)/share/Pythia8/examples/Makefile.inc Makefile.inc
endif
endif

# Remove optimization flags from CXX_COMMON
CXX_COMMON:=$(shell echo $(CXX_COMMON) | sed 's|-O1||g' | sed 's|-O2||g' | sed 's|-O3||g')
CXX_COMMON:=$(shell echo $(CXX_COMMON) | sed 's|-std=c++98||g')

# Add Dire compilation flags.
#CXX_COMMON:=-I$(PYTHIA8_INCLUDE) $(CXX_COMMON) -O2 -ansi -pedantic -W -Wall -Wshadow -fPIC
CXX_COMMON:=-I$(PYTHIA8_INCLUDE) $(CXX_COMMON) -ansi -pedantic -W -Wall -Wshadow -fPIC

# Handle GZIP support.
LIB_COMMON=-Wl,-rpath,$(PREFIX_LIB) -ldl
ifeq ($(GZIP_USE),true)
  CXX_COMMON+= -DGZIPSUPPORT -I$(GZIP_INCLUDE)
  LIB_COMMON+= -L$(GZIP_LIB) -Wl,-rpath,$(GZIP_LIB) -lz
endif

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
# Includes for ME corrections.

ifeq ($(MG5MES_USE),true)
  MG5MES_TMP=$(LOCAL_TMP)
  OBJECTS+=$(MG5MES_TMP)/HelAmps_sm.o\
	 $(MG5MES_TMP)/Parameters_sm.o\
	 $(MG5MES_TMP)/PY8MEs.o\
	 $(MG5MES_TMP)/read_slha.o\
	 $(patsubst $(MG5MES_INCLUDE)/Processes_sm/%.cc,$(MG5MES_TMP)/%.o,$(wildcard $(MG5MES_INCLUDE)/Processes_sm/PY8ME*.cc))
  CXX_COMMON+=-I$(MG5MES_INCLUDE) -DMG5MES
endif

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

# Dire
$(DIRE_TMP)/Dire.o: $(DIRE_SRC)/Dire.cc
	$(CXX) $< -o $@ -c $(CXX_COMMON)
$(DIRE_TMP)/%.o: $(DIRE_SRC)/%.cc
	$(CXX) $< -o $@ -c $(CXX_COMMON)
# External matrix elements
$(MG5MES_TMP)/%.o: $(MG5MES_INCLUDE)/Processes_sm/%.cc
	$(CXX) $< -o $@ -c $(CXX_COMMON)
# Static library
$(DIRE_LIB)/libdire.a: $(OBJECTS) $(INC_OBJECTS)
	ar cr $@ $^
# Shared library
$(DIRE_LIB)/libdire$(LIB_SUFFIX): $(OBJECTS)
	$(CXX) $^ -o $@ $(CXX_COMMON) $(CXX_SHARED) $(CXX_SONAME)$(notdir $@)\
	  $(LIB_COMMON)

# Install (rsync is used for finer control).
install: all
	mkdir -p $(PREFIX_BIN) $(PREFIX_INCLUDE) $(PREFIX_LIB) $(PREFIX_SHARE)
	rsync -a $(LOCAL_BIN)/* $(PREFIX_BIN) --exclude .svn
	rsync -a $(LOCAL_INCLUDE)/* $(PREFIX_INCLUDE) --exclude .svn
	rsync -a $(LOCAL_LIB)/* $(PREFIX_LIB) --exclude .svn
	rsync -a $(LOCAL_EXAMPLES) $(PREFIX_SHARE) --exclude .svn
	cd $(PREFIX_SHARE)/main; find . -type f -name "dire" -print0 | xargs -0 rm -f; $(MAKE) dire; cp dire $(PREFIX_BIN)

# Clean.
clean:
	rm -rf $(DIRE_TMP) $(DIRE_LIB)

# Clean all temporary and generated files.
distclean: clean
	find . -type f -name Makefile.inc -print0 | xargs -0 rm -f
	find . -type f -name "*.o" -print0 | xargs -0 rm -f
	find bin -type f -name "dire" -print0 | xargs -0 rm -f

# Try creating the Makefile configuration.
Makefile.inc:
	./configure --with-pythia8=$$(locate libpythia8.a | xargs ls -lrth | tail -n 1 | rev | cut -d ' ' -f1 | cut -d '/' -f3- | rev)
