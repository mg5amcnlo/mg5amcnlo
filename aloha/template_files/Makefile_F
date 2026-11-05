# ----------------------------------------------------------------------------
#
# Makefile for DHELAS 4.0 library
# Feb. 28, 2001
#
# ----------------------------------------------------------------------------
#
# Use a TAB to precede shell commands (e.g., f90, ar, etc).
# Note: this Makefile uses features that *may not*
# be supported by make utilities other than GNU make.
#
# ----------------------------------------------------------------------------

include ../make_opts

LIBRARY	= libdhelas.$(libext)
LIBRARY_SHARED	= libdhelas.$(dylibext)
LIBDIR  = ../../lib/

BASIC_OBJS = aloha_functions.o

FFLAGS += $(ALOHA_FLAG) -I ../MODEL
FFLAGS += -fPIC $(ALOHA_FLAG)

include ./aloha_file.inc


$(LIBDIR)$(LIBRARY): $(BASIC_OBJS) $(ALOHARoutine)
	$(call CREATELIB, $@, $^)

# The ALOHA routines 'use aloha_object', the module defined by aloha_functions.f
# ($(BASIC_OBJS)). Order-only, so 'make -j' cannot compile one before the .mod
# exists. This rule MUST stay below the target above: a bare rule for
# $(ALOHARoutine) placed first would hijack the default goal, and 'cd DHELAS &&
# make' would then build the objects but never libdhelas.a.
$(ALOHARoutine): | $(BASIC_OBJS)

# Shared library (optional)
$(LIBDIR)$(LIBRARY_SHARED): $(BASIC_OBJS) $(ALOHARoutine)
	$(FC) -shared -o $@ $^ $(LDFLAGS)

shared:  $(LIBDIR)$(LIBRARY_SHARED)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
	
all: $(LIBDIR)$(LIBRARY)
