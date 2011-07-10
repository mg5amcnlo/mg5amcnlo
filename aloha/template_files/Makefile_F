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
LIBDIR  = ../../lib/

BASIC_OBJS = aloha_functions.o

include ./aloha_file.inc

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(BASIC_OBJS) $(ALOHARoutine)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
