# ----------------------------------------------------------------------------
#
# Makefile for MODEL directory
# Jan 25 2006
#
# ----------------------------------------------------------------------------

include ../make_opts

LIBDIR        = ../../lib/
LIBRARY	      = libmodel.$(libext)
MODEL         = couplings.o printout.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(MODEL)
	$(call CREATELIB, $@, $^)
