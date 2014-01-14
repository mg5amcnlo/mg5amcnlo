# ----------------------------------------------------------------------------
#
# Makefile for TOOLS
#
# ----------------------------------------------------------------------------

LIBRARY       = libtools.$(libext)
LIBDIR        = ../../../lib/

include ../../make_opts

PLOT= dbook.o topgraph.o
OBJS= nexper_2.o  eq_solver.o vegas.o stat.o htuple.o mint-integrator.o xran1.o toms659.o $(PLOT)

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(OBJS)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)

dbook.f: dbook.inc
topgraph.f: dbook.inc
