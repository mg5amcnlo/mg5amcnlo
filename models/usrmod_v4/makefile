# ----------------------------------------------------------------------------
#
# Makefile for user MODEL directory
# Jan 25 2006
#
# ----------------------------------------------------------------------------

ifeq ($(wildcard ../make_opts), ../make_opts)
  include ../make_opts
else
  FFLAGS+= -ffixed-line-length-132
endif

LIBDIR        = ../../lib/
LIBRARY	      = libmodel.$(libext)
MODEL         = couplings.o lha_reading.o printout.o couplings_test.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(MODEL)
	$(call CREATELIB, $@, $^)

testprog: testprog.o $(MODEL)
	$(FC) $(FFLAGS) -o $@ $^

couplings: couplingsvalues.o $(MODEL)
	$(FC) $(FFLAGS) -o $@ $^
