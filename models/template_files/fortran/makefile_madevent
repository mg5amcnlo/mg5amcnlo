# ----------------------------------------------------------------------------
#
# Makefile for model library
#
# ----------------------------------------------------------------------------

# Check for ../make_opts
ifeq ($(wildcard ../make_opts), ../make_opts)
  include ../make_opts
else
  FFLAGS+= -ffixed-line-length-132
  FC=gfortran
endif

include makeinc.inc

LIBDIR=../../lib/
LIBRARY=libmodel.$(libext)
RUNNING = ../alfas_functions.o ../RUNNING/matrix_exponential.o ../RUNNING/c8lib.o  ../RUNNING/r8lib.o

all: $(LIBDIR)$(LIBRARY) ../param_card.inc

rw_para.o: ../param_card.inc 
	$(FC) $(FFLAGS) -c -o rw_para.o rw_para.f

../param_card.inc: ../../Cards/param_card.dat
	../../bin/madevent treatcards param
helas_couplings: helas_couplings.o $(LIBRARY)
	$(FC) $(FFLAGS) -o $@ $^

testprog: testprog.o $(LIBRARY) # $(RUNNING)
	$(FC) $(FFLAGS) -o $@ $^

$(LIBRARY): $(MODEL)
	ar cru $(LIBRARY) $(MODEL)
	ranlib $(LIBRARY)

$(LIBDIR)$(LIBRARY): $(MODEL)
	$(call CREATELIB, $@, $^)

clean: 
	$(RM) *.o $(LIBDIR)$(LIBRARY)

couplings.f: ../maxparticles.inc ../run.inc

../run.inc:
	touch ../run.inc

../maxparticles.inc:
	touch ../maxparticles.inc