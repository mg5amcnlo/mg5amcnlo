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

all: $(LIBDIR)$(LIBRARY)

helas_couplings: helas_couplings.o $(LIBRARY)
	$(FC) $(FFLAGS) -o $@ $^

testprog: testprog.o $(LIBRARY)
	$(FC) $(FFLAGS) -o $@ $^

$(LIBRARY): $(MODEL)
	ar cru $(LIBRARY) $(MODEL)
	ranlib $(LIBRARY)

$(LIBDIR)$(LIBRARY): $(MODEL)
	$(call CREATELIB, $@, $^)

clean: 
	$(RM) *.o $(LIBDIR)$(LIBRARY)

couplings.f: ../maxparticles.inc ../run.inc ../cuts.inc

../maxparticles.inc:
	touch ../maxparticles.inc

../run.inc:
	touch ../run.inc

../cuts.inc:
	echo "        logical fixed_extra_scale" > ../cuts.inc
	echo "        integer maxjetflavor" >> ../cuts.inc
	echo "        double precision mue_ref_fixed, mue_over_ref" >> ../cuts.inc
	