# ----------------------------------------------------------------------------
#
# Makefile for model library
#
# ----------------------------------------------------------------------------

G77=g77
FFLAGS=-O -ffixed-line-length-132
LIBDIR=../../lib

include makeinc.inc

.f.o: ; $(G77) $(FFLAGS) -c $*.f

all: $(LIBDIR)libmodel.a

helas_couplings: helas_couplings.o libmodel.a
	$(G77) $(FFLAGS) -o $@ $^

testprog: testprog.o libmodel.a
	$(G77) $(FFLAGS) -o $@ $^

libmodel.a: $(MODEL)
	ar cru libmodel.a $(MODEL)
	ranlib libmodel.a

$(LIBDIR)libmodel.a: libmodel.a
	cp libmodel.a $(LIBDIR)

clean: 
	rm *.o libmodel.a

