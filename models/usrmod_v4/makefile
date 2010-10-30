# ----------------------------------------------------------------------------
#
# Makefile for user MODEL directory
# Jan 25 2006
#
# ----------------------------------------------------------------------------

F77           = f77
FFLAGS        = -O -ffixed-line-length-132
LIBRARY       = ../libmodel.a
LIBDIR        = ../../lib/
MODEL         = couplings.o lha_reading.o printout.o couplings_test.o

.f.o: ; $(F77) $(FFLAGS) -c $*.f

all: $(LIBDIR)libmodel.a

testprog: testprog.o $(MODEL)
	$(F77) $(FFLAGS) -o $@ $^

couplings: couplingsvalues.o $(MODEL)
	$(F77) $(FFLAGS) -o $@ $^

$(LIBDIR)libmodel.a: $(MODEL) makefile
	ar cru libmodel.a $(MODEL)
	ranlib libmodel.a
	mv libmodel.a  $(LIBDIR)
