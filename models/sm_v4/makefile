# ----------------------------------------------------------------------------
#
# Makefile for MODEL directory
# Jan 25 2006
#
# ----------------------------------------------------------------------------


F77           = f77
FFLAGS        = -O -ffixed-line-length-132
LIBRARY	      = ../libmodel.a
LIBDIR        = ../../lib/
MODEL         = printout.o couplings.o

.f.o: ; $(F77) $(FFLAGS) -c $*.f

all: $(LIBDIR)libmodel.a

$(LIBDIR)libmodel.a: $(MODEL)
	ar cru libmodel.a $(MODEL)
	ranlib libmodel.a
	mv libmodel.a  $(LIBDIR)

smcalc: 
	$(F77) $(FFLAGS) SMCalc.f -o smcalc 
