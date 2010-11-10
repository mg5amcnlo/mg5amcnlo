# ----------------------------------------------------------------------------
#
# Makefile for MODEL directory
# Jan 25 2006
#
# ----------------------------------------------------------------------------


g77           = g77
FFLAGS        = -O -ffixed-line-length-132
LIBRARY	      = ../libmodel.a
LIBDIR        = ../../lib/
MODEL         = printout.o couplings.o

.f.o: ; $(g77) $(FFLAGS) -c $*.f

all: $(LIBDIR)libmodel.a

$(LIBDIR)libmodel.a: $(MODEL)
	ar cru libmodel.a $(MODEL)
	ranlib libmodel.a
	mv libmodel.a  $(LIBDIR)

smcalc: 
	$(g77) $(FFLAGS) SMCalc.f -o smcalc 
