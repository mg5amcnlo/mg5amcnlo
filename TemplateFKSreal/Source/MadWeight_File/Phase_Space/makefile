# ----------------------------------------------------------------------------
#
# Makefile for PHASE SPACE library
# Apr. 25 2003
#
# ----------------------------------------------------------------------------
#this is a partial combinaison

gfortran           = f77
FFLAGS        = -O -ffixed-line-length-132
LIBDIR        = ../../../lib/
OBJS= get_point.o initialize.o


.f.o: ; $(gfortran) $(FFLAGS) -c $*.f

all: $(LIBDIR)libMWPS.a

$(LIBDIR)libMWPS.a: $(OBJS)
	ar cru libMWPS.a $(OBJS)
	ranlib libMWPS.a
	mv libMWPS.a  $(LIBDIR)

get_point.f: ../../nexternal.inc ../../genps.inc
initialize.f: ../../nexternal.inc ../../genps.inc ../../coupl.inc ../../pmass.inc
