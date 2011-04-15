# ----------------------------------------------------------------------------
#
# Makefile for PDF library
# Apr. 25 2003
#
# ----------------------------------------------------------------------------


gfortran           = f77
FFLAGS        = -O -ffixed-line-length-132
LIBRARY	      = ../libTF.a
LIBDIR        = ../../../lib/

OBJ           =  transfer_functions_fct.o 

.f.o: ; $(gfortran) $(FFLAGS) -c $*.f

all: $(LIBDIR)libTF.a

$(LIBDIR)libTF.a: $(OBJ)
	ar cru libTF.a $(OBJ)
	ranlib libTF.a
	mv libTF.a  $(LIBDIR)

transfer_functions_fct.f: ../../nexternal.inc TF_param.inc

