# ----------------------------------------------------------------------------
#
# Makefile for PDF library
# Apr. 25 2003
#
# ----------------------------------------------------------------------------


gfortran           = gfortran
FFLAGS        = -O
LIBRARY	      = ../libpdf.a
LIBDIR        = ../../lib/
PDF           = Ctq4Fn.o Ctq5Par.o Ctq5Pdf.o Partonx5.o Ctq6Pdf.o cteq3.o \
                mrs98.o mrs98lo.o mrs98ht.o mrs99.o mrst2001.o mrst2002.o jeppe02.o \
                pdfwrap.o opendata.o pdf.o PhotonFlux.o pdg2pdf.o

.f.o: ; $(gfortran) $(FFLAGS) -c $*.f

all: $(LIBDIR)libpdf.a

$(LIBDIR)libpdf.a: $(PDF)
	ar cru libpdf.a $(PDF)
	ranlib libpdf.a
	mv libpdf.a  $(LIBDIR)
