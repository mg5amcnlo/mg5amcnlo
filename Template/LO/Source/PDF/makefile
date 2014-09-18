# ----------------------------------------------------------------------------
#
# Makefile for PDF library
# Apr. 25 2003
#
# ----------------------------------------------------------------------------

LIBRARY	      = libpdf.$(libext)
LIBDIR        = ../../lib/

include ../make_opts

ifdef lhapdf
  PDF         = pdfwrap_lhapdf.o pdf_lhapdf.o pdg2pdf_lhapdf.o opendata.o
else
  PDF         = Ctq4Fn.o Ctq5Par.o Ctq5Pdf.o Partonx5.o Ctq6Pdf.o cteq3.o \
                mrs98.o mrs98lo.o mrs98ht.o mrs99.o mrst2001.o mrst2002.o jeppe02.o \
                pdfwrap.o opendata.o pdf.o PhotonFlux.o pdg2pdf.o NNPDFDriver.o
endif

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(PDF)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
