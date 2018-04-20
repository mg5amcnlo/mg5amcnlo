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
  ifeq ($(lhapdfversion),5)
    PDF         = pdfwrap_lhapdf.o pdf_lhapdf.o pdg2pdf_lhapdf.o opendata.o  PhotonFlux.o
  else
    PDF         = pdfwrap_lhapdf.o pdf_lhapdf6.o pdg2pdf_lhapdf6.o opendata.o PhotonFlux.o
  endif
else
  PDF         = Ctq6Pdf.o pdfwrap.o opendata.o pdf.o PhotonFlux.o pdg2pdf.o NNPDFDriver.o
endif

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(PDF)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
