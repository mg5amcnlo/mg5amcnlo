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
    PDF         = pdfwrap_lhapdf.o pdf_lhapdf.o pdg2pdf_lhapdf.o opendata.o
  else
    ifdef applgrid
      PDF         = pdfwrap_lhapdf.o pdf_lhapdf.o pdg2pdf_lhapdf.o opendata.o
    else # lhapdf6
      ifeq ($(lhapdfsubversion),1) # 6.1.X
        PDF         = pdfwrap_lhapdf.o pdf_lhapdf6.o pdg2pdf_lhapdf6.o opendata.o
      else # 6.2.X
        CXXFLAGS+=-std=c++11    
        PDF         = pdfwrap_lhapdf.o pdf_lhapdf62.o pdg2pdf_lhapdf6.o opendata.o
      endif
    endif
  endif
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
