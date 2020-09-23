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
    $(error Bad lhadpfversion version 6 is now required)
  else
    ifeq ($(lhapdfsubversion),1) # 6.1.X
	PDF         = pdfwrap_lhapdf.o pdf_lhapdf6.o pdg2pdf_lhapdf6.o opendata.o PhotonFlux.o
    else # 6.2.X
        CXXFLAGS+=-std=c++11
	PDF         = pdfwrap_lhapdf.o pdf_lhapdf62.o pdg2pdf_lhapdf6.o opendata.o PhotonFlux.o
    endif    
  endif
else
  PDF         = Ctq6Pdf.o pdfwrap.o opendata.o pdf.o PhotonFlux.o pdg2pdf.o NNPDFDriver.o
endif

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(PDF)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
