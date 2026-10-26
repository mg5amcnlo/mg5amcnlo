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
  else ifneq (,$(filter 0 1,$(lhapdfsubversion))) # 6.0.X, 6.1.X
    $(error LHAPDF 6.2 or later is required)
  else # lhapdf v6.2 or later
    CXXFLAGS+=-std=c++11
    PDF         = pdfwrap_lhapdf.o pdf_lhapdf62.o pdg2pdf_lhapdf6.o opendata.o
  endif
else ifdef epdf
      CXXFLAGS+=-std=c++11    
      PDF         = pdfwrap_emela.o pdg2pdf_epdf_mela.o opendata.o
else
  PDF         = Ctq4Fn.o Ctq5Par.o Ctq5Pdf.o CT14Pdf.o Partonx5.o Ctq6Pdf.o cteq3.o \
                mrs98.o mrs98lo.o mrs98ht.o mrs99.o mrst2001.o mrst2002.o jeppe02.o \
                pdfwrap.o opendata.o pdf.o PhotonFlux.o pdg2pdf.o NNPDFDriver.o \
		eepdf.o gridpdfaux.o dfint.o kerset.o
endif
# Ctq5Pdf.o  excluded

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(PDF)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
