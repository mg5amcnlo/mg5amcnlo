# ----------------------------------------------------------------------------
#
# Makefile for PDF library
# Apr. 25 2003
#
# ----------------------------------------------------------------------------

LIBRARY	      = libpdf.$(libext)
LIBDIR        = ../../lib/

include ../make_opts

PDF = opendata.o PhotonFlux.o
TOREMOVE =  # file that need to be remove since not linked in current library. important to force recompilation
TOCOMPILE = # the libraries that should be compiled


# check if we need to link to Electroweak PDF
ifneq (,$(filter eva, $(pdlabel1) $(pdlabel2)))
#  go trough here if pdlabel1 or pdlabel2 is on "eva"
  TOREMOVE += ElectroweakFlux_dummy.o
  # remove it to force recompilation of the library if back on it
  PDF += ElectroweakFluxDriver.o ElectroweakFlux.o
else
#  go trough here if NEITHER pdlabel1 or pdlabel2 is on "eva"
   PDF += ElectroweakFlux_dummy.o
   TOREMOVE += ElectroweakFluxDriver.o
endif

# check if we need to link to gammaUPC
ifneq (,$(filter edff chff edfinin chfinin edfinxn chfinxn edfin0n chfin0n edfin1n chfin1n edfin2n chfin2n edfin3n chfin3n edfin4n chfin4n edfxnin chfxnin edfxnxn chfxnxn edfxn0n chfxn0n edfxn1n chfxn1n edfxn2n chfxn2n edfxn3n chfxn3n edfxn4n chfxn4n edf0nin chf0nin edf0nxn chf0nxn edf0n0n chf0n0n edf0n1n chf0n1n edf0n2n chf0n2n edf0n3n chf0n3n edf0n4n chf0n4n edf1nin chf1nin edf1nxn chf1nxn edf1n0n chf1n0n edf1n1n chf1n1n edf1n2n chf1n2n edf1n3n chf1n3n edf1n4n chf1n4n edf2nin chf2nin edf2nxn chf2nxn edf2n0n chf2n0n edf2n1n chf2n1n edf2n2n chf2n2n edf2n3n chf2n3n edf2n4n chf2n4n edf3nin chf3nin edf3nxn chf3nxn edf3n0n chf3n0n edf3n1n chf3n1n edf3n2n chf3n2n edf3n3n chf3n3n edf3n4n chf3n4n edf4nin chf4nin edf4nxn chf4nxn edf4n0n chf4n0n edf4n1n chf4n1n edf4n2n chf4n2n edf4n3n chf4n3n edf4n4n chf4n4n, $(pdlabel1) $(pdlabel2)))
# go through here if pdflabel1 or pdlabel2 is on "edff" or "chff"
   TOREMOVE += $(LIBDIR)libgammaUPC.$(libext)
# remove it to force recompilation of the library if back on it
   TOCOMPILE += makefile_gammaUPC
else
# go throught here if neither pdlabel1 or pdlabel2 is on "edff" or "chff"
   TOCOMPILE += makefile_gammaUPC_dummy
   TOREMOVE += $(LIBDIR)libgammaUPC.$(libext)
endif

ifdef lhapdf
  TOREMOVE +=  pdfwrap.o
  PDF         += pdfwrap_lhapdf.o pdg2pdf_lhapdf6.o
  ifeq ($(lhapdfversion),5)
    $(error Bad lhadpfversion version 6 is now required)
  else
    ifeq ($(lhapdfsubversion),1) # 6.1.X
	PDF         += pdf_lhapdf6.o
        TOREMOVE += pdf_lhapdf62.o
    else # 6.2.X
        CXXFLAGS+=-std=c++11
	PDF  += pdf_lhapdf62.o
	TOREMOVE += pdf_lhapdf6.o
    endif    
  endif
else
  TOREMOVE += pdfwrap_lhapdf.o
  PDF += Ctq6Pdf.o pdfwrap.o pdf.o pdg2pdf.o NNPDFDriver.o eepdf.o gridpdfaux.o dfint.o kerset.o
endif


all: reset $(LIBDIR)$(LIBRARY) $(TOCOMPILE)

reset: 
	echo "remove previous compilation $(TOREMOVE)"
	echo "need to compile $(PDF)"
	rm -rf $(TOREMOVE)


$(LIBDIR)$(LIBRARY): $(PDF)
	rm $(LIBDIR)$(LIBRARY) || echo "recompilation of $(LIBDIR)$(LIBRARY)"
	$(call CREATELIB, $@, $^)

makefile_gammaUPC:
	(cd ./gammaUPC ; make ; cd ../)

makefile_gammaUPC_dummy:
	(cd ./gammaUPC ; make -f makefile_dummy; cd ../)

clean:
	@($(RM) *.o $(LIBDIR)$(LIBRARY))
	@(cd ./gammaUPC ; make clean ; cd ../)
