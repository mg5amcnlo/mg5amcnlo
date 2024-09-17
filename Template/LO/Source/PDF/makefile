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
ifneq (,$(filter edff chff, $(pdlabel1) $(pdlabel2)))
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
