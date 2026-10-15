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
#  go through here if pdlabel1 or pdlabel2 is on "eva"
  TOREMOVE += ElectroweakFlux_dummy.o
# remove it to force recompilation of the library if back on it
  PDF += ElectroweakFluxDriver.o ElectroweakFlux.o
else
#  go through here if NEITHER pdlabel1 or pdlabel2 is on "eva"
  PDF += ElectroweakFlux_dummy.o
  TOREMOVE += ElectroweakFluxDriver.o ElectroweakFlux.o 
endif

# check if we need to link to gammaUPC
ifneq (,$(filter edff chff edpf edmf chpf chmf edfinin chfinin edfinxn chfinxn edfin0n chfin0n edfin1n chfin1n edfin2n chfin2n edfin3n chfin3n edfin4n chfin4n edfxnin chfxnin edfxnxn chfxnxn edfxn0n chfxn0n edfxn1n chfxn1n edfxn2n chfxn2n edfxn3n chfxn3n edfxn4n chfxn4n edf0nin chf0nin edf0nxn chf0nxn edf0n0n chf0n0n edf0n1n chf0n1n edf0n2n chf0n2n edf0n3n chf0n3n edf0n4n chf0n4n edf1nin chf1nin edf1nxn chf1nxn edf1n0n chf1n0n edf1n1n chf1n1n edf1n2n chf1n2n edf1n3n chf1n3n edf1n4n chf1n4n edf2nin chf2nin edf2nxn chf2nxn edf2n0n chf2n0n edf2n1n chf2n1n edf2n2n chf2n2n edf2n3n chf2n3n edf2n4n chf2n4n edf3nin chf3nin edf3nxn chf3nxn edf3n0n chf3n0n edf3n1n chf3n1n edf3n2n chf3n2n edf3n3n chf3n3n edf3n4n chf3n4n edf4nin chf4nin edf4nxn chf4nxn edf4n0n chf4n0n edf4n1n chf4n1n edf4n2n chf4n2n edf4n3n chf4n3n edf4n4n chf4n4n edpinin edpinxn edpin0n edpin1n edpin2n edpin3n edpin4n edpxnin edpxnxn edpxn0n edpxn1n edpxn2n edpxn3n edpxn4n edp0nin edp0nxn edp0n0n edp0n1n edp0n2n edp0n3n edp0n4n edp1nin edp1nxn edp1n0n edp1n1n edp1n2n edp1n3n edp1n4n edp2nin edp2nxn edp2n0n edp2n1n edp2n2n edp2n3n edp2n4n edp3nin edp3nxn edp3n0n edp3n1n edp3n2n edp3n3n edp3n4n edp4nin edp4nxn edp4n0n edp4n1n edp4n2n edp4n3n edp4n4n edminin edminxn edmin0n edmin1n edmin2n edmin3n edmin4n edmxnin edmxnxn edmxn0n edmxn1n edmxn2n edmxn3n edmxn4n edm0nin edm0nxn edm0n0n edm0n1n edm0n2n edm0n3n edm0n4n edm1nin edm1nxn edm1n0n edm1n1n edm1n2n edm1n3n edm1n4n edm2nin edm2nxn edm2n0n edm2n1n edm2n2n edm2n3n edm2n4n edm3nin edm3nxn edm3n0n edm3n1n edm3n2n edm3n3n edm3n4n edm4nin edm4nxn edm4n0n edm4n1n edm4n2n edm4n3n edm4n4n chpinin chpinxn chpin0n chpin1n chpin2n chpin3n chpin4n chpxnin chpxnxn chpxn0n chpxn1n chpxn2n chpxn3n chpxn4n chp0nin chp0nxn chp0n0n chp0n1n chp0n2n chp0n3n chp0n4n chp1nin chp1nxn chp1n0n chp1n1n chp1n2n chp1n3n chp1n4n chp2nin chp2nxn chp2n0n chp2n1n chp2n2n chp2n3n chp2n4n chp3nin chp3nxn chp3n0n chp3n1n chp3n2n chp3n3n chp3n4n chp4nin chp4nxn chp4n0n chp4n1n chp4n2n chp4n3n chp4n4n chminin chminxn chmin0n chmin1n chmin2n chmin3n chmin4n chmxnin chmxnxn chmxn0n chmxn1n chmxn2n chmxn3n chmxn4n chm0nin chm0nxn chm0n0n chm0n1n chm0n2n chm0n3n chm0n4n chm1nin chm1nxn chm1n0n chm1n1n chm1n2n chm1n3n chm1n4n chm2nin chm2nxn chm2n0n chm2n1n chm2n2n chm2n3n chm2n4n chm3nin chm3nxn chm3n0n chm3n1n chm3n2n chm3n3n chm3n4n chm4nin chm4nxn chm4n0n chm4n1n chm4n2n chm4n3n chm4n4n, $(pdlabel1) $(pdlabel2)))
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
