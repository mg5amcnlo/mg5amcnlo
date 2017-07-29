include make_opts

LIBDIR= ../lib/
MCATNLOLIBDIR= ../MCatNLO/lib/
BINDIR= ../bin/
PDFDIR= ./PDF/
CUTTOOLSDIR= ./CutTools/
IREGIDIR= ./IREGI/src/
STDHEPDIR= ./StdHEP/

GENERIC = $(alfas_functions).o rw_routines.o kin_functions.o	\
	  run_printout.o dgauss.o ranmar.o setrun.o

.f.o: ; $(FC) $(FFLAGS) -c $*.f

all: $(LIBDIR)libdhelas.a $(LIBDIR)libgeneric.a $(LIBDIR)libpdf.a	\
    $(LIBDIR)libmodel.a $(LIBDIR)libcernlib.a param_card.inc
	rm -f PDF/*.o

$(LIBDIR)libdhelas.a: DHELAS
	rm -f $(LIBDIR)libdhelas.a
	cd DHELAS; make

$(LIBDIR)libgeneric.a: $(GENERIC)
	rm -f $(LIBDIR)libgeneric.a
	ar cru libgeneric.a $(GENERIC) extra_weights.o
	ranlib libgeneric.a
	mv libgeneric.a $(LIBDIR)
	cp -f extra_weights.mod $(LIBDIR)
	rm -f $(alfas_functions).o 

setrun.o: run_card.inc extra_weights.o
	$(FC) $(FFLAGS) -c -o setrun.o setrun.f

$(LIBDIR)libpdf.a: PDF
	rm -f $(LIBDIR)libpdf.a
	cd PDF; make	

$(LIBDIR)libcernlib.a: CERNLIB
	rm -f $(LIBDIR)libcernlib.a
	cd CERNLIB; make	

$(LIBDIR)libmodel.a: MODEL param_card.inc
	rm -f $(LIBDIR)libmodel.a
	cd MODEL; make	

run_card.inc: ../Cards/run_card.dat
	../bin/aMCatNLO treatcards run

param_card.inc: ../Cards/param_card.dat
	../bin/aMCatNLO treatcards param

CutTools: $(LIBDIR)libcts.a
libcuttools: $(LIBDIR)libcts.a

IREGI: $(LIBDIR)libiregi.a
libiregi: $(LIBDIR)libiregi.a

$(LIBDIR)libcts.a: $(CUTTOOLSDIR)
	cd $(CUTTOOLSDIR); make
	ln -sf ../Source/$(CUTTOOLSDIR)includects/libcts.a $(LIBDIR)libcts.a
	ln -sf ../Source/$(CUTTOOLSDIR)includects/mpmodule.mod $(LIBDIR)mpmodule.mod

$(LIBDIR)libiregi.a: $(IREGIDIR)
	cd $(IREGIDIR); make
	ln -sf ../Source/$(IREGIDIR)libiregi.a $(LIBDIR)libiregi.a

StdHEP: $(MCATNLOLIBDIR)libstdhep.a  

$(MCATNLOLIBDIR)libstdhep.a: $(STDHEPDIR)
	cd $(STDHEPDIR); make all
	ln -sf ../../Source/StdHEP/lib/libstdhep.a $(MCATNLOLIBDIR)libstdhep.a
	ln -sf ../../Source/StdHEP/lib/libFmcfio.a $(MCATNLOLIBDIR)libFmcfio.a

clean4pdf:
	rm -f ../lib/libpdf.a
	rm -f ../lib/libgeneric.a

cleanCT:
	cd $(CUTTOOLSDIR); make clean; cd ..

cleanIR:
	cd $(IREGIDIR); make clean; cd ..

cleanStdHEP:
	cd $(STDHEPDIR); make clean; cd ..

libmodel: $(LIBDIR)libmodel.a

libdhelas: $(LIBDIR)libdhelas.a

clean:
	rm -f *.o
	rm -f param_card.inc run_card.inc
	cd MODEL; make clean; cd ..
	cd CERNLIB; make clean; cd ..
	cd PDF; make clean; cd ..
	cd DHELAS; make clean; cd ..
	if [ -d $(CUTTOOLSDIR) ]; then cd $(CUTTOOLSDIR); make clean; cd ..; fi
	if [ -d $(IREGIDIR) ]; then cd $(IREGIDIR); make clean; cd ..; fi
	if [ -d $(STDHEPDIR) ]; then cd $(STDHEPDIR); make clean; cd ..; fi
