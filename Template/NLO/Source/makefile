include make_opts

LIBDIR= ../lib/
MCATNLOLIBDIR= ../MCatNLO/lib/
BINDIR= ../bin/
PDFDIR= ./PDF/
CUTTOOLSDIR= ./CutTools/
IREGIDIR= ./IREGI/src/
STDHEPDIR= ./StdHEP/

PROCESS= hfill.o matrix.o myamp.o 

HBOOK = hfill.o hcurve.o hbook1.o hbook2.o

GENERIC = $(alfas_functions).o invarients.o hfill.o pawgraphs.o ran1.o rw_events.o rw_routines.o kin_functions.o \
	  open_file.o basecode.o setrun.o run_printout.o dgauss.o ranmar.o getissud.o

INCLUDEF= coupl.inc genps.inc hbook.inc psample.inc pmass.inc nexternal.inc cluster.inc

BANNER  =  write_banner.o  rw_events.o ranmar.o  kin_functions.o open_file.o  rw_routines.o alfas_functions.o

COMBINE = combine_events.o  rw_events.o ranmar.o  kin_functions.o open_file.o rw_routines.o alfas_functions.o setrun.o



.f.o: ; $(FC) $(FFLAGS) -c $*.f

all: $(LIBDIR)libdhelas.a $(LIBDIR)libgeneric.a $(LIBDIR)libpdf.a  $(LIBDIR)libmodel.a  $(LIBDIR)libcernlib.a \
    param_card.inc
	rm -f PDF/*.o

scaled.dat:
	combine_events

$(BINDIR)sum_html: sum_html.o 
	$(FC) $(FFLAGS) -o sum_html sum_html.o   
	mv sum_html $(BINDIR)

$(LIBDIR)libdhelas.a: DHELAS
	rm -f $(LIBDIR)libdhelas.a
	cd DHELAS; make

$(LIBDIR)libgeneric.a: $(GENERIC)
	rm -f $(LIBDIR)libgeneric.a
	ar cru libgeneric.a $(GENERIC)
	ranlib libgeneric.a
	mv libgeneric.a $(LIBDIR)
	rm -f $(alfas_functions).o 

setrun.o: run_card.inc 
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

genps.inc: nexternal.inc
	touch genps.inc

run_card.inc: ../Cards/run_card.dat
	../bin/aMCatNLO treatcards run

param_card.inc: ../Cards/param_card.dat
	../bin/aMCatNLO treatcards param

$(BINDIR)gen_ximprove: gen_ximprove.o ranmar.o rw_routines.o open_file.o
	$(FC) $(FFLAGS) -o gen_ximprove gen_ximprove.o ranmar.o rw_routines.o open_file.o
	mv gen_ximprove $(BINDIR)

$(BINDIR)combine_events: $(COMBINE)
	$(FC) $(FFLAGS) -o combine_events $(COMBINE) $(LIBDIR)/libmodel.a $(LIBDIR)/libpdf.a 
	mv combine_events $(BINDIR)

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
	rm -f $(BINDIR)/combine_events $(BINDIR)/gen_ximprove
