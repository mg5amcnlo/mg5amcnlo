# Definitions

LIBDIR= ../lib/
BINDIR= ../bin/
PDFDIR= ./PDF/
PWD = $(shell pwd)
CUTTOOLSDIR= $(PWD)/CutTools/

include make_opts

# Source files

PROCESS= hfill.o matrix.o myamp.o 
DSAMPLE = dsample.o ranmar.o 
HBOOK = hfill.o hcurve.o hbook1.o hbook2.o
GENERIC = $(alfas_functions).o transpole.o invarients.o hfill.o pawgraphs.o ran1.o \
          rw_events.o rw_routines.o kin_functions.o open_file.o basecode.o setrun.o \
          run_printout.o dgauss.o readgrid.o getissud.o
INCLUDEF= coupl.inc genps.inc hbook.inc DECAY/decay.inc psample.inc cluster.inc sudgrid.inc
BANNER  =  write_banner.o  rw_events.o ranmar.o  kin_functions.o open_file.o  rw_routines.o alfas_functions.o
COMBINE = combine_events.o  rw_events.o ranmar.o  kin_functions.o open_file.o rw_routines.o alfas_functions.o setrun.o
GENSUDGRID = gensudgrid.o is-sud.o setrun_gen.o rw_routines.o open_file.o

# Locally compiled libraries

LIBRARIES=$(LIBDIR)libdsample.$(libext) $(LIBDIR)libgeneric.$(libext)

# Compile commands

all: $(LIBRARIES) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcernlib.$(libext)

# Libraries

$(LIBDIR)libdsample.$(libext): $(DSAMPLE)
	$(call CREATELIB, $@, $^)
$(LIBDIR)libgeneric.$(libext): $(GENERIC)
	$(call CREATELIB, $@, $^)
$(LIBDIR)libdhelas.$(libext): DHELAS
	cd DHELAS; make
$(LIBDIR)libpdf.$(libext): PDF
	cd PDF; make	
$(LIBDIR)libcernlib.$(libext): CERNLIB
	cd CERNLIB; make	
$(LIBDIR)libmodel.$(libext): MODEL
	cd MODEL; make	 

CutTools: $(LIBDIR)libcts.a

$(LIBDIR)libcts.a: $(CUTTOOLSDIR)
	cd $(CUTTOOLSDIR); make
	ln -sf ../Source/CutTools/includects/libcts.a $(LIBDIR)libcts.a
	ln -sf ../Source/CutTools/includects/mpmodule.mod $(LIBDIR)mpmodule.mod

cleanCT:
	cd CUTTOOLS; make clean; cd ..

# Binaries

$(BINDIR)sum_html: sum_html.o 
	$(FC) $(FFLAGS) -o $@ $^
$(BINDIR)gen_ximprove: gen_ximprove.o ranmar.o rw_routines.o open_file.o
	$(FC) $(FFLAGS) -o $@ $^
$(BINDIR)combine_events: $(COMBINE) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libpdf.$(libext)
	$(FC) $(FFLAGS) -o $@ $(COMBINE) -L$(LIBDIR) -lmodel -lpdf $(lhapdf)
$(BINDIR)gensudgrid: $(GENSUDGRID) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libcernlib.$(libext)
	$(FC) $(FFLAGS) -o $@ $(GENSUDGRID) -L$(LIBDIR) -lmodel -lpdf -lcernlib $(lhapdf)
$(BINDIR)combine_runs: combine_runs.o rw_events.o
	$(FC) $(FFLAGS) -o $@ $^

# Dependencies

dsample.o: dsample.f genps.inc
invarients.o: invarients.f genps.inc
setrun.o: setrun.f nexternal.inc leshouche.inc genps.inc
sum_html.o: sum_html.f genps.inc
gen_ximprove.o: gen_ximprove.f run_config.inc
combine_events.o: combine_events.f run_config.inc
select_events.o: select_events.f run_config.inc
setrun.o: setrun.f nexternal.inc leshouche.inc

clean:
	$(RM) *.o $(LIBRARIES)
	if [ -d PDF ]; then cd PDF; make clean; cd ..; fi
	if [ -d CERNLIB ]; then cd CERNLIB; make clean; cd ..; fi
	cd DHELAS; make clean; cd ..
	cd MODEL; make clean; cd ..
