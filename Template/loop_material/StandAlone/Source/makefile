# Definitions

LIBDIR= ../lib/
BINDIR= ../bin/
PDFDIR= ./PDF/
CUTTOOLSDIR= ./CutTools/
IREGIDIR= ./IREGI/src/

include make_opts

# Source files

PROCESS= hfill.o matrix.o myamp.o 
HBOOK = hfill.o hcurve.o hbook1.o hbook2.o
GENERIC = $(alfas_functions).o transpole.o invarients.o hfill.o pawgraphs.o ran1.o \
          rw_events.o rw_routines.o kin_functions.o open_file.o basecode.o setrun.o \
          run_printout.o dgauss.o readgrid.o getissud.o
INCLUDEF= coupl.inc genps.inc hbook.inc DECAY/decay.inc psample.inc cluster.inc sudgrid.inc
BANNER  =  write_banner.o  rw_events.o ranmar.o  kin_functions.o open_file.o  rw_routines.o alfas_functions.o
COMBINE = combine_events.o  rw_events.o ranmar.o  kin_functions.o open_file.o rw_routines.o alfas_functions.o setrun.o
GENSUDGRID = gensudgrid.o is-sud.o setrun_gen.o rw_routines.o open_file.o

# Locally compiled libraries
#
# CutTools and IREGI are only shipped *inside* this Source directory when
# MG5aMC is run with output_dependencies='internal' (see
# LoopExporterFortran.link_CutTools / link_IREGI). For the default
# 'external' setting -- and for 'environment_paths' -- only the ready-made
# libraries are symlinked into ../lib and Source/CutTools, Source/IREGI do
# not exist at all. Requiring them unconditionally here made the default
# 'make' in this directory abort with
#     No rule to make target `.../Source/CutTools', needed by `../lib/libcts.a'
# so only ask for them when they are actually present to be built.

LIBRARIES=
ifneq ($(wildcard $(CUTTOOLSDIR:%/=%)),)
LIBRARIES+= $(LIBDIR)libcts.a
endif
ifneq ($(wildcard $(IREGIDIR:%/=%)),)
LIBRARIES+= $(LIBDIR)libiregi.a
endif

# Compile commands

all: $(LIBRARIES) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
# Libraries
$(LIBDIR)libdhelas.$(libext): DHELAS
	cd DHELAS; make
$(LIBDIR)libmodel.$(libext): MODEL
	cd MODEL; make	 

CutTools: $(LIBDIR)libcts.a
libcuttools: $(LIBDIR)libcts.a

IREGI: $(LIBDIR)libiregi.a
libiregi: $(LIBDIR)libiregi.a

$(LIBDIR)libcts.a: $(CUTTOOLSDIR)
	cd $(CUTTOOLSDIR); make
	ln -sf ../Source/CutTools/includects/libcts.a $(LIBDIR)libcts.a
	ln -sf ../Source/CutTools/includects/mpmodule.mod $(LIBDIR)mpmodule.mod

$(LIBDIR)libiregi.a: $(IREGIDIR)
	cd $(IREGIDIR); make
	ln -sf ../Source/$(IREGIDIR)libiregi.a $(LIBDIR)libiregi.a

cleanCT:
	cd $(CUTTOOLSDIR); make clean; cd ..

cleanIR:
	cd $(IREGIDIR); make clean; cd ..

libdhelas: $(LIBDIR)libdhelas.$(libext)

libmodel: $(LIBDIR)libmodel.$(libext)

treatCardsLoopNoInit:
	echo "Card treatment not necessary in MadLoop standalone mode."

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
	rm -f *.o
	rm -f param_card.inc run_card.inc
	cd MODEL; make clean; cd ..
	cd DHELAS; make clean; cd ..
	if [ -d $(CUTTOOLSDIR) ]; then cd $(CUTTOOLSDIR); make clean; cd ..; fi
	if [ -d $(STDHEPDIR) ]; then cd $(STDHEPDIR); make clean; cd ..; fi
	rm -f $(BINDIR)/combine_events $(BINDIR)/gen_ximprove
