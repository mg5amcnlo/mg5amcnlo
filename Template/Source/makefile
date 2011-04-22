g77   = g77
FFLAGS= -O -ffixed-line-length-132
#FFLAGS= -g -ffixed-line-length-132 -fbounds-check -ffpe-trap=invalid,zero,overflow,underflow,denormal
LIBDIR= ../lib/
BINDIR= ../bin/
PDFDIR= ./PDF/

PROCESS= hfill.o matrix.o myamp.o 

DSAMPLE = dsample.o ranmar.o 

HBOOK = hfill.o hcurve.o hbook1.o hbook2.o

GENERIC = alfas_functions.o transpole.o invarients.o hfill.o pawgraphs.o ran1.o rw_events.o rw_routines.o kin_functions.o open_file.o basecode.o setrun.o run_printout.o \
          dgauss.o readgrid.o getissud.o

INCLUDEF= coupl.inc genps.inc hbook.inc DECAY/decay.inc psample.inc cluster.inc sudgrid.inc

BANNER  =  write_banner.o  rw_events.o ranmar.o  kin_functions.o open_file.o  rw_routines.o alfas_functions.o

COMBINE = combine_events.o  rw_events.o ranmar.o  kin_functions.o open_file.o rw_routines.o alfas_functions.o setrun.o

GENSUDGRID = gensudgrid.o is-sud.o setrun_gen.o rw_routines.o open_file.o

# $B$ add_def $E$ !this is a tag for add library GENERIC_MW  in MadWeight compilation


.f.o: ; $(g77) $(FFLAGS) -c $*.f

# $B$ mod_all $B$ !this is a tag for modify main compilation in MadWeight compilation. Don't edit this line
all: $(LIBDIR)libdhelas3.a $(LIBDIR)libdsample.a $(LIBDIR)libgeneric.a $(LIBDIR)libpdf.a  $(LIBDIR)libmodel.a $(LIBDIR)libcernlib.a
# $E$ mod_all $E$ !this is a tag for modify main compilation in MadWeight compilation. Don't edit this line

scaled.dat:
	combine_events

#$(BINDIR)write_banner:  $(BANNER) 
#	$(g77) $(FFLAGS) -o write_banner $(BANNER) $(LIBDIR)/libmodel.a $(LIBDIR)/libpdf.a 
#	mv write_banner $(BINDIR)

#$(BINDIR)scale_events: scale_events.o rw_events.o
#	$(g77) $(FFLAGS) -o scale_events scale_events.o rw_events.o
#	mv scale_events $(BINDIR)

#$(BINDIR)select_events: select_events.o rw_events.o
#	$(g77) $(FFLAGS) -o select_events select_events.o rw_events.o
#	mv select_events $(BINDIR)

$(BINDIR)sum_html: sum_html.o 
	$(g77) $(FFLAGS) -o sum_html sum_html.o   
	mv sum_html $(BINDIR)

$(LIBDIR)libdhelas3.a: DHELAS
	cd DHELAS; make
	ranlib libdhelas3.a
	mv libdhelas3.a $(LIBDIR)

$(LIBDIR)libdsample.a: $(DSAMPLE)
	ar cru libdsample.a $(DSAMPLE)
	ranlib libdsample.a
	mv libdsample.a $(LIBDIR)

$(LIBDIR)libgeneric.a: $(GENERIC)
	ar cru libgeneric.a $(GENERIC)
	ranlib libgeneric.a
	mv libgeneric.a $(LIBDIR)


$(LIBDIR)libpdf.a: PDF
	cd PDF; make	

$(LIBDIR)libcernlib.a: CERNLIB
	cd CERNLIB; make	

$(LIBDIR)libmodel.a: MODEL
	cd MODEL; make	

# $B$ add_compile $E$ !this is a tag for add library in MadWeight compilation

dsample.o: dsample.f genps.inc
invarients.o: invarients.f genps.inc
setrun.o: setrun.f nexternal.inc leshouche.inc genps.inc
sum_html.o: sum_html.f genps.inc
gen_ximprove.o: gen_ximprove.f run_config.inc
combine_events.o: combine_events.f run_config.inc
select_events.o: select_events.f run_config.inc
setrun.o: setrun.f nexternal.inc leshouche.inc

$(BINDIR)gen_ximprove: gen_ximprove.o ranmar.o rw_routines.o open_file.o
	$(g77) $(FFLAGS) -o gen_ximprove gen_ximprove.o ranmar.o rw_routines.o open_file.o
	mv gen_ximprove $(BINDIR)

$(BINDIR)combine_events: $(COMBINE)
	$(g77) $(FFLAGS) -o combine_events $(COMBINE) $(LIBDIR)libmodel.a $(LIBDIR)libpdf.a 
	mv combine_events $(BINDIR)

$(BINDIR)gensudgrid: $(GENSUDGRID) $(LIBDIR)libpdf.a $(LIBDIR)libcernlib.a
	$(g77) $(FFLAGS) -o gensudgrid $^
	mv gensudgrid $(BINDIR)

$(BINDIR)combine_runs: combine_runs.o rw_events.o
	$(g77) $(FFLAGS) -o combine_runs combine_runs.o rw_events.o
	mv combine_runs $(BINDIR)

