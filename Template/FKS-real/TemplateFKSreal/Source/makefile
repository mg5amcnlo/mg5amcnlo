include make_opts

LIBDIR= ../lib/
BINDIR= ../bin/
PDFDIR= ./PDF/

PROCESS= hfill.o matrix.o myamp.o 

DSAMPLE = dsample.o ranmar.o 
DSAMPLE_FKS = dsample_fks.o ranmar.o 

HBOOK = hfill.o hcurve.o hbook1.o hbook2.o

GENERIC = $(alfas_functions).o transpole.o invarients.o hfill.o pawgraphs.o ran1.o rw_events.o rw_routines.o kin_functions.o open_file.o basecode.o setrun.o run_printout.o dgauss.o

INCLUDEF= coupl.inc genps.inc hbook.inc DECAY/decay.inc psample.inc pmass.inc nexternal.inc cluster.inc

BANNER  =  write_banner.o  rw_events.o ranmar.o  kin_functions.o open_file.o  rw_routines.o alfas_functions.o

COMBINE = combine_events.o  rw_events.o ranmar.o  kin_functions.o open_file.o rw_routines.o alfas_functions.o setrun.o



.f.o: ; $(F77) $(FFLAGS) -c $*.f

all: $(LIBDIR)libdhelas.a $(LIBDIR)libdsample.a $(LIBDIR)libdsample_fks.a $(LIBDIR)libgeneric.a $(LIBDIR)libpdf.a  $(LIBDIR)libmodel.a

scaled.dat:
	combine_events

$(BINDIR)sum_html: sum_html.o 
	$(F77) $(FFLAGS) -o sum_html sum_html.o   
	mv sum_html $(BINDIR)

$(LIBDIR)libdhelas.a: DHELAS
	cd DHELAS; make

$(LIBDIR)libdsample.a: $(DSAMPLE)
	ar cru libdsample.a $(DSAMPLE)
	ranlib libdsample.a
	mv libdsample.a $(LIBDIR)

$(LIBDIR)libdsample_fks.a: $(DSAMPLE_FKS)
	ar cru libdsample_fks.a $(DSAMPLE_FKS)
	ranlib libdsample_fks.a
	mv libdsample_fks.a $(LIBDIR)

$(LIBDIR)libgeneric.a: $(GENERIC)
	ar cru libgeneric.a $(GENERIC)
	ranlib libgeneric.a
	mv libgeneric.a $(LIBDIR)


$(LIBDIR)libpdf.a: PDF
	cd PDF; make	

$(LIBDIR)libmodel.a: MODEL
	cd MODEL; make	


dsample.o: dsample.f genps.inc
dsample_fks.o: dsample_fks.f genps.inc

genps.inc: nexternal.inc
	touch genps.inc

$(BINDIR)gen_ximprove: gen_ximprove.o ranmar.o rw_routines.o open_file.o
	$(F77) $(FFLAGS) -o gen_ximprove gen_ximprove.o ranmar.o rw_routines.o open_file.o
	mv gen_ximprove $(BINDIR)

$(BINDIR)combine_events: $(COMBINE)
	$(F77) $(FFLAGS) -o combine_events $(COMBINE) $(LIBDIR)/libmodel.a $(LIBDIR)/libpdf.a 
	mv combine_events $(BINDIR)

