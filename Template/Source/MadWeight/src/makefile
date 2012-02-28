include ../../Source/make_opts
FFLAGS+= -w

#gfortran   = g77
#FFLAGS= -O -ffixed-line-length-132
#FFLAGS = -g -ffixed-line-length-132
LIBDIR = ../../lib/
BINDIR = ../../bin/
PROG   = comp_madweight
#LINKLIBS =  -L../../lib/ -ldhelas3 -lmodel -lgeneric_mw -lcernlib -lpdf -lblocks -ltools -lTF 
#LIBS = $(LIBDIR)libdhelas3.a $(LIBDIR)libgeneric_mw.a $(LIBDIR)libpdf.a $(LIBDIR)libmodel.a $(LIBDIR)libcernlib.a $(LIBDIR)libblocks.a $(LIBDIR)libtools.a $(LIBDIR)libTF.a 
LINKLIBS =  -L../../lib/ -ldhelas -lmodel -lgeneric  -lcernlib -lpdf -lblocks -ltools -lTF 
LIBS = $(LIBDIR)libdhelas.a   $(LIBDIR)libmodel.a $(LIBDIR)libgeneric.a $(LIBDIR)libcernlib.a $(LIBDIR)libpdf.a $(LIBDIR)libblocks.a $(LIBDIR)libtools.a $(LIBDIR)libTF.a 

PROCESS= driver.o auto_dsig.o matrix.o  setscales.o   d_choices.o
MW_PS= main_code.o gen_ps.o initialization.o call_TF.o permutation.o multi_channel.o


$(PROG): $(PROCESS) $(MW_PS)  makefile $(LIBS) 
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(MW_PS) $(LINKLIBS)

transfer_card.inc:../../Cards/transfer_card.dat ../../Cards/MadWeight_card.dat
	python ../../Source/MadWeight/Python/Cards.py
madweight_card.inc:../../Cards/transfer_card.dat ../../Cards/MadWeight_card.dat
	python ../../Source/MadWeight/Python/Cards.py
driver.o: phasespace.inc nexternal.inc data.inc coupl.inc madweight_card.inc 
initialization.o: madweight_card.inc
matrix.o:nexternal.inc coupl.inc
call_TF.o:transfer_card.inc ../../Cards/transfer_card.dat



