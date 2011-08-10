gfortran   = f77
FFLAGS= -O -ffixed-line-length-132
#FFLAGS = -g -ffixed-line-length-132
LIBDIR = ../../lib/
BINDIR = ../../bin/
PROG   = comp_madweight
LINKLIBS =  -L../../lib/ -ldhelas3 -lmodel -lgeneric_mw -lcernlib -lpdf -lblocks -ltools -lTF -lMWPS
LIBS = $(LIBDIR)libdhelas3.a $(LIBDIR)libgeneric_mw.a $(LIBDIR)libpdf.a $(LIBDIR)libmodel.a $(LIBDIR)libcernlib.a $(LIBDIR)libblocks.a $(LIBDIR)libtools.a $(LIBDIR)libTF.a $(LIBDIR)libMWPS.a

PROCESS= driver.o auto_dsig_MW.o matrix.o setcuts.o cuts_MW.o setscales.o MW_setrun.o setTF.o initcluster.o cluster_MW.o reweight.o d_choices.o
MW_PS= main_code.o gen_propa.o gen_mom.o get_PS_point.o get_feynman_content.o multi_channel.o transfer_functions.o myamp_mw.o


$(PROG): $(PROCESS) $(MW_PS)  makefile $(LIBS) $(include_files)
	$(gfortran) $(FFLAGS) -o $(PROG) $(PROCESS) $(MW_PS) $(LINKLIBS)

transfer_card.inc:../../Cards/transfer_card.dat ../../Cards/MadWeight_card.dat
	python ../../Source/MadWeight_File/Python/Cards.py
madweight_card.inc:../../Cards/transfer_card.dat ../../Cards/MadWeight_card.dat
	python ../../Source/MadWeight_File/Python/Cards.py
cuts_MW.o:genps.inc coupl.inc run.inc madweight_card.inc
driver.o: genps.inc nexternal.inc data.inc coupl.inc run.inc madweight_card.inc 
MW_setrun.o:madweight_card.inc
gen_mom.o:genps.inc data.inc
auto_dsig_MW.o:genps.inc coupl.inc run.inc
matrix.o:nexternal.inc coupl.inc
MW_setcuts.o:genps.inc coupl.inc run.inc leshouche.inc pmass.inc 
setscales.o:genps.inc coupl.inc run.inc leshouche.inc pmass.inc message.inc
gen_propa.o:genps.inc data.inc coupl.inc props.inc
get_PS_point.o:genps.inc data.inc 
get_feynman_content.o:genps.inc leshouche.inc
setTF.o:transfer_card.inc ../../Cards/transfer_card.dat
MW_setrun.o:madweight_card.inc 

%.o : %.f
	$(gfortran) $(FFLAGS) -c $<


