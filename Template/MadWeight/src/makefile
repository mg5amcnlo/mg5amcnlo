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
LINKLIBS =  -L../../lib/ -ldhelas -lgeneric -lmodel  -lcernlib -lpdf -lblocks -ltools -lTF  
LIBS = $(LIBDIR)libdhelas.a $(LIBDIR)libgeneric.a  $(LIBDIR)libmodel.a  $(LIBDIR)libcernlib.a $(LIBDIR)libpdf.a $(LIBDIR)libblocks.a $(LIBDIR)libtools.a $(LIBDIR)libTF.a 
 
PROCESS= driver.o  setscales.o   d_choices.o cuts.o\
         $(patsubst %.f,%.o,$(wildcard auto_dsig*.f)) \
         $(patsubst %.f,%.o,$(wildcard matrix*.f))


MW_PS= main_code.o gen_ps.o initialization.o call_TF.o permutation.o multi_channel.o permutation_weight.o


$(PROG): $(PROCESS) $(MW_PS)  makefile $(LIBS) 
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(MW_PS) $(LINKLIBS)

transfer_card.inc:../../Cards/transfer_card.dat ../../Cards/MadWeight_card.dat
	python ../../bin/internal/madweight/Cards.py
madweight_card.inc:../../Cards/transfer_card.dat ../../Cards/MadWeight_card.dat
	python ../../bin/internal/madweight/Cards.py

driver.o: phasespace.inc nexternal.inc data.inc coupl.inc madweight_card.inc nb_tf.inc
initialization.o: madweight_card.inc TF_param.inc coupl.inc run.inc madweight_param.inc pmass.inc nexternal.inc maxamps.inc leshouche.inc nb_tf.inc
matrix.o:nexternal.inc coupl.inc
call_TF.o:transfer_card.inc ../../Cards/transfer_card.dat nb_tf.inc nexternal.inc TF_param.inc phasespace.inc run.inc 
gen_ps.o: data.inc phasespace.inc nexternal.inc coupl.inc madweight_param.inc props.inc run.inc
d_choices.o: d_choices.inc
multi_channel.o:  coupl.inc d_choices.inc
permutation.o:  nexternal.inc phasespace.inc
setscale.o : genps.inc nexternal.inc coupl.inc maxamps.inc leshouche.inc run.inc
cuts.o: maxparticles.inc nexternal.inc coupl.inc cuts.inc
permutation.o: phasespace.inc nexternal.inc data.inc coupl.inc madweight_card.inc  maxparticles.inc
maxparticles.inc:
	ln -s ../../Source/maxparticles.inc
cuts.inc:
	ln -s ../../Source/cuts.inc
