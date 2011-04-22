g77   = g77
FFLAGS= -O -w -ffixed-line-length-132
#FFLAGS = -g -ffixed-line-length-132 -fbounds-check -ffpe-trap=invalid,zero,overflow,underflow,denormal
LIBDIR = ../../lib/
BINDIR = ../../bin/
# $B$ obj_for_MW $B$ !this is a tag for MadWeight. Don't edit this line
PROG   = madevent
#    $E$ obj_for_MW $E$ !this is a tag for MadWeight. Don't edit this line
LINKLIBS =  -L../../lib/ -ldhelas3 -ldsample -lmodel -lgeneric -lpdf -lcernlib
LIBS = $(LIBDIR)libdhelas3.a $(LIBDIR)libdsample.a $(LIBDIR)libgeneric.a $(LIBDIR)libpdf.a $(LIBDIR)libmodel.a $(LIBDIR)libcernlib.a


PROCESS= driver.o myamp.o genps.o unwgt.o setcuts.o \
         cuts.o cluster.o reweight.o initcluster.o addmothers.o setscales.o \
         $(patsubst %.f,%.o,$(wildcard auto_dsig*.f)) \
         $(patsubst %.f,%.o,$(wildcard matrix*.f))
SYMMETRY = myamp.o genps.o initcluster.o setscales.o reweight.o\
           $(patsubst %.f,%.o,$(wildcard matrix*.f))

# $B$ obj_for_MW $E$ !this is a tag for MadWeight. Don't edit this line

%.o : %.f
	$(g77) $(FFLAGS) -c $<

# $B$ main_make $B$ !this is a tag for MadWeight. Don't edit this line
$(PROG): $(PROCESS) auto_dsig.o makefile $(LIBS)
	$(g77) $(FFLAGS) -o $(PROG) $(PROCESS) $(LINKLIBS)
#    $E$ main_make $E$ !this is a tag for MadWeight. Don't edit this line

symmetry.dat: mg.sym gensym
	./gensym

gensym: symmetry.o genps.inc setcuts.o cuts.o cluster.o $(SYMMETRY) configs.inc  
	$(g77) $(FFLAGS) -o gensym symmetry.o setcuts.o cuts.o cluster.o $(SYMMETRY) $(LINKLIBS)

driver.f: genps.inc qmass.inc
setcuts.f: qmass.inc

qmass.inc: nexternal.inc
	touch qmass.inc

symmetry.o: genps.inc nexternal.inc configs.inc ../../Source/run_config.inc
genps.o: genps.inc nexternal.inc configs.inc
cuts.o: genps.inc nexternal.inc pmass.inc
invarients.o: genps.inc nexternal.inc
myamp.o: props.inc genps.inc nexternal.inc
reweight.o: sudakov.inc cluster.inc sudakov.inc run.inc message.inc
cluster.o: cluster.inc genps.inc nexternal.inc message.inc
addmothers.o: genps.inc nexternal.inc symswap.inc message.inc
unwgt.o: genps.inc nexternal.inc symswap.inc cluster.inc run.inc message.inc
initcluster.o: message.inc
