include ../../Source/make_opts
FFLAGS+= -w

# Definitions

LIBDIR = ../../lib/
BINDIR = ../../bin/
PROG   = madevent

LINKLIBS =  -L../../lib/ -ldhelas -ldsample -lmodel -lgeneric -lpdf -lcernlib $(lhapdf)

LIBS = $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libdsample.$(libext) $(LIBDIR)libgeneric.$(libext) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcernlib.$(libext)

# Source files

PROCESS= driver.o myamp.o genps.o unwgt.o setcuts.o get_color.o \
         cuts.o cluster.o reweight.o initcluster.o addmothers.o setscales.o \
	 idenparts.o \
         $(patsubst %.f,%.o,$(wildcard auto_dsig*.f)) \
         $(patsubst %.f,%.o,$(wildcard matrix*.f))
SYMMETRY = symmetry.o setcuts.o cuts.o cluster.o myamp.o genps.o \
	   initcluster.o setscales.o reweight.o get_color.o idenparts.o \
           $(patsubst %.f,%.o,$(wildcard matrix*.f))

# Binaries

$(PROG): $(PROCESS) auto_dsig.o $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(LINKLIBS)

gensym: $(SYMMETRY) configs.inc $(LIBS)
	$(FC) $(FFLAGS) -o gensym $(SYMMETRY) $(LINKLIBS)

$(LIBDIR)libmodel.$(libext): ../../Cards/param_card.dat
	cd ../../Source/MODEL; make

$(LIBDIR)libgeneric.$(libext): ../../Cards/run_card.dat
	cd ../../Source; make

$(LIBDIR)libpdf.$(libext): 
	cd ../../Source/PDF; make

# Dependencies

driver.f: genps.inc qmass.inc
setcuts.f: qmass.inc
qmass.inc: nexternal.inc
	   touch qmass.inc
symmetry.o: genps.inc nexternal.inc configs.inc run_config.inc
genps.o: genps.inc nexternal.inc configs.inc
cuts.o: genps.inc nexternal.inc pmass.inc
setcuts.o: genps.inc run_config.inc
invarients.o: genps.inc nexternal.inc
myamp.o: props.inc genps.inc nexternal.inc
reweight.o: sudakov.inc cluster.inc sudakov.inc run.inc message.inc \
	    run_config.inc
cluster.o: cluster.inc genps.inc nexternal.inc message.inc
addmothers.o: genps.inc nexternal.inc symswap.inc message.inc
unwgt.o: genps.inc nexternal.inc symswap.inc cluster.inc run.inc message.inc \
	 run_config.inc
initcluster.o: message.inc

clean:
	$(RM) *.o gensym madevent
