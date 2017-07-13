include ../../Source/make_opts
FFLAGS+= -w

# Load additional dependencies of the bias module, if present
ifeq (,$(wildcard ../bias_dependencies))
BIASDEPENDENCIES =
else
include ../bias_dependencies
endif

# Definitions

LIBDIR = ../../lib/
BINDIR = ../../bin/
PROG   = madevent

ifneq ("$(wildcard ../MadLoop_makefile_definitions)","")
    include ../MadLoop_makefile_definitions
else
    LINK_LOOP_LIBS =
    LOOP_LIBS =
    LOOP_INCLUDE =
    LINK_MADLOOP_LIB =
    MADLOOP_LIB =
endif

LINKLIBS = $(LINK_MADLOOP_LIB) $(LINK_LOOP_LIBS) -L../../lib/ -ldhelas -ldsample -lmodel -lgeneric -lpdf -lcernlib $(llhapdf) -lbias 

LIBS = $(LIBDIR)libbias.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libdsample.$(libext) $(LIBDIR)libgeneric.$(libext) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcernlib.$(libext) $(MADLOOP_LIB) $(LOOP_LIBS)

# Source files

PROCESS= driver.o myamp.o genps.o unwgt.o setcuts.o get_color.o \
         cuts.o cluster.o reweight.o initcluster.o addmothers.o setscales.o \
	 idenparts.o dummy_fct.o \
         $(patsubst %.f,%.o,$(wildcard auto_dsig*.f)) \
         $(patsubst %.f,%.o,$(wildcard matrix*.f))

SYMMETRY = symmetry.o idenparts.o 

# Binaries

$(PROG): $(PROCESS) auto_dsig.o $(LIBS)
	$(FC) -o $(PROG) $(PROCESS) $(LINKLIBS) $(LDFLAGS) $(BIASDEPENDENCIES)

gensym: $(SYMMETRY) configs.inc $(LIBDIR)libmodel.$(libext) $(LIBDIR)libgeneric.$(libext)
	$(FC) -o gensym $(SYMMETRY) -L../../lib/ -lmodel -lgeneric -lpdf $(llhapdf) $(LDFLAGS)

$(LIBDIR)libmodel.$(libext): ../../Cards/param_card.dat
	cd ../../Source/MODEL; make

$(LIBDIR)libgeneric.$(libext): ../../Cards/run_card.dat
	cd ../../Source; make

$(LIBDIR)libpdf.$(libext): 
	cd ../../Source/PDF; make

# Add source so that the compiler finds the DiscreteSampler module.
%.o: %.f
	$(FC) $(FFLAGS) -c $< -I../../Source/

# Dependencies

driver.f: genps.inc
symmetry.o: genps.inc nexternal.inc configs.inc run_config.inc
genps.o: genps.inc nexternal.inc configs.inc
dummy_fct.0: run.inc genps.inc
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
