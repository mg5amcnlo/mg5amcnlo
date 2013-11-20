ifeq ($(wildcard ../../Source/make_opts),) 
    ROOT = ..
else 
    ROOT = ../..
endif


if [ -a ../../Source/make_opts]; then ROOT = ../..; fi;

if [ -a ../Source/make_opts]; then ROOT = ../; fi;

include $(ROOT)/Source/make_opts
LIBDIR = $(ROOT)/lib/

PROG   = check
OLP    = OLP
STABCHECKDRIVER = StabilityCheckDriver
CHECK_SA_BORN_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel
LIBS =  $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS= MadLoopParamReader.o MadLoopCommons.o \
		 $(patsubst %.f,%.o,$(wildcard loop_matrix.f)) \
		 $(patsubst %.f,%.o,$(wildcard improve_ps.f)) \
		 $(patsubst %.f,%.o,$(wildcard born_matrix.f)) \
		 $(patsubst %.f,%.o,$(wildcard CT_interface.f)) \
		 $(patsubst %.f,%.o,$(wildcard loop_num.f)) \
		 $(patsubst %.f,%.o,$(wildcard polynomial.f)) \
		 $(patsubst %.f,%.o,$(wildcard helas_calls*.f)) \
		 $(patsubst %.f,%.o,$(wildcard jamp?_calls_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_born_amps_and_wfs.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_compute_loop_coefs.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_helas_calls*.f)) \
		 $(patsubst %.f,%.o,$(wildcard coef_construction_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard loop_CT_calls_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_coef_construction_*.f))

OLP_PROCESS= MadLoopParamReader.o MadLoopCommons.o \
		 $(patsubst %.f,%.o,$(wildcard P*/loop_matrix.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/improve_ps.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/born_matrix.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/CT_interface.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/loop_num.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/polynomial.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/helas_calls*.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/jamp?_calls_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/mp_born_amps_and_wfs.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/mp_compute_loop_coefs.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/mp_helas_calls*.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/coef_construction_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/loop_CT_calls_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard P*/mp_coef_construction_*.f))

$(PROG): check_sa.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)

$(STABCHECKDRIVER):  StabilityCheckDriver.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(STABCHECKDRIVER) StabilityCheckDriver.o $(PROCESS) $(LINKLIBS)

# The program below is not essential but just an helpful one to run the born only
$(CHECK_SA_BORN_SPLITORDERS):  check_sa_born_splitOrders.o $(patsubst %.f,%.o,$(wildcard *born_matrix.f)) makefile $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
	$(FC) $(FFLAGS) -o $(CHECK_SA_BORN_SPLITORDERS) check_sa_born_splitOrders.o $(patsubst %.f,%.o,$(wildcard *born_matrix.f)) -L$(LIBDIR) -ldhelas -lmodel

%.o : %.f 
	$(FC) $(FFLAGS) -c $< -o $@

$(OLP): $(OLP_PROCESS) $(LIBS)
	$(FC) -shared $(OLP_PROCESS) -o libMadLoop.so $(LINKLIBS)

clean:
		@rm -f *.o
