ifeq ($(wildcard ../../Source/make_opts),)
    ROOT = ..
else
    ROOT = ../..
endif

include $(ROOT)/Source/make_opts
LIBDIR = $(ROOT)/lib/

PROG   = check
OLP    = OLP
STABCHECKDRIVER = StabilityCheckDriver
CHECK_SA_BORN_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel %(link_tir_libs)s
LIBS =  $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext) %(tir_libs)s
PROCESS= MadLoopParamReader.o MadLoopCommons.o \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard loop_matrix.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard improve_ps.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard born_matrix.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard CT_interface.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard loop_num.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard polynomial.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard helas_calls*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard jamp?_calls_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_born_amps_and_wfs.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_compute_loop_coefs.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_helas_calls*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard coef_construction_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard loop_CT_calls_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_coef_construction_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard TIR_interface.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard GOLEM_interface.f))

OLP_PROCESS= MadLoopParamReader.o MadLoopCommons.o \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/loop_matrix.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/improve_ps.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/born_matrix.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/CT_interface.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/loop_num.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/polynomial.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/helas_calls*.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/jamp?_calls_*.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/mp_born_amps_and_wfs.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/mp_compute_loop_coefs.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/mp_helas_calls*.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/coef_construction_*.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/loop_CT_calls_*.f)) \
                 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/mp_coef_construction_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/TIR_interface.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard P*/GOLEM_interface.f))

$(PROG): check_sa.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)

$(STABCHECKDRIVER):  StabilityCheckDriver.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(STABCHECKDRIVER) StabilityCheckDriver.o $(PROCESS) $(LINKLIBS)

# The program below is not essential but just an helpful one to run the born only
$(CHECK_SA_BORN_SPLITORDERS):  check_sa_born_splitOrders.o $(patsubst %(dotf)s,%(doto)s,$(wildcard *born_matrix.f)) makefile $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
	$(FC) $(FFLAGS) -o $(CHECK_SA_BORN_SPLITORDERS) check_sa_born_splitOrders.o $(patsubst %(dotf)s,%(doto)s,$(wildcard *born_matrix.f)) -L$(LIBDIR) -ldhelas -lmodel

%(doto)s : %(dotf)s
	$(FC) $(FFLAGS) -c $< -o $@ %(tir_include)s

$(OLP): $(OLP_PROCESS) $(LIBS)
	$(FC) -shared $(OLP_PROCESS) -o libMadLoop.$(dylibext) $(LINKLIBS)

clean:
		@rm -f *.o
