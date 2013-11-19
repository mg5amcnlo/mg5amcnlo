include ../../Source/make_opts

LIBDIR = ../../lib/
PROG   = check
PJDIR = /Users/erdissshaw/Works/PJFry/pjfry-1.1.0-beta1/pjfry_install/lib/
STABCHECKDRIVER = StabilityCheckDriver
CHECK_SA_BORN_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel %(link_tir_libs)s -L$(PJDIR) -lpjfry
LIBS =  $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext) %(tir_libs)s $(PJDIR)libpjfry.$(libext)
PROCESS= loop_matrix.o improve_ps.o\
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard born_matrix.f)) \
		 loop_num.o CT_interface.o TIR_interface.o\
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard polynomial.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard MadLoopParamReader.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard helas_calls*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard jamp?_calls_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_born_amps_and_wfs.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_compute_loop_coefs.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_helas_calls*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard coef_construction_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard loop_CT_calls_*.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard mp_coef_construction_*.f))

$(PROG): check_sa.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)

$(STABCHECKDRIVER):  StabilityCheckDriver.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(STABCHECKDRIVER) StabilityCheckDriver.o $(PROCESS) $(LINKLIBS)

# The program below is not essential but just an helpful one to run the born only
$(CHECK_SA_BORN_SPLITORDERS):  check_sa_born_splitOrders.o born_matrix.o makefile $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
	$(FC) $(FFLAGS) -o $(CHECK_SA_BORN_SPLITORDERS) check_sa_born_splitOrders.o born_matrix.o -L$(LIBDIR) -ldhelas -lmodel

clean:
		@rm -f *.o
