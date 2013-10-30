include ../../Source/make_opts

LIBDIR = ../../lib/
# BEGIN HSS
# SPECIAL FOR PJFRY AND IREGI
PJDIR = /Users/erdissshaw/Works/PJFry/pjfry-1.1.0-beta1/pjfry_install/lib/
IREGIDIR = /Users/erdissshaw/Works/TIR/iregi-ML5/iregi/
ONELOOPDIR=$(IREGIDIR)/oneloop/
QCDLOOPDIR=$(IREGIDIR)/qcdloop/ql/
FFDIR=$(IREGIDIR)/qcdloop/ff/
FFLAGS= $(FFLAGS) -I$(IREGIDIR)
# EDN HSS
PROG   = check
STABCHECKDRIVER = StabilityCheckDriver
CHECK_SA_BORN_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel -L$(PJDIR) -lpjfry -L$(IREGIDIR) -liregi -L$(ONELOOPDIR) -lavh_olo -L$(QCDLOOPDIR) -lqcdloop -L$(FFDIR) -lff
LIBS =  $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext) $(PJDIR)libpjfry.$(libext) $(IREGIDIR)libiregi.$(libext) $(ONELOOPDIR)libavh_olo.$(libext) $(QCDLOOPDIR)libqcdloop.$(libext) $(FFDIR)libff.$(libext)
PROCESS= loop_matrix.o improve_ps.o\
		 $(patsubst %.f,%.o,$(wildcard born_matrix.f)) \
		 loop_num.o CT_interface.o TIR_interface.o\
		 $(patsubst %.f,%.o,$(wildcard polynomial.f)) \
		 $(patsubst %.f,%.o,$(wildcard MadLoopParamReader.f)) \
		 $(patsubst %.f,%.o,$(wildcard helas_calls*.f)) \
		 $(patsubst %.f,%.o,$(wildcard jamp?_calls_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_born_amps_and_wfs.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_compute_loop_coefs.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_helas_calls*.f)) \
		 $(patsubst %.f,%.o,$(wildcard coef_construction_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard loop_CT_calls_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_coef_construction_*.f))

$(PROG): check_sa.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)

$(STABCHECKDRIVER):  StabilityCheckDriver.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(STABCHECKDRIVER) StabilityCheckDriver.o $(PROCESS) $(LINKLIBS)

# The program below is not essential but just an helpful one to run the born only
$(CHECK_SA_BORN_SPLITORDERS):  check_sa_born_splitOrders.o born_matrix.o makefile $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
	$(FC) $(FFLAGS) -o $(CHECK_SA_BORN_SPLITORDERS) check_sa_born_splitOrders.o born_matrix.o -L$(LIBDIR) -ldhelas -lmodel

clean:
		@rm -f *.o
