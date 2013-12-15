include ../../Source/make_opts

LIBDIR = ../../lib/
PROG   = check
STABCHECKDRIVER = StabilityCheckDriver
LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel
LIBS =  $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS= loop_matrix.o improve_ps.o\
		 $(patsubst %.f,%.o,$(wildcard born_matrix.f)) \
		 loop_num.o CT_interface.o \
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

clean:
		@rm -f *.o
