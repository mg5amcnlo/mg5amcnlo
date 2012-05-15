include ../../Source/make_opts

LIBDIR = ../../lib/
PROG   = check
LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel
LIBS =  $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS= check_sa.o loop_matrix.o \
		 $(patsubst %.f,%.o,$(wildcard born_matrix.f)) \
		 loop_num.o CT_interface.o \
		 $(patsubst %.f,%.o,$(wildcard polynomial.f)) \		 
		 $(patsubst %.f,%.o,$(wildcard MadLoopParamReader.f)) \
		 $(patsubst %.f,%.o,$(wildcard helas_calls*.f)) \
		 $(patsubst %.f,%.o,$(wildcard jamp?_calls_*.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_born_amps_and_wfs.f)) \
		 $(patsubst %.f,%.o,$(wildcard mp_helas_calls*.f))


$(PROG):  $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(LINKLIBS)

clean:
		@rm -f *.o
