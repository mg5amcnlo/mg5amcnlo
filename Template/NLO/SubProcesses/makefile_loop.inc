include ../../../Source/make_opts

LIBDIR = ../../../lib/
LOOPLIB= libMadLoop.a

LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel %(link_tir_libs)s
LIBS = $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext)	\
	$(LIBDIR)libmodel.$(libext) %(tir_libs)s
PROCESS= loop_matrix.o improve_ps.o born_matrix.o loop_num.o CT_interface.o	MadLoopCommons.o \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard polynomial.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard MadLoopParamReader.f)) \
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

%(doto)s : %(dotf)s
	$(FC) $(FFLAGS) -c $< %(tir_include)s

all: cleanlib $(LOOPLIB) 

$(LOOPLIB): $(PROCESS)   
	@ar cr $(LOOPLIB) $(PROCESS)
	mv $(LOOPLIB) ../

clean: cleanlib
		@rm -f *.o 

cleanlib:
	rm -f ../$(LOOPLIB) $(PROG)
