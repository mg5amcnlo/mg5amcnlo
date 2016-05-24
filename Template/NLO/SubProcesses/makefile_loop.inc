include ../../../Source/make_opts

LIBDIR = ../../../lib/
LOOPLIB= libMadLoop.a

# For the compilation of the MadLoop file polynomial.f it makes a big difference to use -O3 and
# to turn off the bounds check. These can however be modified here if really necessary.
POLYNOMIAL_OPTIMIZATION = -O3
POLYNOMIAL_BOUNDS_CHECK =

LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel %(link_tir_libs)s
LIBS = $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext)	\
	$(LIBDIR)libmodel.$(libext) %(tir_libs)s
PROCESS= $(patsubst %(dotf)s,%(doto)s,$(wildcard polynomial.f)) \
         loop_matrix.o improve_ps.o born_matrix.o loop_num.o CT_interface.o	MadLoopCommons.o \
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
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard GOLEM_interface.f)) \
		 $(patsubst %(dotf)s,%(doto)s,$(wildcard compute_color_flows.f))

# This is the core of madloop computationally wise, so make sure to turn optimizations on and bound checks off.
# We use %%olynomial.o and not directly polynomial.o because we want it to match when both doing make check here
# or make OLP one directory above
%%olynomial.o : %%olynomial.f
	$(FC) $(patsubst -O%%,, $(subst -fbounds-check,,$(FFLAGS))) $(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) -c $< -o $@ $(LOOP_INCLUDE)

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
