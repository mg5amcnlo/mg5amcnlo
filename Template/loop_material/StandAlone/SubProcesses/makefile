
ifeq ($(wildcard ../Source/make_opts),)
    ifeq ($(wildcard ../../Source/make_opts),)
       ROOT = ../../..
    else
       ROOT = ../..
    endif
else
    ROOT = ..
endif
LIBDIR = $(ROOT)/lib/

PROG   = check
all : $(PROG)

# For the compilation of the MadLoop file polynomial.f it makes a big difference to use -O3 and
# to turn off the bounds check. These can however be modified here if really necessary.
POLYNOMIAL_OPTIMIZATION = -O3
POLYNOMIAL_BOUNDS_CHECK =

include $(ROOT)/Source/make_opts
include $(ROOT)/SubProcesses/MadLoop_makefile_definitions
SHELL = /bin/bash

OLP    = OLP
STABCHECKDRIVER = StabilityCheckDriver
CHECK_SA_BORN_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS = -L$(LIBDIR) -ldhelas -lmodel $(LINK_LOOP_LIBS) $(LDFLAGS)
LIBS =  $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext) $(LOOP_LIBS)

PROCESS= MadLoopParamReader.o MadLoopCommons.o \
 $(patsubst $(DOTF),$(DOTO),$(wildcard polynomial.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard improve_ps.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard born_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard CT_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_num.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard jamp?_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_born_amps_and_wfs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_compute_loop_coefs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_CT_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard TIR_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard GOLEM_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard compute_color_flows.f))

OLP_PROCESS= MadLoopParamReader.o MadLoopCommons.o \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/polynomial.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/improve_ps.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/born_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/CT_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_num.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/jamp?_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_born_amps_and_wfs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_compute_loop_coefs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_CT_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/TIR_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/GOLEM_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/compute_color_flows.f))

$(PROG): check_sa.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)

$(STABCHECKDRIVER):  StabilityCheckDriver.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(STABCHECKDRIVER) StabilityCheckDriver.o $(PROCESS) $(LINKLIBS)

# The program below is not essential but just an helpful one to run the born only
$(CHECK_SA_BORN_SPLITORDERS):  check_sa_born_splitOrders.o $(patsubst $(DOTF),$(DOTO),$(wildcard *born_matrix.f)) makefile $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
	$(FC) $(FFLAGS) -o $(CHECK_SA_BORN_SPLITORDERS) check_sa_born_splitOrders.o $(patsubst $(DOTF),$(DOTO),$(wildcard *born_matrix.f)) -L$(LIBDIR) -ldhelas -lmodel

# This is the core of madloop computationally wise, so make sure to turn optimizations on and bound checks off.
# We use %olynomial.o and not directly polynomial.o because we want it to match when both doing make check here
# or make OLP one directory above
%olynomial.o : %olynomial.f
	$(FC) $(patsubst -O%,, $(subst -fbounds-check,,$(FFLAGS))) $(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) -c $< -o $@ $(LOOP_INCLUDE)

$(DOTO) : $(DOTF)
	$(FC) $(FFLAGS) -c $< -o $@ $(LOOP_INCLUDE)

$(OLP): $(OLP_PROCESS) $(LIBS)
	$(FC) -shared $(OLP_PROCESS) -o libMadLoop.$(dylibext) $(LINKLIBS)

$(OLP)_static: $(OLP_PROCESS)
	ar rcs libMadLoop.$(libext) $(OLP_PROCESS)
	mv libMadLoop.$(libext) $(MADLOOP_LIB)

../$(OLP):
	rm -f libMadLoop.$(dylibext)
	ln -s ../libMadLoop.$(dylibext)
	cd $(ROOT)/SubProcesses; make $(OLP)

../$(OLP)_static:
	cd $(ROOT)/SubProcesses; make $(OLP)_static

libMadLoop.$(dylibext): ../$(OLP)

ifeq ($(origin MENUM),undefined)
  MENUM=2
endif
matrix$(MENUM)py.so: ../$(OLP)_static f2py_wrapper.f 
	touch __init__.py
	$(F2PY) $(MADLOOP_LIB) -m matrix$(MENUM)py -c f2py_wrapper.f --fcompiler=$(FC)   -L../../lib/ -ldhelas -lmodel $(LINK_LOOP_LIBS) $(STDLIB)

clean:
	@rm -f *.o *.so *.$(libext) *.$(dylibext)
