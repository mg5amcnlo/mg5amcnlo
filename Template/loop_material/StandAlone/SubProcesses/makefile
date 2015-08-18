
ifeq ($(wildcard ../Source/make_opts),)
    ifeq ($(wildcard ../../Source/make_opts),)
       ROOT = ../../..
    else
       ROOT = ../..
    endif
else
    ROOT = ..
endif

include $(ROOT)/Source/make_opts
include $(ROOT)/SubProcesses/MadLoop_makefile_definitions
SHELL = /bin/bash

LIBDIR = $(ROOT)/lib/
PROG   = check
all : $(PROG)
OLP    = OLP
STABCHECKDRIVER = StabilityCheckDriver
CHECK_SA_BORN_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS =  -L$(LIBDIR) -ldhelas -lmodel $(LINK_LOOP_LIBS)
LIBS =  $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext) $(LOOP_LIBS)

PROCESS= MadLoopParamReader.o MadLoopCommons.o \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard improve_ps.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard born_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard CT_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_num.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard polynomial.f)) \
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
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/improve_ps.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/born_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/CT_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_num.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/polynomial.f)) \
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

$(DOTO) : $(DOTF)
	$(FC) $(FFLAGS) -c $< -o $@ $(LOOP_INCLUDE)

$(OLP): $(OLP_PROCESS) $(LIBS)
	$(FC) -shared $(OLP_PROCESS) -o libMadLoop.$(dylibext) $(LINKLIBS)

$(OLP)_static: $(OLP_PROCESS)
	ar rcs libMadLoop.$(libext) $(OLP_PROCESS)
	mv libMadLoop.$(libext) $(MADLOOP_LIB)

clean:
	@rm -f *.o *.so

../$(OLP):
	cd ..; make $(OLP); cd -
	rm -f libMadLoop.dylib
	ln -s ../libMadLoop.dylib

libMadLoop.dylib: ../$(OLP)

ifeq ($(origin MENUM),undefined)
  MENUM=2
endif
matrix$(MENUM)py.so: ../$(OLP)
	touch __init__.py
	$(F2PY) -L. -lMadLoop -m matrix$(MENUM)py -c check_sa.f --fcompiler=$(FC)