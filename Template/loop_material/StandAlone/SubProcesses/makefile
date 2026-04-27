
ifeq ($(wildcard ../Source/make_opts),)
    ifeq ($(wildcard ../../Source/make_opts),)
       ROOT = ../../..
    else
       ROOT = ../..
    endif
else
    ROOT = ..
endif
LIBDIR = $(abspath $(ROOT))/lib/

PROG   = check
all : $(PROG)

HERE := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
ROOTNAME = $(notdir $(abspath $(ROOT)))

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
DYLIBS =  $(LIBDIR)libdhelas.$(dylibext) $(LIBDIR)libmodel.$(dylibext) $(LOOP_LIBS)

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
 $(patsubst $(DOTF),$(DOTO),$(wildcard COLLIER_interface.f)) \
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
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/COLLIER_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/compute_color_flows.f))

POLYNOMIAL = $(patsubst $(DOTF),$(DOTO),$(wildcard polynomial.f))
OLP_POLYNOMIAL = $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/polynomial.f))



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
%oloop_matrix.o : %olynomial.o %oloop_matrix.f
%olynomial.o : %olynomial.f
	$(FC) $(patsubst -O%,, $(subst -fbounds-check,,$(FFLAGS))) $(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) -c $< -o $@ $(LOOP_INCLUDE)

%/%oloop_matrix.o : %/polynomial.o %/%oloop_matrix.f
	$(FC) $(patsubst -O%,,$(subst -fbounds-check,,$(FFLAGS))) \
		$(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) \
		-c $< -o $@ $(LOOP_INCLUDE)

%/polynomial.o : %/polynomial.f
	$(FC) $(patsubst -O%,,$(subst -fbounds-check,,$(FFLAGS))) \
		$(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) \
		-c $< -o $@ $(LOOP_INCLUDE)


$(DOTO) : $(DOTF) $(POLYNOMIAL) $(OLP_POLYNOMIAL)
	$(FC) $(FFLAGS) -c $< -o $@ $(LOOP_INCLUDE)

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

WRAPPER_SRCS := $(wildcard */f2py_wrapper.f)
WRAPPER_OBJS := $(patsubst %.f,%.o,$(wildcard */f2py_wrapper.f))

%/f2py_wrapper.o: %/f2py_wrapper.f
	$(MAKE) -C $* f2py_wrapper.o




ALL_DOTF := $(wildcard */polynomial.f */loop_matrix.f */improve_ps.f */born_matrix.f */CT_interface.f \
             */loop_num.f \
             */helas_calls*.f */mp_compute_loop_coefs.f */mp_helas_calls*.f */coef_construction_*.f \
             */loop_CT_calls_*.f */mp_coef_construction_*.f */TIR_interface.f */COLLIER_interface.f \
             MadLoopParamReader.f MadLoopCommons.f */GOLEM_interface.f */compute_color_flows.f \
             */mp_born_amps_and_wfs.f */jamp?_calls_*.f)

# Convert .f to .o
ALL_DOTO := $(patsubst %.f,%.o,$(ALL_DOTF))

ifeq ($(UNAME), Darwin)
  LIBALLME_DYNFLAG = -install_name @rpath/liball$(ROOTNAME)_$(MENUM)me.dylib -undefined dynamic_lookup
  WHOLE_ARCH=-Wl,-force_load,
  NOWHOLE_ARCH=
  STAT_LIB = $(WHOLE_ARCH)$(LIBDIR)libcts.$(libext) $(WHOLE_ARCH)$(LIBDIR)libiregi.$(libext)
  LD_F2PY=
else
  LIBALLME_DYNFLAG =
  WHOLE_ARCH=  -Wl,--whole-archive
  NOWHOLE_ARCH= -Wl,--no-whole-archive
  STAT_LIB = $(WHOLE_ARCH) $(LIBDIR)libcts.$(libext) $(LIBDIR)libiregi.$(libext) $(NOWHOLE_ARCH)
  LD_F2PY=-lgfortran -lquadmath
endif


liball$(ROOTNAME)_$(MENUM)me.$(dylibext): all_matrix.o libMadLoop.$(dylibext) $(WRAPPER_OBJS) $(LIBS)  $(OLP) 
	$(CXX) $(DYNLIBFLAG)  $(LIBALLME_DYNFLAG)  $(STDLIB_FLAG)  -o liball$(ROOTNAME)_$(MENUM)me.$(dylibext) all_matrix.o */f2py_wrapper.o ../Source/DHELAS/*.o ../Source/MODEL/*.o $(STAT_LIB) $(RPATH_LIBS) $(LINK_LOOP_LIBS) $(ALL_DOTO) $(STDLIB) $(LINK_LOOP_LIBS) 







shared: liball$(ROOTNAME)_$(MENUM)me.$(dylibext)

ifeq ($(origin MENUM),undefined)
  MENUM=2
endif



matrix$(MENUM)py.so: ../$(OLP)_static f2py_wrapper.f 
	touch __init__.py
	$(F2PY) $(MADLOOP_LIB) -m matrix$(MENUM)py -c f2py_wrapper.f --f77exec=$(FC)   -L../../lib/ -ldhelas -lmodel $(LINK_LOOP_LIBS) $(STDLIB)

allmatrix$(MENUM)py.so: $(OLP)_static all_matrix.f $(LIBS) $(WRAPPER)
	touch __init__.py
	$(F2PY) $(MADLOOP_LIB)  -m allmatrix$(MENUM)py -c all_matrix.f $(wildcard $(LOOP_PREFIX)*/f2py_wrapper.f) --f77exec=$(FC)   -L../lib/ -ldhelas -lmodel $(LINK_LOOP_LIBS) $(STDLIB) 


all_matrix$(MENUM)py.so:  liball$(ROOTNAME)_$(MENUM)me.$(dylibext) f2py_wrapper.f makefile
	LDFLAGS="-Wl,-rpath,$(HERE) -L$(HERE) $(RPATH_LIBS)  $(LINK_LOOP_LIBS) $(LD_F2PY)" $(F2PY)  -c f2py_wrapper.f -L$(HERE) -lall$(ROOTNAME)_$(MENUM)me -m all_matrix$(MENUM)py  
	touch all_matrix$(MENUM)py.so
	touch __init__.py


clean:
	@rm -f *.o *.so *.$(libext) *.$(dylibext)
