include ../../Source/make_opts

SHELL = /bin/bash
HERE := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
LIBDIR := $(abspath $(HERE)/../../lib)
PDIR := $(strip $(notdir $(patsubst %/,%,$(strip $(HERE)))))
PROG   = check
PROG_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS =  -L$(LIBDIR) -ldhelas -lmodel
LIBS = $(LIBDIR)/libdhelas.$(libext) $(LIBDIR)/libmodel.$(libext)
LIBS_SHARED = $(LIBDIR)/libdhelas.$(dylibext) $(LIBDIR)/libmodel.$(dylibext)
PROCESS=  matrix.o
CHECK_SA=  check_sa.o
CHECK_SA_SPLITORDERS=  check_sa_born_splitOrders.o
 
$(PROG): $(PROCESS) $(CHECK_SA) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(CHECK_SA) $(LINKLIBS)

$(PROG_SPLITORDERS): $(PROCESS) $(CHECK_SA_SPLITORDERS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(CHECK_SA_SPLITORDERS) $(LINKLIBS)

driver.f: nexternal.inc pmass.inc ngraphs.inc coupl.inc

# For python linking (require f2py part of numpy)
ifeq ($(origin MENUM),undefined)
  MENUM=2
endif

libme$(PDIR).$(dylibext): $(LIBDIR)/libdhelas.$(dylibext) $(LIBDIR)/libmodel.$(dylibext) matrix.o
	gfortran $(DYNLIBFLAG) -o libme$(PDIR).$(dylibext) matrix.o ../../Source/DHELAS/*.o ../../Source/MODEL/*.o


$(LIBDIR)/libdhelas.$(dylibext):
	$(MAKE) -C "$(LIBDIR)/../Source/DHELAS" shared
$(LIBDIR)/libmodel.$(dylibext):
	$(MAKE) -C "$(LIBDIR)/../Source/MODEL" shared

matrix$(MENUM)py.so: f2py_matrix_wrapper.f libme$(PDIR).$(dylibext)  makefile 
	touch __init__.py
	LDFLAGS="-Wl,-rpath,$(HERE)" $(F2PY)  -c f2py_matrix_wrapper.f -L$(HERE) -lme$(PDIR) $(LINKLIBS) -m matrix$(MENUM)py 
	touch matrix$(MENUM)py.so
	cp $(LIBDIR)/*$(dylibext) .


