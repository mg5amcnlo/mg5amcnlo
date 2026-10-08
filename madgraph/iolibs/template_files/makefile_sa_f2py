include ../Source/make_opts
SHELL = /bin/bash
HERE := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
LIBDIR = $(HERE)/../lib/
PROG   = check
LINKLIBS_ME =  -L$(LIBDIR) -ldhelas -lmodel
LINKLIBS_ALL =  -L$(LIBDIR) -lmatrix -ldhelas -lmodel
LIBS = $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS=  allmatrix.o
LIBRARY = libmatrix.a

ifeq ($(strip $(HERE)),)
    HERE := ./
endif


PROCNAME= $(notdir $(dir $(abspath $(HERE))))

# For python linking (require f2py part of numpy)
ifeq ($(origin MENUM),undefined)
  MENUM=2
endif

#%$(MENUM)py.so: %/matrix.f
#	$(F2PY) --f77exec=$(FC) $(LINKLIBS_ME) -c $^ -m $(patsubst %/matrix.f,%$(MENUM)py,$^) --include-paths=$(HERE)/$(patsubst %/matrix.f,%,$^)
	
	
#all_matrix$(MENUM)py.so: $(LIBDIR)/$(LIBRARY) all_matrix.f
#	$(F2PY) --f77exec=$(FC) $(LINKLIBS_ALL) -c all_matrix.f -m all_matrix$(MENUM)py --include-paths=$(HERE)

    
#allmatrix$(MENUM)py.so: $(patsubst %/matrix.f,%$(MENUM)py.so,$(wildcard */matrix.f)) all_matrix$(MENUM)py.so
#	for lib in $(patsubst %.so,%,$^); do \
#		printf "try:\n    from $$lib import *\nexcept:\n    from .$$lib import *\n" >> allmatrix$(MENUM)py.py; \
#	done
	
	
#$(LIBDIR)/$(LIBRARY): $(patsubst %.f,%.o,$(wildcard */matrix.f))  all_matrix.o
#	$(call CREATELIB, $@, $^)  


$(LIBDIR)/libdhelas.$(dylibext):
	$(MAKE) -C "$(LIBDIR)/../Source/DHELAS" shared
$(LIBDIR)/libmodel.$(dylibext):
	$(MAKE) -C "$(LIBDIR)/../Source/MODEL" shared


MATRIX_SRCS := $(wildcard */matrix.f)
MATRIX_OBJS := $(patsubst %.f,%.o,$(wildcard */matrix.f))

%/matrix.o: %/matrix.f
	$(MAKE) -C $* matrix.o

liball$(PROCNAME)_$(MENUM)me.$(dylibext): $(LIBDIR)/libdhelas.$(dylibext) $(LIBDIR)/libmodel.$(dylibext) all_matrix.o $(MATRIX_OBJS)
	 $(FC) $(DYNLIBFLAG) $(RPATHFLAG)liball$(PROCNAME)_$(MENUM)me.$(dylibext) -o liball$(PROCNAME)_$(MENUM)me.$(dylibext) all_matrix.o */matrix.o ../Source/DHELAS/*.o ../Source/MODEL/*.o


all_matrix$(MENUM)py.so:  liball$(PROCNAME)_$(MENUM)me.$(dylibext) f2py_wrapper.f makefile
	LDFLAGS="-Wl,-rpath,$(HERE)" $(F2PY)  -c f2py_wrapper.f -L$(HERE) -lall$(PROCNAME)_$(MENUM)me $(LINKLIBS) -m                 all_matrix$(MENUM)py
	touch all_matrix$(MENUM)py.so
