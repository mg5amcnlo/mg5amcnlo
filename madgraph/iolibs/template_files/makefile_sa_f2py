include ../Source/make_opts
SHELL = /bin/bash
LIBDIR = ../lib/
PROG   = check
LINKLIBS =  -L../lib/ -ldhelas -lmodel
LIBS = $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS=  allmatrix.o
LIBRARY = libmatrix.a

# For python linking (require f2py part of numpy)
ifeq ($(origin MENUM),undefined)
  MENUM=2
endif


allmatrix$(MENUM)py.so: $(LIBDIR)/$(LIBRARY)  all_matrix.o
	$(F2PY) --f77exec=$(FC) $(LINKLIBS) -lmatrix  -c all_matrix.f */matrix.f -m allmatrix$(MENUM)py



$(LIBDIR)/$(LIBRARY): $(patsubst %.f,%.o,$(wildcard */matrix.f))  all_matrix.o
	$(call CREATELIB, $@, $^)  

