include ../../Source/make_opts

LIBDIR = ../../lib/
PROG   = check
LINKLIBS =  -L$(LIBDIR) -lcts -ldhelas -lmodel
LIBS =  $(LIBDIR)libcts.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS= check_sa.o loop_matrix.o born_matrix.o loop_num.o CT_interface.o

$(PROG):  $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(LINKLIBS)

clean:
		@rm -f *.o
