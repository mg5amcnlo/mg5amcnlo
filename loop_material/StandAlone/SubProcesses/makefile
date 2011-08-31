include ../../Source/make_opts

LIBDIR = ../../lib/
PROG   = check
LINKLIBS =  -L../../lib/ -ldhelas -lmodel -lcts 
LIBS = $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcts.$(libext)
PROCESS=  check_sa.o loop_matrix.o loop_num.o CT_interface.o
 
$(PROG): $(PROCESS)  makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(LINKLIBS)

driver.f: nexternal.inc pmass.inc ngraphs.inc coupl.inc
