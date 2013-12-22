include ../../Source/make_opts

LIBDIR = ../../lib/
PROG   = check
PROG_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS =  -L../../lib/ -ldhelas -lmodel
LIBS = $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS=  matrix.o
CHECK_SA=  check_sa.o
CHECK_SA_SPLITORDERS=  check_sa_born_splitOrders.o
 
$(PROG): $(PROCESS) $(CHECK_SA) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(CHECK_SA) $(LINKLIBS)

$(PROG_SPLITORDERS): $(PROCESS) $(CHECK_SA_SPLITORDERS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(CHECK_SA_SPLITORDERS) $(LINKLIBS)

driver.f: nexternal.inc pmass.inc ngraphs.inc coupl.inc