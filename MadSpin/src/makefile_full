include ../../Source/make_opts

LIBDIR = ../../lib/
PROG   = check
INIT   = init
LINKLIBS =  -L../../lib/ -ldhelas -lmodel 
LIBS = $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
PROCESS=  driver.o matrix.o matrix_prod.o ranmar.o

 
$(PROG): $(PROCESS)  makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(PROG) $(PROCESS) $(LINKLIBS)

$(INIT): initialize.o makefile $(LIBS) 
	$(FC) $(FFLAGS) -o $(INIT) initialize.o $(LINKLIBS)

driver.f: nexternal.inc pmass.inc ngraphs.inc coupl.inc

clean:
	$(RM) *.o
