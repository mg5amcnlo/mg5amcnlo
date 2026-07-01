include ../Source/make_opts

LIBDIR = ../lib/
LINKLIBS = -L$(LIBDIR) -lgeneric
LIBS = $(LIBDIR)libgeneric.a

ifneq ("$(wildcard ../../Source/RUNNING)","")
    LINKLIBS += -lrunning
    LIBS += $(LIBDIR)librunning.$(libext)
endif

# Files for the read40 combiner of top drawer files
READ40=read40.o

%.o : %.f
	$(FC) $(FFLAGS) -c -I. -I$(LIBDIR) $<
%.o : %.for
	$(FC) $(FFLAGS) -c $<

read40: $(READ40)
	$(FC) $(LDFLAGS) -o read40 $(READ40)

printalpha: printalpha.o 
	$(FC) -o printalpha printalpha.o  $(llhapdf) $(STDLIB)
