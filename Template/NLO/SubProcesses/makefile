include ../Source/make_opts

LIBDIR = ../lib/
LINKLIBS = -L$(LIBDIR) -lgeneric
LIBS = $(LIBDIR)libgeneric.a

# Files for the read40 combiner of top drawer files
READ40=read40.o

# Files to collect all the events in the separate integration channels into a single file
COLLECT_EVENTS=collect_events.o handling_lhe_events.o fill_MC_mshell.o

%.o : %.f
	$(FC) $(FFLAGS) -c -I. -I$(LIBDIR) $<
%.o : %.for
	$(FC) $(FFLAGS) -c $<

read40: $(READ40)
	$(FC) $(LDFLAGS) -o read40 $(READ40)

collect_events: $(COLLECT_EVENTS) $(LIBS)
	$(FC) -o collect_events $(COLLECT_EVENTS) $(LINKLIBS) $(LDFLAGS)
	rm handling_lhe_events.o

