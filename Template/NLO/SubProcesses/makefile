include ../Source/make_opts

# Files for the read40 combiner of top drawer files
READ40=read40.o

# Files for the split40 splitter of top drawer files
SPLIT40=split40.o

# Files for the combining of splitted event files
COMBINE_SPLIT_EVENTS=combine_jobs.o handling_lhe_events.o fill_MC_mshell.o

# Files to collect all the events in the separate integration channels into a single file
COLLECT_EVENTS=collect_events.o handling_lhe_events.o fill_MC_mshell.o

combine_jobs.o : combine_jobs.f max_split.inc
	$(FC) $(FFLAGS) -c combine_jobs.f

%.o : %.f
	$(FC) $(FFLAGS) -c $<
%.o : %.for
	$(FC) $(FFLAGS) -c $<

read40: $(READ40)
	$(FC) $(FFLAGS) -o read40 $(READ40)

split40: $(SPLIT40)
	$(FC) $(FFLAGS) -o split40 $(SPLIT40)

combine_jobs: $(COMBINE_SPLIT_EVENTS) max_split.inc
	$(FC) $(FFLAGS) -o combine_jobs $(COMBINE_SPLIT_EVENTS)

collect_events: $(COLLECT_EVENTS)
	$(FC) $(FFLAGS) -o collect_events $(COLLECT_EVENTS)

