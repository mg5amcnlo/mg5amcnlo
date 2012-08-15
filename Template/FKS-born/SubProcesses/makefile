include ../Source/make_opts

# Files for the read40 combiner of top drawer files
READ40=read40.o

# Files for the split40 splitter of top drawer files
SPLIT40=split40.o

# Files for the combining of splitted event files
COMBINE_SPLIT_EVENTS=combine_jobs.o handling_lhe_events.o

%.o : %.f
	$(FC) $(FFLAGS) -c $<
%.o : %.for
	$(FC) $(FFLAGS) -c $<

read40: $(READ40)
	$(FC) $(FFLAGS) -o read40 $(READ40)

split40: $(SPLIT40)
	$(FC) $(FFLAGS) -o split40 $(SPLIT40)

combine_jobs: $(COMBINE_SPLIT_EVENTS)
	$(FC) $(FFLAGS) -o combine_jobs $(COMBINE_SPLIT_EVENTS)

