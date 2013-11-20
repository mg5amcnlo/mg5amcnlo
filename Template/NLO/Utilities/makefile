P0=$(shell ls -dl ../SubProcesses/P0* | head -1)
P01=$(shell echo $(P0) | cut -d'/' -f3)

INCLUDE=-I../SubProcesses/$(P01)
FF=gfortran -g -fno-automatic -ffixed-line-length-132 -fbounds-check $(INCLUDE)
FILES=handling_lhe_events.o fill_MC_mshell.o
FILES2=madfks_plot.f madfks_dbook.f setcuts.o

%.o : %.f
	$(FF) -c $<

all: check_events combine_plots compare_events merge_events plot_events \
	plot_events_lhef select_events split_events sum_plots

check_events : $(FILES) check_events.o
	$(FF) $^ -o $@

combine_plots : combine_plots.o
	$(FF) $^ -o $@

compare_events : $(FILES) compare_events.o
	$(FF) $^ -o $@

merge_events : $(FILES) merge_events.o
	$(FF) $^ -o $@

plot_events : $(FILES) $(FILES2) plot_events.o
	$(FF) $^ -o $@

plot_events_lhef : $(FILES) $(FILES2) plot_events_lhef.o
	$(FF) $^ -o $@

select_events : $(FILES) select_events.o
	$(FF) $^ -o $@

split_events : $(FILES) split_events.o
	$(FF) $^ -o $@

sum_plots : sum_plots.o
	$(FF) $^ -o $@

clean:
	@rm -f *.o
	@rm -f check_events combine_plots compare_events merge_events plot_events \
	plot_events_lhef select_events split_events sum_plots
	@rm -f LHEF.* res_wgt headfile