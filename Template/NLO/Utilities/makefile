P0=$(shell ls -dl ../SubProcesses/P0* | head -1)
P01=$(shell echo $(P0) | cut -d'/' -f3)

INCLUDE=-I../SubProcesses/$(P01)
FF=gfortran -g -fno-automatic -ffixed-line-length-132 -fbounds-check $(INCLUDE)
FILES=handling_lhe_events.o fill_MC_mshell.o

%.o : %.f
	$(FF) -c $<

all: check_events compare_events merge_events select_events split_events

check_events : $(FILES) check_events.o
	$(FF) $^ -o $@

compare_events : $(FILES) compare_events.o
	$(FF) $^ -o $@

merge_events : $(FILES) merge_events.o
	$(FF) $^ -o $@

select_events : $(FILES) select_events.o
	$(FF) $^ -o $@

split_events : $(FILES) split_events.o
	$(FF) $^ -o $@

clean:
	@rm -f *.o
	@rm -f check_events compare_events merge_events select_events split_events
	@rm -f LHEF.* res_wgt headfile