g77   = g77
FFLAGS= -O -ffixed-line-length-132
DEC   =  decay_couplings.o decay.o decay_matrix.o decay_mom.o decay_event.o vegas.o decay_printout.o hdecay.o ran1.o rw_events.o open_file.o alfas_functions.o
.f.o: ;$(g77) $(FFLAGS) -c $*.f
HELAS = ../HELAS/lib/libdhelas3.a

all: $(HELAS) $(DEC) 
	$(g77) $(FFLAGS) -o decay  $(DEC) $(HELAS)

$(HELAS):
	cd ../HELAS;make;ranlib lib/libdhelas3.a