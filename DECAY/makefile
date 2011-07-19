# Check for ../make_opts
ifeq ($(wildcard ../Template/Source/make_opts), ../Template/Source/make_opts)
  include ../Template/Source/make_opts
else
  include ../madevent/Source/make_opts
endif

DEC   =  decay_couplings.o decay.o decay_matrix.o decay_mom.o decay_event.o vegas.o decay_printout.o hdecay.o ran1.o rw_events.o open_file.o alfas_functions.o
HELAS = ../HELAS/lib/libdhelas3.$(libext)
BIN = decay
LIB = -L../HELAS/lib -ldhelas3

all: $(BIN)

$(BIN): $(DEC) $(HELAS)
	$(FC) $(FFLAGS) -o $@ $(DEC) $(LIB)

$(HELAS):
	cd ../HELAS;make

clean:
	$(RM) *.o $(BIN)
