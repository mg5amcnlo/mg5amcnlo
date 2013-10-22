PRECISION= QP
#PRECISION= MP
CTS_VERSION = v1.9.0
SRC = ./src
EXE = 
FC  = gfortran
FFLAGS = -fno-automatic -O2 -funroll-all-loops -g  
#FFLAGS = -fno-automatic -Ofast -funroll-all-loops  
BLD = includects
CTS_DIR = cuttools_$(CTS_VERSION)
CTS_TAR = $(CTS_DIR).tar.gz

ARGS = \
  EXE="$(EXE)" \
  FC="$(FC)" \
  FFLAGS="$(FFLAGS)" \

ifeq ($(PRECISION),MP)
#
# For building of the version with internal multiprecision routines:
#
mp: cpmp clean$(BLD)
else
#
# For building of the version with quadruple precision compiler (if present):
#
qp: cpqp clean$(BLD)
endif

cpmp:   
	cp  -p ./src/cts/cts_mpr.in ./src/cts/cts_mpr.h
	cp  -p ./src/cts/cts_mpc.in ./src/cts/cts_mpc.h
	cp  -p ./src/cts/cts_mprec.in ./src/cts/cts_mprec.h 
	cp  -p ./src/cts/cts_mpinit.in ./src/cts/cts_mpinit.h
cpqp:   
	cp  -p ./src/cts/cts_qpr.in ./src/cts/cts_mpr.h
	cp  -p ./src/cts/cts_qpc.in ./src/cts/cts_mpc.h
	cp  -p ./src/cts/cts_qprec.in ./src/cts/cts_mprec.h
	cp  -p ./src/cts/cts_qpinit.in ./src/cts/cts_mpinit.h

clean$(BLD): default
	rm -fr  $(BLD)/*.f
	rm -fr  $(BLD)/*.f90
	rm -fr  $(BLD)/*.o
	rm -fr  $(BLD)/makefile

default: force
	cd $(BLD) && $(MAKE) $(ARGS) $@

force: $(BLD)/version.h

$(BLD)/version.h: 
	-mkdir -p $(BLD)
	cp  -p ./src/avh/* $(BLD)/ 
	cp  -p ./src/cts/* $(BLD)/ 
	cp  -p ./src/mpfun90/* $(BLD)/ 
	cp  -p ./src/qcdloop/* $(BLD)/ 
	cp  -p ./src/makefile $(BLD)/ 

tar:
	tar -czvf $(CTS_TAR) *

clean:
	rm -fr $(BLD) $(CTS_TAR)

