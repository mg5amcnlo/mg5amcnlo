PRECISION= QP
#PRECISION= MP
CTS_VERSION = v1.9.3
SRC = ./src
CTS_DIR = cuttools_$(CTS_VERSION)
CTS_TAR = $(CTS_DIR).tar.gz

# Check for ../make_opts
ifeq ($(wildcard ../make_opts), ../make_opts)
  include ../make_opts
else
  FFLAGS = 
  FC=gfortran
endif

# These flags are vital for CT, so we enforce them here
FFLAGS += -fPIC -fno-automatic -O2 -funroll-all-loops
#FFLAGS += -fno-automatic -Ofast -funroll-all-loops  

# make sure no bound checks is used as it crashes avholo because of string message too long
tmp := $(FFLAGS)
FFLAGS = $(tmp:-fbounds-check=)

EXE = 
BLD = includects

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
	$(FC) -dumpversion > $(BLD)/compiler_version.log
# Below was an overkill
#	$(FC) --version | egrep -o "\d*\.\d*?(\.\d*)" | sed -n 1p > $(BLD)/compiler_version.log

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

