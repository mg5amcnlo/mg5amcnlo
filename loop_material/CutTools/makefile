#
# Please choose the PLATFORM where you are running the code
#
PLATFORM= LINUX
#PLATFORM= DOS
#
#
CTS_VERSION = v1.6.5
SRC = ./src
EXE = 
FC  = gfortran
#FC  = lf95
FFLAGS = -fno-automatic -O2 -funroll-all-loops  
#FFLAGS = -O0 -g -ffixed-line-length-none -Wall   -DU77EXT=0
#FFLAGS = -fno-automatic -g -O0 -fbounds-check 
#FFLAGS = -O0 -ffixed-line-length-none
#FFLAGS =  --nfix --o0 -g
#FFLAGS = --qaud
BLD = includects

VERSION = $(BLD)/version.h

ARGS = \
  EXE="$(EXE)" \
  FC="$(FC)" \
  FFLAGS="$(FFLAGS)" \

clean$(BLD):   default
ifeq ($(PLATFORM),LINUX)
	rm -fr  $(BLD)/*.f
	rm -fr  $(BLD)/*.f90
	rm -fr  $(BLD)/*.o
	rm -fr  $(BLD)/makefile
else
	del /Q .\$(BLD)\*.f
	del /Q .\$(BLD)\*.f90
	del /Q .\$(BLD)\*.o
	del /Q .\$(BLD)\makefile
endif

default lib: force
	cd $(BLD) && $(MAKE) $(ARGS) $@

force: $(BLD)/version.h

$(BLD)/version.h:
ifeq ($(PLATFORM),LINUX)
	-mkdir -p $(BLD)
	cp  -p ./src/avh/* $(BLD)/ 
	cp  -p ./src/cts/* $(BLD)/ 
	cp  -p ./src/mpfun90/* $(BLD)/ 
	cp  -p ./src/qcdloop/* $(BLD)/ 
	cp  -p ./src/makefile $(BLD)/ 
else
	-mkdir $(BLD)
	copy .\src\qcdloop\* .\$(BLD)
	copy .\src\mpfun90\* .\$(BLD)
	copy .\src\avh\* .\$(BLD)
	copy .\src\cts\* .\$(BLD)
	copy .\src\makefile .\$(BLD)
endif

CTS_DIR = Cuttools_$(CTS_VERSION)
CTS_TAR = $(CTS_DIR).tar.gz

clean:
ifeq ($(PLATFORM),LINUX)
	rm -fr $(BLD) $(CTS_TAR)
else
	del /S /Q $(CTS_TAR)
	rmdir /S /Q $(BLD) $(CTS_TAR)
endif


