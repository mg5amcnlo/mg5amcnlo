LIBDIR   = ../../lib
TOOLSDIR = ../../../../../tools/
INCFLAGS = -I. -I../../src -I$(TOOLSDIR)
MODELLIB = model_%(model)s
OPTFLAGS = -O3
CXXFLAGS = $(OPTFLAGS) -std=c++11 $(INCFLAGS) $(USE_NVTX) -Wall -Wshadow $(MGONGPU_CONFIG)
LIBFLAGS = -L$(LIBDIR) -l$(MODELLIB)
CXX     ?= g++

#Export CUDA_HOME to select a cuda installation
ifndef CUDA_HOME
  NVCC ?= $(shell which nvcc 2>/dev/null)
  ifneq ($(NVCC),)
    # NVCC is in the PATH or set explicitly
    CUDA_HOME  = $(patsubst %%bin/nvcc,%%,$(NVCC))
    CUDA_HOME := $(warning No CUDA_HOME exported. Using "$(CUDA_HOME)") $(CUDA_HOME)
  endif
endif

ifdef CUDA_HOME
  NVCC = $(CUDA_HOME)/bin/nvcc
  CUARCHNUM=70
  USE_NVTX?=-DUSE_NVTX
  CUARCHFLAGS= -arch=compute_$(CUARCHNUM)
  CUINC       = -I$(CUDA_HOME)/include/
  CULIBFLAGS  = -L$(CUDA_HOME)/lib64/ -lcuda -lcurand
  CUFLAGS= $(OPTFLAGS) -std=c++14 $(INCFLAGS) $(CUINC) $(USE_NVTX) $(CUARCHFLAGS) -use_fast_math -lineinfo $(MGONGPU_CONFIG)

  cu_main     = gcheck.exe
  cu_objects  = gCPPProcess.o
else
  # No cuda. Switch cuda compilation off and go to common random numbers in C++
  NVCC       := $(warning No cuda found. Export CUDA_HOME to compile with cuda)
  USE_NVTX   :=
  CULIBFLAGS :=
  ifndef MGONGPU_CONFIG
    export MGONGPU_CONFIG = -DMGONGPU_COMMONRAND_ONHOST
  endif
endif

GTESTLIBDIR = $(TOOLSDIR)/googletest/build/lib/
GTESTLIBS   = $(GTESTLIBDIR)/libgtest.a $(GTESTLIBDIR)/libgtest_main.a
MAKEDEBUG=

cxx_main=check.exe
cxx_objects=CPPProcess.o

# Assuming uname is available, detect if architecture is power
UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),ppc64le)
    CUFLAGS+= -Xcompiler -mno-float128
endif

all: ../../src $(cu_main) $(cxx_main) runTest.exe

debug: OPTFLAGS = -g -O0 -DDEBUG2
debug: CUFLAGS := $(filter-out -lineinfo,$(CUFLAGS))
debug: CUFLAGS += -G
debug: MAKEDEBUG := debug
debug: all

$(LIBDIR)/lib$(MODELLIB).a:
	$(MAKE) -C ../../src $(MAKEDEBUG)

gcheck_sa.o: gcheck_sa.cu *.h ../../src/*.h ../../src/*.cu
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c $< -o $@

%%.o : %%.cu *.h ../../src/*.h
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c $< -o $@

%%.o : %%.cc *.h ../../src/*.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CUINC) -c $< -o $@

$(cu_main): gcheck_sa.o $(LIBDIR)/lib$(MODELLIB).a $(cu_objects)
	$(NVCC) $< -o $@ $(cu_objects) $(LIBFLAGS) $(CUARCHFLAGS) $(CULIBFLAGS)

$(cxx_main): check_sa.o $(LIBDIR)/lib$(MODELLIB).a $(cxx_objects)
	$(CXX) $< -o $@ $(cxx_objects) $(CPPFLAGS) $(CXXFLAGS) -ldl -pthread $(LIBFLAGS) $(CULIBFLAGS)


runTest.o:   $(GTESTLIBS)
runTest.exe: $(GTESTLIBS)
runTest.exe: INCFLAGS += -I$(TOOLSDIR)/googletest/googletest/include/
runTest.exe: LIBFLAGS += -L$(GTESTLIBDIR)/ -lgtest -lgtest_main
ifeq ($(NVCC),)
runTest.exe: runTest.o $(LIBDIR)/lib$(MODELLIB).a $(cxx_objects) $(GTESTLIBS)
	$(CXX) -o $@ $(cxx_objects) runTest.o $(CPPFLAGS) $(CXXFLAGS) -ldl -pthread $(LIBFLAGS) $(CULIBFLAGS)
else
runTest.exe: runTest.o $(LIBDIR)/lib$(MODELLIB).a $(cxx_objects) $(GTESTLIBS)
	ln -sf runTest.cc runTest_tmp.cu
	$(NVCC) -o $@ $(cxx_objects) runTest.o $(cu_objects) runTest_tmp.cu $(CPPFLAGS) $(CUFLAGS) -ldl $(LIBFLAGS) $(CULIBFLAGS) -lcuda
	unlink runTest_tmp.cu
endif

$(GTESTLIBS):
	$(MAKE) -C $(TOOLSDIR)

check: runTest.exe
	./runTest.exe

.PHONY: clean

clean:
	cd ../../src && make clean
	rm -f *.o *.exe

memcheck: $(cu_main)
	/usr/local/cuda/bin/cuda-memcheck --check-api-memory-access yes --check-deprecated-instr yes --check-device-heap yes --demangle full --language c --leak-check full --racecheck-report all --report-api-errors all --show-backtrace yes --tool memcheck --track-unused-memory yes ./gcheck.exe 2 32 2

perf: force
	make clean && make
	time ./gcheck.exe -p 16348 32 12 && date

test: force
	./gcheck.exe -v 1 32 1

info:
ifdef CUDA_HOME
	$(NVCC) --version
	echo ""
endif
	$(CXX) --version

force:

#Allowed values for this option: 'compute_30', 'compute_32', 'compute_35', 'compute_37', 'compute_50', 'compute_52', 'compute_53', 'compute_60', 'compute_61', 'compute_62', 'compute_70', 'compute_72', 'compute_75', 'sm_30', 'sm_32', 'sm_35', 'sm_37', 'sm_50', 'sm_52', 'sm_53', 'sm_60', 'sm_61', 'sm_62', 'sm_70', 'sm_72', 'sm_75'.

# Max compute architectures
# cern batch (tesla v100): 70
# jetson nano (maxwell): 35
