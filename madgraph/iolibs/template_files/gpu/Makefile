CUARCHNUM=70
LIBDIR=../../lib
INCDIR=../../src
MODELLIB=model_sm
CXXFLAGS= -O3 -I$(INCDIR) -I.
CUARCHFLAGS= -arch=compute_$(CUARCHNUM) #--gpu-architecture=compute_35 --gpu-code=sm_35
CUFLAGS= $(CUARCHFLAGS) -use_fast_math -lineinfo
LIBFLAGS= -L$(LIBDIR) -l$(MODELLIB)
NVCC=nvcc
CXX=g++
MAKEDEBUG=

main=check.exe
cu_objects=CPPProcess.o check_sa.o
cxx_objects=

# Assuming uname is available, detect if architecture is power
UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),ppc64le)
    CUFLAGS+= -Xcompiler -mno-float128
endif

all: check

check: $(main)

debug: CXXFLAGS:=$(filter-out -O3,$(CXXFLAGS))
debug: CXXFLAGS += -g -O0 -DDEBUG2
debug: CUFLAGS:=$(filter-out -lineinfo,$(CUFLAGS))
debug: CUFLAGS += -G
debug: MAKEDEBUG := debug
debug: check

$(LIBDIR)/lib$(MODELLIB).a:
	@cd ../../src && make $(MAKEDEBUG)

check_sa.o: check_sa.cu
	$(NVCC) $(CPPFLAGS) $(CXXFLAGS) $(CUFLAGS) -c $< -o $@

%.o : %.cu %.h
	$(NVCC) $(CPPFLAGS) $(CXXFLAGS) $(CUFLAGS) -c $< -o $@

%.o : %.cc %.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(main): $(LIBDIR)/lib$(MODELLIB).a $(cxx_objects) $(cu_objects)
	$(NVCC)  -o $@ $(cxx_objects) $(cu_objects) $(LIBFLAGS) $(CUARCHFLAGS) -lcuda

.PHONY: clean

clean:
	cd ../../src && make clean
	rm -f $(main)
	rm -f $(cu_objects) $(cxx_objects)

memcheck: check
	/usr/local/cuda/bin/cuda-memcheck --check-api-memory-access yes --check-deprecated-instr yes --check-device-heap yes --demangle full --language c --leak-check full --racecheck-report all --report-api-errors all --show-backtrace yes --tool memcheck --track-unused-memory yes ./check.exe 5 5 10

perf: force
	make clean && make
	time ./check.exe -p 1024 16 384 && date

test: force
	./check.exe -v 1 1 10

force:

#Allowed values for this option: 'compute_30', 'compute_32', 'compute_35', 'compute_37', 'compute_50', 'compute_52', 'compute_53', 'compute_60', 'compute_61', 'compute_62', 'compute_70', 'compute_72', 'compute_75', 'sm_30', 'sm_32', 'sm_35', 'sm_37', 'sm_50', 'sm_52', 'sm_53', 'sm_60', 'sm_61', 'sm_62', 'sm_70', 'sm_72', 'sm_75'.

# Max compute architectures
# cern batch (tesla v100): 70
# jetson nano (maxwell): 35
