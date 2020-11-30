LIBDIR =../../lib
INCDIRS=-I. -I../../src -I../../../../../tools/
MODELLIB=model_%(model)s
CXXFLAGS= -O3 -std=c++11 $(INCDIRS) $(USE_NVTX) -Wall -Wshadow ${MGONGPU_CONFIG}
LIBFLAGS= -L$(LIBDIR) -l$(MODELLIB)
CXX=g++
NVCC?=$(shell which nvcc)
ifeq (, $(NVCC))
	NVCC=@echo No nvcc to run nvcc
else
	USE_NVTX?=-DUSE_NVTX
	CUARCHNUM=70
	#CUARCHNUM=61 #(For Pascal Architecture Cards)
	CUINC=-I/usr/local/cuda/targets/x86_64-linux/include
	CULIBFLAGS= -L/usr/local/cuda/targets/x86_64-linux/lib -lcurand
	CUARCHFLAGS= -arch=compute_$(CUARCHNUM)
	# CUARCHFLAGS= -gencode arch=compute_$(CUARCHNUM),code=sm_$(CUARCHNUM)
	CUFLAGS= -O3 $(INCDIRS) $(USE_NVTX) $(CUARCHFLAGS) -use_fast_math -lineinfo ${MGONGPU_CONFIG}
	# Without -maxrregcount: baseline throughput: 6.5E8 (16384 32 12) up to 7.3E8 (65536 128 12)
	###CUFLAGS+= --maxrregcount 160 # improves throughput: 6.9E8 (16384 32 12) up to 7.7E8 (65536 128 12)
	###CUFLAGS+= --maxrregcount 128 # improves throughput: 7.3E8 (16384 32 12) up to 7.6E8 (65536 128 12)
	###CUFLAGS+= --maxrregcount 96 # degrades throughput: 4.1E8 (16384 32 12) up to 4.5E8 (65536 128 12)
	###CUFLAGS+= --maxrregcount 64 # degrades throughput: 1.7E8 (16384 32 12) flat at 1.7E8 (65536 128 12)
endif
MAKEDEBUG=

cu_main=gcheck.exe
cxx_main=check.exe

cu_objects=gCPPProcess.o gcheck_sa.o
cxx_objects=CPPProcess.o

# Assuming uname is available, detect if architecture is power
UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),ppc64le)
    CUFLAGS+= -Xcompiler -mno-float128
endif

all:
	cd ../../src && make
	make $(cu_main)
	make $(cxx_main)

debug: CXXFLAGS:=$(filter-out -O3,$(CXXFLAGS))
debug: CXXFLAGS += -g -O0 -DDEBUG2
debug: CUFLAGS:=$(filter-out -lineinfo,$(CUFLAGS))
debug: CUFLAGS:=$(filter-out -O3,$(CUFLAGS))
debug: CUFLAGS += -g -G
debug: MAKEDEBUG := debug
debug: $(cu_main) $(cxx_main)

$(LIBDIR)/lib$(MODELLIB).a:
	@cd ../../src && make $(MAKEDEBUG)

gcheck_sa.o: gcheck_sa.cu *.h ../../src/*.h ../../src/*.cu
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c $< -o $@

%%.o : %%.cu *.h ../../src/*.h
	$(NVCC) $(CPPFLAGS) $(CUFLAGS) -c $< -o $@

%%.o : %%.cc *.h ../../src/*.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CUINC) -c $< -o $@

$(cu_main): $(LIBDIR)/lib$(MODELLIB).a $(cu_objects)
	$(NVCC) -o $@ $(cu_objects) $(LIBFLAGS) $(CUARCHFLAGS) -lcuda -lcurand

$(cxx_main): $(LIBDIR)/lib$(MODELLIB).a $(cxx_objects) check.o
	$(CXX) -o $@ $(cxx_objects) check_sa.o -ldl -pthread $(LIBFLAGS) $(CULIBFLAGS)

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

force:

#Allowed values for this option: 'compute_30', 'compute_32', 'compute_35', 'compute_37', 'compute_50', 'compute_52', 'compute_53', 'compute_60', 'compute_61', 'compute_62', 'compute_70', 'compute_72', 'compute_75', 'sm_30', 'sm_32', 'sm_35', 'sm_37', 'sm_50', 'sm_52', 'sm_53', 'sm_60', 'sm_61', 'sm_62', 'sm_70', 'sm_72', 'sm_75'.

# Max compute architectures
# cern batch (tesla v100): 70
# jetson nano (maxwell): 35
