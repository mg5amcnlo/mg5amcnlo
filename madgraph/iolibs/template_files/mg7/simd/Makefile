include config.mk

SUBPROC_PATH=$(PROC_PATH)/SubProcesses/P$(SUBPROC_NUMBER)_$(SUBPROC_NAME)
COMMON_LIB_NAME=libmg5amc_common_cpp.so
SUBPROC_LIB_NAME=libmg5amc_$(SUBPROC_NAME)_cpp.so

include $(PROC_PATH)/src/cudacpp_config.mk
include cudacpp.mk

INCDIR_SP=$(SUBPROC_PATH)
INCDIR_SRC=$(PROC_PATH)/src
COMMON_LIB=$(PROC_PATH)/lib/$(COMMON_LIB_NAME)
SUBPROC_LIB=$(PROC_PATH)/lib/$(SUBPROC_LIB_NAME)
MAIN=api.so

all: $(MAIN)

$(COMMON_LIB_NAME): $(COMMON_LIB)
	cp $(COMMON_LIB) $(COMMON_LIB_NAME)

$(SUBPROC_LIB_NAME): $(SUBPROC_LIB)
	cp $(SUBPROC_LIB) $(SUBPROC_LIB_NAME)
	if [ "$(shell uname)" = "Darwin" ]; then\
		install_name_tool -id @rpath/$(SUBPROC_LIB_NAME) $(SUBPROC_LIB_NAME);\
	fi

$(MAIN): api_simd.cpp $(COMMON_LIB_NAME) $(SUBPROC_LIB_NAME)
	$(CXX) -shared $(CXXFLAGS) -I$(INCDIR_SP) -I$(INCDIR_SRC) -I. -Wl,-rpath $(shell pwd) -o $@ \
		api_simd.cpp $(COMMON_LIB_NAME) $(SUBPROC_LIB_NAME)

.PHONY: clean
clean:
	rm -f $(MAIN) $(COMMON_LIB_NAME) $(SUBPROC_LIB_NAME)
