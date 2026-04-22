################################################################################
#                     MAKEFILE DEVOTED TO MADANALYSIS JOB                      #
################################################################################

# Compilers
CXX = g++

# C++ Compilation options
CXXFLAGS  = -Wall -std=c++11 -O3 -fPIC -I$(MA5_BASE)/tools/
CXXFLAGS +=  -I./

# Linking options
LIBFLAGS  = 
LIBFLAGS += -L$(MA5_BASE)/tools/SampleAnalyzer/Lib -L$(MA5_BASE)/tools/SampleAnalyzer/ExternalSymLink/Lib
LIBFLAGS += -lprocess_for_ma5
LIBFLAGS += -lcommons_for_ma5

# Requirements to check before building
REQUIRED1 = $(MA5_BASE)/tools/SampleAnalyzer/Lib/libcommons_for_ma5.so
REQUIRED2 = $(MA5_BASE)/tools/SampleAnalyzer/Lib/libprocess_for_ma5.so

# Files
SRCS  = $(wildcard Main/*.cpp)
SRCS += $(wildcard SampleAnalyzer/User/*/*.cpp)
HDRS  = $(wildcard Main/*.h)
HDRS += $(wildcard SampleAnalyzer/User/*/*.h)
OBJS  = $(SRCS:.cpp=.o)

# Name of the executable
PROGRAM = MadAnalysis5job

# Defining colours
GREEN  = "\\033[1;32m"
RED    = "\\033[1;31m"
PINK   = "\\033[1;35m"
BLUE   = "\\033[1;34m"
YELLOW = "\\033[1;33m"
CYAN   = "\\033[1;36m"
NORMAL = "\\033[0;39m"

# All target
all: header library_check compile_header compile link_header link

# Check library
library_check:
ifeq ($(wildcard $(REQUIRED1)),)
	@echo -e $(RED)"The shared library "$(REQUIRED1)" is not found"
	@echo -e $(RED)" 1) Please check that MadAnalysis 5 is installed in the folder : "$(MA5_BASE)
	@echo -e $(RED)" 2) Launch MadAnalysis 5 in normal mode in order to build this library."
	@echo -e $(NORMAL)
	@false
endif
ifeq ($(wildcard $(REQUIRED2)),)
	@echo -e $(RED)"The shared library "$(REQUIRED2)" is not found"
	@echo -e $(RED)" 1) Please check that MadAnalysis 5 is installed in the folder : "$(MA5_BASE)
	@echo -e $(RED)" 2) Launch MadAnalysis 5 in normal mode in order to build this library."
	@echo -e $(NORMAL)
	@false
endif

# Header target
header:
	@echo -e $(YELLOW)"--------------------------------------------------------"
	@echo -e "                Building MadAnalysis Job                "
	@echo -e "--------------------------------------------------------"$(NORMAL)

# Compile_header target
compile_header:
	@echo -e $(YELLOW)"--------------------------------------------------------"
	@echo -e "                      Compilation                       "
	@echo -e "--------------------------------------------------------"$(NORMAL)

# Link_header target
link_header:
	@echo -e $(YELLOW)"--------------------------------------------------------"
	@echo -e "                        Linking                         "
	@echo -e "--------------------------------------------------------"$(NORMAL)

# clean_header target
clean_header:
	@echo -e $(YELLOW)"--------------------------------------------------------"
	@echo -e "       Removing intermediate files from building        "
	@echo -e "--------------------------------------------------------"$(NORMAL)

# mrproper_header target
mrproper_header:
	@echo -e $(YELLOW)"--------------------------------------------------------"
	@echo -e "                Cleaning all the project                "
	@echo -e "--------------------------------------------------------"$(NORMAL)

# Precompile target
precompile:

# Compile target
compile: precompile $(OBJS)

# Compile each file
%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# Link target
link: $(OBJS)
	$(CXX) $(OBJS) $(LIBFLAGS) -o ./$(PROGRAM)

# Phony target
.PHONY: do_clean header compile_header link_header

# Clean target
clean: clean_header do_clean

# Do clean target
do_clean: 
	@rm -f $(OBJS)

# Mr Proper target 
mrproper: mrproper_header do_mrproper

# Do Mr Proper target 
do_mrproper: do_clean
	@rm -f ./$(PROGRAM)
	@rm -f *~ */*~
	@rm -f Log/compilation.log Log/linking.log Log/cleanup.log Log/mrproper.log

