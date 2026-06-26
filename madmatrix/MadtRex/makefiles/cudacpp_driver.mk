# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: S. Roiser (Feb 2020) for the MG5aMC CUDACPP plugin.
# Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi, Z. Wettersten (2020-2024).

#=== Determine the name of this makefile (https://ftp.gnu.org/old-gnu/Manuals/make-3.80/html_node/make_17.html)
#=== NB: use ':=' to ensure that the value of CUDACPP_MAKEFILE is not modified further down after including make_opts
#=== NB: use 'override' to ensure that the value can not be modified from the outside
override CUDACPP_MAKEFILE := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
###$(info CUDACPP_MAKEFILE='$(CUDACPP_MAKEFILE)')

#=== NB: different names (e.g. cudacpp.mk and cudacpp_src.mk) are used in the Subprocess and src directories
override CUDACPP_SRC_MAKEFILE = cudacpp_src.mk

#-------------------------------------------------------------------------------

#=== Include cudacpp_config.mk

# Check that the user-defined choices of BACKEND, FPTYPE, HELINL, HRDCOD are supported (and configure defaults if no user-defined choices exist)
# Stop with an error if BACKEND=cuda and nvcc is missing or if BACKEND=hip and hipcc is missing.
# Determine CUDACPP_BUILDDIR from a DIRTAG based on BACKEND, FPTYPE, HELINL, HRDCOD and from the user-defined choice of USEBUILDDIR
include ../src/cudacpp_config.mk

# Export CUDACPP_BUILDDIR (so that there is no need to check/define it again in cudacpp_src.mk)
export CUDACPP_BUILDDIR

#-------------------------------------------------------------------------------

#=== Use bash in the Makefile (https://www.gnu.org/software/make/manual/html_node/Choosing-the-Shell.html)

SHELL := /bin/bash

#-------------------------------------------------------------------------------

#=== Detect O/S and architecture (assuming uname is available, https://en.wikipedia.org/wiki/Uname)

# Detect O/S kernel (Linux, Darwin...)
UNAME_S := $(shell uname -s)
###$(info UNAME_S='$(UNAME_S)')

# Detect architecture (x86_64, ppc64le...)
UNAME_P := $(shell uname -p)
###$(info UNAME_P='$(UNAME_P)')
UNAME_M := $(shell uname -m)

#-------------------------------------------------------------------------------

#=== Include the common MG5aMC Makefile options

# OM: including make_opts is crucial for MG5aMC flag consistency/documentation
# AV: disable the inclusion of make_opts if the file has not been generated (standalone cudacpp)
ifneq ($(wildcard ../Source/make_opts),)
  include ../Source/make_opts
endif

#-------------------------------------------------------------------------------

#=== Redefine BACKEND if the current value is 'cppauto'

# Set the default BACKEND choice corresponding to 'cppauto' (the 'best' C++ vectorization available)
ifeq ($(BACKEND),cppauto)
  ifeq ($(UNAME_P),ppc64le)
    override BACKEND = cppsse4
  else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
    override BACKEND = cppsse4
  else ifeq ($(wildcard /proc/cpuinfo),)
    override BACKEND = cppnone
    ###$(warning Using BACKEND='$(BACKEND)' because host SIMD features cannot be read from /proc/cpuinfo)
  else ifeq ($(shell grep -m1 -c avx512vl /proc/cpuinfo)$(shell $(CXX) --version | grep ^clang),1)
    override BACKEND = cpp512y
  else ifeq ($(shell grep -m1 -c avx2 /proc/cpuinfo),1)
    override BACKEND = cppavx2
    ###ifneq ($(shell grep -m1 -c avx512vl /proc/cpuinfo),1)
    ###  $(warning Using BACKEND='$(BACKEND)' because host does not support avx512vl)
    ###else
    ###  $(warning Using BACKEND='$(BACKEND)' because this is faster than avx512vl for clang)
    ###endif
  else ifeq ($(shell grep -m1 -c sse4_2 /proc/cpuinfo),1)
    override BACKEND = cppsse4
  else
    override BACKEND = cppnone
  endif
  $(info BACKEND=$(BACKEND) (was cppauto))
else
  $(info BACKEND='$(BACKEND)')
endif

#-------------------------------------------------------------------------------

#=== Configure the C++ compiler

CXXFLAGS = $(OPTFLAGS) -std=c++17 -Wall -Wshadow -Wextra
ifeq ($(shell $(CXX) --version | grep ^nvc++),)
  CXXFLAGS += -ffast-math # see issue #117
endif
###CXXFLAGS+= -Ofast # performance is not different from --fast-math
###CXXFLAGS+= -g # FOR DEBUGGING ONLY

# Optionally add debug flags to display the full list of flags (eg on Darwin)
###CXXFLAGS+= -v

# Note: AR, CXX and FC are implicitly defined if not set externally
# See https://www.gnu.org/software/make/manual/html_node/Implicit-Variables.html

# Add -mmacosx-version-min=11.3 to avoid "ld: warning: object file was built for newer macOS version than being linked"
ifneq ($(shell $(CXX) --version | egrep '^Apple clang'),)
  CXXFLAGS += -mmacosx-version-min=11.3
endif

# Export CXXFLAGS (so that there is no need to check/define it again in cudacpp_src.mk)
export CXXFLAGS

#-------------------------------------------------------------------------------

#=== Configure the GPU compiler (CUDA or HIP)
#=== (note, this is done also for C++, as NVTX and CURAND/ROCRAND are also needed by the C++ backends)

# Set CUDA_HOME from the path to nvcc, if it exists
#override CUDA_HOME = $(patsubst %%/bin/nvcc,%%,$(shell which nvcc 2>/dev/null))
CUDA_HOME := $(patsubst %/bin/nvcc,%,$(shell which nvcc 2>/dev/null))

# Set HIP_HOME from the path to hipcc, if it exists
override HIP_HOME = $(patsubst %%/bin/hipcc,%%,$(shell which hipcc 2>/dev/null))

# Configure CUDA_INC (for CURAND and NVTX) and NVTX if a CUDA installation exists
# (FIXME? Is there any equivalent of NVTX FOR HIP? What should be configured if both CUDA and HIP are installed?)
ifneq ($(CUDA_HOME),)
  USE_NVTX ?=-DUSE_NVTX
  CUDA_INC = -I$(CUDA_HOME)/include/
else
  override USE_NVTX=
  override CUDA_INC=
endif

# NB: NEW LOGIC FOR ENABLING AND DISABLING CUDA OR HIP BUILDS (AV Feb-Mar 2024)
# - In the old implementation, by default the C++ targets for one specific AVX were always built together with either CUDA or HIP.
# If both CUDA and HIP were installed, then CUDA took precedence over HIP, and the only way to force HIP builds was to disable
# CUDA builds by setting CUDA_HOME to an invalid value (as CUDA_HOME took precdence over PATH to find the installation of nvcc).
# Similarly, C++-only builds could be forced by setting CUDA_HOME and/or HIP_HOME to invalid values. A check for an invalid nvcc
# in CUDA_HOME or an invalid hipcc HIP_HOME was necessary to ensure this logic, and had to be performed at the very beginning.
# - In the new implementation (PR #798), separate individual builds are performed for one specific C++/AVX mode, for CUDA or
# for HIP. The choice of the type of build is taken depending on the value of the BACKEND variable (replacing the AVX variable).
# Unlike what happened in the past, nvcc and hipcc must have already been added to PATH. Using 'which nvcc' and 'which hipcc',
# their existence and their location is checked, and the variables CUDA_HOME and HIP_HOME are internally set by this makefile.
# This must be still done before backend-specific customizations, e.g. because CURAND and NVTX are also used in C++ builds.
# Note also that a preliminary check for nvcc and hipcc if BACKEND is cuda or hip is performed in cudacpp_config.mk.
# - Note also that the REQUIRE_CUDA variable (which was used in the past, e.g. for CI tests on GPU #443) is now (PR #798) no
# longer necessary, as it is now equivalent to BACKEND=cuda. Similarly, there is no need to introduce a REQUIRE_HIP variable.

#=== Configure the CUDA or HIP compiler (only for the CUDA and HIP backends)
#=== (NB: throughout all makefiles, an empty GPUCC is used to indicate that this is a C++ build, i.e. that BACKEND is neither cuda nor hip!)

ifeq ($(BACKEND),cuda)

  # If CXX is not a single word (example "clang++ --gcc-toolchain...") then disable CUDA builds (issue #505)
  # This is because it is impossible to pass this to "GPUFLAGS += -ccbin <host-compiler>" below
  ifneq ($(words $(subst ccache ,,$(CXX))),1) # allow at most "CXX=ccache <host-compiler>" from outside
    $(error BACKEND=$(BACKEND) but CUDA builds are not supported for multi-word CXX "$(CXX)")
  endif

  # Set GPUCC as $(CUDA_HOME)/bin/nvcc (it was already checked above that this exists)
  GPUCC = $(CUDA_HOME)/bin/nvcc
  XCOMPILERFLAG = -Xcompiler
  GPULANGUAGE = cu
  GPUSUFFIX = cuda

  # Basic compiler flags (optimization and includes)
  GPUFLAGS = $(foreach opt, $(OPTFLAGS), $(XCOMPILERFLAG) $(opt))

  # NVidia CUDA architecture flags
  # See https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
  # See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # Default: use compute capability 70 for V100 (CERN lxbatch, CERN itscrd, Juwels Cluster).
  # This will embed device code for 70, and PTX for 70+.
  # One may pass MADGRAPH_CUDA_ARCHITECTURE (comma-separated list) to the make command to use another value or list of values (see #533).
  # Examples: use 60 for P100 (Piz Daint), 80 for A100 (Juwels Booster, NVidia raplab/Curiosity).
  MADGRAPH_CUDA_ARCHITECTURE ?= 70
  ###GPUARCHFLAGS = -gencode arch=compute_$(MADGRAPH_CUDA_ARCHITECTURE),code=compute_$(MADGRAPH_CUDA_ARCHITECTURE) -gencode arch=compute_$(MADGRAPH_CUDA_ARCHITECTURE),code=sm_$(MADGRAPH_CUDA_ARCHITECTURE) # Older implementation (AV): go back to this one for multi-GPU support #533
  ###GPUARCHFLAGS = --gpu-architecture=compute_$(MADGRAPH_CUDA_ARCHITECTURE) --gpu-code=sm_$(MADGRAPH_CUDA_ARCHITECTURE),compute_$(MADGRAPH_CUDA_ARCHITECTURE)  # Newer implementation (SH): cannot use this as-is for multi-GPU support #533
  comma:=,
  GPUARCHFLAGS = $(foreach arch,$(subst $(comma), ,$(MADGRAPH_CUDA_ARCHITECTURE)),-gencode arch=compute_$(arch),code=compute_$(arch) -gencode arch=compute_$(arch),code=sm_$(arch))
  GPUFLAGS += $(GPUARCHFLAGS)

  # Other NVidia-specific flags
  CUDA_OPTFLAGS = -lineinfo
  GPUFLAGS += $(CUDA_OPTFLAGS)

  # NVCC version
  ###GPUCC_VERSION = $(shell $(GPUCC) --version | grep 'Cuda compilation tools' | cut -d' ' -f5 | cut -d, -f1)

  # Fast math
  GPUFLAGS += -use_fast_math

  # Extra build warnings
  ###GPUFLAGS += $(XCOMPILERFLAG) -Wall $(XCOMPILERFLAG) -Wextra $(XCOMPILERFLAG) -Wshadow

  # CUDA includes and NVTX
  GPUFLAGS += $(CUDA_INC) $(USE_NVTX) 

  # C++ standard
  GPUFLAGS += -std=c++17 # need CUDA >= 11.2 (see #333): this is enforced in mgOnGpuConfig.h

  # For nvcc, use -maxrregcount to control the maximum number of registries (this does not exist in hipcc)
  # Without -maxrregcount: baseline throughput: 6.5E8 (16384 32 12) up to 7.3E8 (65536 128 12)
  ###GPUFLAGS+= --maxrregcount 160 # improves throughput: 6.9E8 (16384 32 12) up to 7.7E8 (65536 128 12)
  ###GPUFLAGS+= --maxrregcount 128 # improves throughput: 7.3E8 (16384 32 12) up to 7.6E8 (65536 128 12)
  ###GPUFLAGS+= --maxrregcount 96 # degrades throughput: 4.1E8 (16384 32 12) up to 4.5E8 (65536 128 12)
  ###GPUFLAGS+= --maxrregcount 64 # degrades throughput: 1.7E8 (16384 32 12) flat at 1.7E8 (65536 128 12)

  # Set the host C++ compiler for nvcc via "-ccbin <host-compiler>"
  # (NB issue #505: this must be a single word, "clang++ --gcc-toolchain..." is not supported)
  GPUFLAGS += -ccbin $(shell which $(subst ccache ,,$(CXX)))

  # Allow newer (unsupported) C++ compilers with older versions of CUDA if ALLOW_UNSUPPORTED_COMPILER_IN_CUDA is set (#504)
  ifneq ($(origin ALLOW_UNSUPPORTED_COMPILER_IN_CUDA),undefined)
    GPUFLAGS += -allow-unsupported-compiler
  endif

else ifeq ($(BACKEND),hip)

  # Set GPUCC as $(HIP_HOME)/bin/hipcc (it was already checked above that this exists)
  GPUCC = $(HIP_HOME)/bin/hipcc
  XCOMPILERFLAG =
  GPULANGUAGE = hip
  GPUSUFFIX = hip

  # Basic compiler flags (optimization and includes)
  GPUFLAGS = $(foreach opt, $(OPTFLAGS), $(XCOMPILERFLAG) $(opt))

  # AMD HIP architecture flags
  GPUARCHFLAGS = --offload-arch=gfx90a
  GPUFLAGS += $(GPUARCHFLAGS)

  # Other AMD-specific flags
  GPUFLAGS += -target x86_64-linux-gnu -DHIP_PLATFORM=amd

  # Fast math (is -DHIP_FAST_MATH equivalent to -ffast-math?)
  GPUFLAGS += -DHIP_FAST_MATH

  # Extra build warnings
  ###GPUFLAGS += $(XCOMPILERFLAG) -Wall $(XCOMPILERFLAG) -Wextra $(XCOMPILERFLAG) -Wshadow

  # HIP includes
  HIP_INC = -I$(HIP_HOME)/include/
  GPUFLAGS += $(HIP_INC)

  # C++ standard
  GPUFLAGS += -std=c++17

else

  # Backend is neither cuda nor hip
  override GPUCC=
  override GPUFLAGS=

  # Sanity check, this should never happen: if GPUCC is empty, then this is a C++ build, i.e. BACKEND is neither cuda nor hip.
  # In practice, in the following, "ifeq ($(GPUCC),)" is equivalent to "ifneq ($(findstring cpp,$(BACKEND)),)".
  # Conversely, note that GPUFLAGS is non-empty also for C++ builds, but it is never used in that case.
  ifeq ($(findstring cpp,$(BACKEND)),)
    $(error INTERNAL ERROR! Unknown backend BACKEND='$(BACKEND)': supported backends are $(foreach backend,$(SUPPORTED_BACKENDS),'$(backend)'))
  endif

endif

# Export GPUCC, GPUFLAGS, GPULANGUAGE, GPUSUFFIX (these are needed by both src and rwgt_runners, but should not be overwritten there)
export CUDA_HOME
export GPUCC
export GPUFLAGS
export GPULANGUAGE
export GPUSUFFIX
export XCOMPILERFLAG

#-------------------------------------------------------------------------------

#=== Configure ccache for C++ and CUDA/HIP builds

# Enable ccache if USECCACHE=1
ifeq ($(USECCACHE)$(shell echo $(CXX) | grep ccache),1)
  override CXX:=ccache $(CXX)
endif
#ifeq ($(USECCACHE)$(shell echo $(AR) | grep ccache),1)
#  override AR:=ccache $(AR)
#endif
ifneq ($(GPUCC),)
  ifeq ($(USECCACHE)$(shell echo $(GPUCC) | grep ccache),1)
    override GPUCC:=ccache $(GPUCC)
  endif
endif

#-------------------------------------------------------------------------------

#=== Configure common compiler flags for C++ and CUDA/HIP

INCFLAGS = -I.
OPTFLAGS = -O3 # this ends up in GPUFLAGS too (should it?), cannot add -Ofast or -ffast-math here

# Dependency on src directory
ifeq ($(GPUCC),)
MG5AMC_COMMONLIB = mg5amc_common_cpp
else
MG5AMC_COMMONLIB = mg5amc_common_$(GPUSUFFIX)
endif
LIBFLAGS = -L$(LIBDIR) -l$(MG5AMC_COMMONLIB) -lrex -ltearex
INCFLAGS += -I../src

# Compiler-specific googletest build directory (#125 and #738)
ifneq ($(shell $(CXX) --version | grep '^Intel(R) oneAPI DPC++/C++ Compiler'),)
  override CXXNAME = icpx$(shell $(CXX) --version | head -1 | cut -d' ' -f5)
else ifneq ($(shell $(CXX) --version | egrep '^clang'),)
  override CXXNAME = clang$(shell $(CXX) --version | head -1 | cut -d' ' -f3)
else ifneq ($(shell $(CXX) --version | grep '^g++ (GCC)'),)
  override CXXNAME = gcc$(shell $(CXX) --version | head -1 | cut -d' ' -f3)
else
  override CXXNAME = unknown
endif
###$(info CXXNAME=$(CXXNAME))
override CXXNAMESUFFIX = _$(CXXNAME)

# Export CXXNAMESUFFIX (so that there is no need to check/define it again in cudacpp_test.mk)
export CXXNAMESUFFIX

#-------------------------------------------------------------------------------

#=== Configure PowerPC-specific compiler flags for C++ and CUDA/HIP

# PowerPC-specific CXX compiler flags (being reviewed)
ifeq ($(UNAME_P),ppc64le)
  CXXFLAGS+= -mcpu=power9 -mtune=power9 # gains ~2-3%% both for cppnone and cppsse4
  # Throughput references without the extra flags below: cppnone=1.41-1.42E6, cppsse4=2.15-2.19E6
  ###CXXFLAGS+= -DNO_WARN_X86_INTRINSICS # no change
  ###CXXFLAGS+= -fpeel-loops # no change
  ###CXXFLAGS+= -funroll-loops # gains ~1%% for cppnone, loses ~1%% for cppsse4
  ###CXXFLAGS+= -ftree-vectorize # no change
  ###CXXFLAGS+= -flto # would increase to cppnone=4.08-4.12E6, cppsse4=4.99-5.03E6!
else
  ###CXXFLAGS+= -flto # also on Intel this would increase throughputs by a factor 2 to 4...
  ######CXXFLAGS+= -fno-semantic-interposition # no benefit (neither alone, nor combined with -flto)
endif

# PowerPC-specific CUDA/HIP compiler flags (to be reviewed!)
ifeq ($(UNAME_P),ppc64le)
  GPUFLAGS+= $(XCOMPILERFLAG) -mno-float128
endif

#-------------------------------------------------------------------------------

#=== Configure defaults for OMPFLAGS

# Disable OpenMP by default: enable OpenMP only if USEOPENMP=1 (#758)
ifeq ($(USEOPENMP),1)
  ###$(info USEOPENMP==1: will build with OpenMP if possible)
  ifneq ($(findstring hipcc,$(GPUCC)),)
    override OMPFLAGS = # disable OpenMP MT when using hipcc #802
  else ifneq ($(shell $(CXX) --version | egrep '^Intel'),)
    override OMPFLAGS = -fopenmp
    ###override OMPFLAGS = # disable OpenMP MT on Intel (was ok without GPUCC but not ok with GPUCC before #578)
  else ifneq ($(shell $(CXX) --version | egrep '^clang version 16'),)
    ###override OMPFLAGS = # disable OpenMP on clang16 #904
    $(error OpenMP is not supported by cudacpp on clang16 - issue #904)
  else ifneq ($(shell $(CXX) --version | egrep '^clang version 17'),)
    ###override OMPFLAGS = # disable OpenMP on clang17 #904
    $(error OpenMP is not supported by cudacpp on clang17 - issue #904)
  else ifneq ($(shell $(CXX) --version | egrep '^(clang)'),)
    override OMPFLAGS = -fopenmp
    ###override OMPFLAGS = # disable OpenMP MT on clang (was not ok without or with nvcc before #578)
  ###else ifneq ($(shell $(CXX) --version | egrep '^(Apple clang)'),) # AV for Mac (Apple clang compiler)
  else ifeq ($(UNAME_S),Darwin) # OM for Mac (any compiler)
    override OMPFLAGS = # AV disable OpenMP MT on Apple clang (builds fail in the CI #578)
    ###override OMPFLAGS = -fopenmp # OM reenable OpenMP MT on Apple clang? (AV Oct 2023: this still fails in the CI)
  else
    override OMPFLAGS = -fopenmp # enable OpenMP MT by default on all other platforms
    ###override OMPFLAGS = # disable OpenMP MT on all other platforms (default before #575)
  endif
else
  ###$(info USEOPENMP!=1: will build without OpenMP)
  override OMPFLAGS =
endif

#-------------------------------------------------------------------------------

#=== Configure defaults and check if user-defined choices exist for RNDGEN (legacy!), HASCURAND, HASHIPRAND

# If the legacy RNDGEN exists, this take precedence over any HASCURAND choice (but a warning is printed out)
###$(info RNDGEN=$(RNDGEN))
ifneq ($(RNDGEN),)
  $(warning Environment variable RNDGEN is no longer supported, please use HASCURAND instead!)
  ifeq ($(RNDGEN),hasCurand)
    override HASCURAND = $(RNDGEN)
  else ifeq ($(RNDGEN),hasNoCurand)
    override HASCURAND = $(RNDGEN)
  else ifneq ($(RNDGEN),hasNoCurand)
    $(error Unknown RNDGEN='$(RNDGEN)': only 'hasCurand' and 'hasNoCurand' are supported - but use HASCURAND instead!)
  endif
endif

# Set the default HASCURAND (curand random number generator) choice, if no prior choice exists for HASCURAND
# (NB: allow HASCURAND=hasCurand even if $(GPUCC) does not point to nvcc: assume CUDA_HOME was defined correctly...)
ifeq ($(HASCURAND),)
  ifeq ($(GPUCC),) # CPU-only build
    ifneq ($(CUDA_HOME),)
      # By default, assume that curand is installed if a CUDA installation exists
      override HASCURAND = hasCurand
    else
      override HASCURAND = hasNoCurand
    endif
  else ifeq ($(findstring nvcc,$(GPUCC)),nvcc) # Nvidia GPU build
    override HASCURAND = hasCurand
  else # non-Nvidia GPU build
    override HASCURAND = hasNoCurand
  endif
endif

# Set the default HASHIPRAND (hiprand random number generator) choice, if no prior choice exists for HASHIPRAND
# (NB: allow HASHIPRAND=hasHiprand even if $(GPUCC) does not point to hipcc: assume HIP_HOME was defined correctly...)
ifeq ($(HASHIPRAND),)
  ifeq ($(GPUCC),) # CPU-only build
    override HASHIPRAND = hasNoHiprand
  else ifeq ($(findstring hipcc,$(GPUCC)),hipcc) # AMD GPU build
    override HASHIPRAND = hasHiprand
  else # non-AMD GPU build
    override HASHIPRAND = hasNoHiprand
  endif
endif

#-------------------------------------------------------------------------------

#=== Set the CUDA/HIP/C++ compiler flags appropriate to user-defined choices of AVX, FPTYPE, HELINL, HRDCOD

# Set the build flags appropriate to OMPFLAGS
$(info OMPFLAGS=$(OMPFLAGS))
CXXFLAGS += $(OMPFLAGS)

# Set the build flags appropriate to each BACKEND choice (example: "make BACKEND=cppnone")
# [NB MGONGPU_PVW512 is needed because "-mprefer-vector-width=256" is not exposed in a macro]
# [Use 'g++ <buildflags> -E -dM - < /dev/null' to check which #define's are enabled]
# [See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96476]
ifeq ($(UNAME_P),ppc64le)
  ifeq ($(BACKEND),cppsse4)
    override AVXFLAGS = -D__SSE4_2__ # Power9 VSX with 128 width (VSR registers)
  else ifeq ($(BACKEND),cppavx2)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on PowerPC for the moment)
  else ifeq ($(BACKEND),cpp512y)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on PowerPC for the moment)
  else ifeq ($(BACKEND),cpp512z)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on PowerPC for the moment)
  endif
else ifeq ($(UNAME_M),arm64) # ARM on Apple silicon
  ifeq ($(BACKEND),cppnone) # this internally undefines __ARM_NEON
    override AVXFLAGS = -DMGONGPU_NOARMNEON
  else ifeq ($(BACKEND),cppsse4) # __ARM_NEON is always defined on Apple silicon
    override AVXFLAGS =
  else ifeq ($(BACKEND),cppavx2)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on ARM for the moment)
  else ifeq ($(BACKEND),cpp512y)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on ARM for the moment)
  else ifeq ($(BACKEND),cpp512z)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on ARM for the moment)
  endif
else ifeq ($(UNAME_M),aarch64) # ARM on Linux
  ifeq ($(BACKEND),cppnone) # +nosimd ensures __ARM_NEON is absent
    override AVXFLAGS = -march=armv8-a+nosimd
  else ifeq ($(BACKEND),cppsse4) # +simd ensures __ARM_NEON is present (128 width Q/quadword registers)
    override AVXFLAGS = -march=armv8-a+simd
  else ifeq ($(BACKEND),cppavx2)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on aarch64 for the moment)
  else ifeq ($(BACKEND),cpp512y)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on aarch64 for the moment)
  else ifeq ($(BACKEND),cpp512z)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'cppnone' and 'cppsse4' are supported on aarch64 for the moment)
  endif
else ifneq ($(shell $(CXX) --version | grep ^nvc++),) # support nvc++ #531
  ifeq ($(BACKEND),cppnone)
    override AVXFLAGS = -mno-sse3 # no SIMD
  else ifeq ($(BACKEND),cppsse4)
    override AVXFLAGS = -mno-avx # SSE4.2 with 128 width (xmm registers)
  else ifeq ($(BACKEND),cppavx2)
    override AVXFLAGS = -march=haswell # AVX2 with 256 width (ymm registers) [DEFAULT for clang]
  else ifeq ($(BACKEND),cpp512y)
    override AVXFLAGS = -march=skylake -mprefer-vector-width=256 # AVX512 with 256 width (ymm registers) [DEFAULT for gcc]
  else ifeq ($(BACKEND),cpp512z)
    override AVXFLAGS = -march=skylake -DMGONGPU_PVW512 # AVX512 with 512 width (zmm registers)
  endif
else
  ifeq ($(BACKEND),cppnone)
    override AVXFLAGS = -march=x86-64 # no SIMD (see #588)
  else ifeq ($(BACKEND),cppsse4)
    override AVXFLAGS = -march=nehalem # SSE4.2 with 128 width (xmm registers)
  else ifeq ($(BACKEND),cppavx2)
    override AVXFLAGS = -march=haswell # AVX2 with 256 width (ymm registers) [DEFAULT for clang]
  else ifeq ($(BACKEND),cpp512y)
    override AVXFLAGS = -march=skylake-avx512 -mprefer-vector-width=256 # AVX512 with 256 width (ymm registers) [DEFAULT for gcc]
  else ifeq ($(BACKEND),cpp512z)
    override AVXFLAGS = -march=skylake-avx512 -DMGONGPU_PVW512 # AVX512 with 512 width (zmm registers)
  endif
endif
# For the moment, use AVXFLAGS everywhere (in C++ builds): eventually, use them only in encapsulated implementations?
ifeq ($(GPUCC),)
  CXXFLAGS+= $(AVXFLAGS)
endif

# Set the build flags appropriate to each FPTYPE choice (example: "make FPTYPE=f")
$(info FPTYPE='$(FPTYPE)')
ifeq ($(FPTYPE),d)
  CXXFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_DOUBLE
  GPUFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_DOUBLE
else ifeq ($(FPTYPE),f)
  CXXFLAGS += -DMGONGPU_FPTYPE_FLOAT -DMGONGPU_FPTYPE2_FLOAT
  GPUFLAGS += -DMGONGPU_FPTYPE_FLOAT -DMGONGPU_FPTYPE2_FLOAT
else ifeq ($(FPTYPE),m)
  CXXFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_FLOAT
  GPUFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_FLOAT
else
  $(error Unknown FPTYPE='$(FPTYPE)': only 'd', 'f' and 'm' are supported)
endif

# Set the build flags appropriate to each HELINL choice (example: "make HELINL=1")
$(info HELINL='$(HELINL)')
ifeq ($(HELINL),1)
  CXXFLAGS += -DMGONGPU_INLINE_HELAMPS
  GPUFLAGS += -DMGONGPU_INLINE_HELAMPS
else ifneq ($(HELINL),0)
  $(error Unknown HELINL='$(HELINL)': only '0' and '1' are supported)
endif

# Set the build flags appropriate to each HRDCOD choice (example: "make HRDCOD=1")
$(info HRDCOD='$(HRDCOD)')
ifeq ($(HRDCOD),1)
  CXXFLAGS += -DMGONGPU_HARDCODE_PARAM
  GPUFLAGS += -DMGONGPU_HARDCODE_PARAM
else ifneq ($(HRDCOD),0)
  $(error Unknown HRDCOD='$(HRDCOD)': only '0' and '1' are supported)
endif

#=== Set the CUDA/HIP/C++ compiler and linker flags appropriate to user-defined choices of HASCURAND, HASHIPRAND

$(info HASCURAND=$(HASCURAND))
$(info HASHIPRAND=$(HASHIPRAND))
override RNDCXXFLAGS=
override RNDLIBFLAGS=

# Set the RNDCXXFLAGS and RNDLIBFLAGS build flags appropriate to each HASCURAND choice (example: "make HASCURAND=hasNoCurand")
ifeq ($(HASCURAND),hasNoCurand)
  override RNDCXXFLAGS += -DMGONGPU_HAS_NO_CURAND
else ifeq ($(HASCURAND),hasCurand)
  override RNDLIBFLAGS += -L$(CUDA_HOME)/lib64/ -lcurand # NB: -lcuda is not needed here!
else
  $(error Unknown HASCURAND='$(HASCURAND)': only 'hasCurand' and 'hasNoCurand' are supported)
endif

# Set the RNDCXXFLAGS and RNDLIBFLAGS build flags appropriate to each HASHIPRAND choice (example: "make HASHIPRAND=hasNoHiprand")
ifeq ($(HASHIPRAND),hasNoHiprand)
  override RNDCXXFLAGS += -DMGONGPU_HAS_NO_HIPRAND
else ifeq ($(HASHIPRAND),hasHiprand)
  override RNDLIBFLAGS += -L$(HIP_HOME)/lib/ -lhiprand
else ifneq ($(HASHIPRAND),hasHiprand)
  $(error Unknown HASHIPRAND='$(HASHIPRAND)': only 'hasHiprand' and 'hasNoHiprand' are supported)
endif

#$(info RNDCXXFLAGS=$(RNDCXXFLAGS))
#$(info RNDLIBFLAGS=$(RNDLIBFLAGS))

#-------------------------------------------------------------------------------

#=== Configure Position-Independent Code
CXXFLAGS += -fPIC
GPUFLAGS += $(XCOMPILERFLAG) -fPIC

#-------------------------------------------------------------------------------

#=== Configure build directories and build lockfiles ===

# Build lockfile "full" tag (defines full specification of build options that cannot be intermixed)
# (Rationale: avoid mixing of builds with different random number generators)
override TAG = $(patsubst cpp%%,%%,$(BACKEND))_$(FPTYPE)_inl$(HELINL)_hrd$(HRDCOD)_$(HASCURAND)_$(HASHIPRAND)

# Export TAG (so that there is no need to check/define it again in cudacpp_src.mk)
export TAG

# Build directory: current directory by default, or build.$(DIRTAG) if USEBUILDDIR==1
override BUILDDIR = $(CUDACPP_BUILDDIR)
ifeq ($(USEBUILDDIR),1)
  override LIBDIR = ../lib/$(BUILDDIR)
  override LIBDIRRPATH = '$$ORIGIN/$(LIBDIR)'
  $(info Building in BUILDDIR=$(BUILDDIR) for tag=$(TAG) (USEBUILDDIR == 1))
else
  override LIBDIR = ../lib
  override LIBDIRRPATH = '$$ORIGIN/$(LIBDIR)'
  $(info Building in BUILDDIR=$(BUILDDIR) for tag=$(TAG) (USEBUILDDIR != 1))
endif
###override INCDIR = ../../include
###$(info Building in BUILDDIR=$(BUILDDIR) for tag=$(TAG))

# On Linux, set rpath to LIBDIR to make it unnecessary to use LD_LIBRARY_PATH
# Use relative paths with respect to the executables or shared libraries ($ORIGIN on Linux)
# On Darwin, building libraries with absolute paths in LIBDIR makes this unnecessary
ifeq ($(UNAME_S),Darwin)
  override CXXLIBFLAGSRPATH =
  override GPULIBFLAGSRPATH =
  override CXXLIBFLAGSRPATH2 =
  override GPULIBFLAGSRPATH2 =
else
  # RPATH to gpu/cpp libs when linking executables
  override CXXLIBFLAGSRPATH = -Wl,-rpath=$(LIBDIRRPATH)
  override GPULIBFLAGSRPATH = -Xlinker -rpath=$(LIBDIRRPATH)
  # RPATH to common lib when linking gpu/cpp libs
  override CXXLIBFLAGSRPATH2 = -Wl,-rpath='$$ORIGIN'
  override GPULIBFLAGSRPATH2 = -Xlinker -rpath='$$ORIGIN'
endif

# Setting LD_LIBRARY_PATH or DYLD_LIBRARY_PATH in the RUNTIME is no longer necessary (neither on Linux nor on Mac)
override RUNTIME =

#===============================================================================
#=== Makefile TARGETS and build rules below
#===============================================================================

.PHONY: all $(DIRS)

DIRS := $(wildcard P*)

# Construct the library paths
cxx_proclibs := $(shell for dir in $(DIRS); do basename $$dir | awk -F_ '{print "-l mg5amc_"$$(NF-1)"_"$$NF"_cpp"}'; done)
gpu_proclibs := $(shell for dir in $(DIRS); do basename $$dir | awk -F_ '{print "-l mg5amc_"$$(NF-1)"_"$$NF"_$(GPUSUFFIX)"}'; done)

ifeq ($(GPUCC),)
  cxx_rwgt=$(BUILDDIR)/rwgt_driver_cpp.exe
  rwgtlib := $(addprefix ,$(addsuffix /librwgt_cpp.so,$(DIRS)))
else
  gpu_rwgt=$(BUILDDIR)/rwgt_driver_gpu.exe
  rwgtlib := $(addprefix ,$(addsuffix /librwgt_$(GPUSUFFIX).so,$(DIRS)))
endif

# Explicitly define the default goal (this is not necessary as it is the first target, which is implicitly the default goal)
.DEFAULT_GOAL := all.$(TAG)

# First target (default goal)
ifeq ($(GPUCC),)
all.$(TAG): $(BUILDDIR)/.build.$(TAG) $(rwgtlib) $(cxx_rwgt)
else
all.$(TAG): $(BUILDDIR)/.build.$(TAG) $(rwgtlib) $(gpu_rwgt)
endif

# Target (and build options): debug
MAKEDEBUG=
debug: OPTFLAGS   = -g -O0
debug: CUDA_OPTFLAGS = -G
debug: MAKEDEBUG := debug
debug: all.$(TAG)

# Target: tag-specific build lockfiles
override oldtagsb=`if [ -d $(BUILDDIR) ]; then find $(BUILDDIR) -maxdepth 1 -name '.build.*' ! -name '.build.$(TAG)' -exec echo $(shell pwd)/{} \; ; fi`
$(BUILDDIR)/.build.$(TAG):
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	@if [ "$(oldtagsb)" != "" ]; then echo "Cannot build for tag=$(TAG) as old builds exist for other tags:"; echo "  $(oldtagsb)"; echo "Please run 'make clean' first\nIf 'make clean' is not enough: run 'make clean USEBUILDDIR=1 AVX=$(AVX) FPTYPE=$(FPTYPE)' or 'make cleanall'"; exit 1; fi
	@touch $(BUILDDIR)/.build.$(TAG)

# # Apply special build flags only to check_sa_<cpp|$(GPUSUFFIX)>.o (NVTX in timermap.h, #679)
$(BUILDDIR)/rwgt_driver_cpp.o: CXXFLAGS += $(USE_NVTX) $(CUDA_INC)
$(BUILDDIR)/rwgt_driver_gpu.o: CXXFLAGS += $(USE_NVTX) $(CUDA_INC)

# Avoid "warning: builtin __has_trivial_... is deprecated; use __is_trivially_... instead" in GPUCC with icx2023 (#592)
ifneq ($(shell $(CXX) --version | egrep '^(Intel)'),)
ifneq ($(GPUCC),)
GPUFLAGS += -Wno-deprecated-builtins
endif
endif

# Generic target and build rules: objects from C++ compilation
# (NB do not include CUDA_INC here! add it only for NVTX or curand #679)
$(BUILDDIR)/%%_cpp.o : %%.cc *.h ../src/*.h $(BUILDDIR)/.build.$(TAG)
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	$(CXX) $(CPPFLAGS) $(INCFLAGS) $(CXXFLAGS) -c $< -o $@

# Generic target and build rules: objects from CUDA or HIP compilation
ifneq ($(GPUCC),)
$(BUILDDIR)/%%_$(GPUSUFFIX).o : %%.cc *.h ../src/*.h $(BUILDDIR)/.build.$(TAG)
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	$(GPUCC) $(CPPFLAGS) $(INCFLAGS) $(GPUFLAGS) -c -x $(GPULANGUAGE) $< -o $@
endif

#-------------------------------------------------------------------------------

# Target (and build rules): common (src) library
commonlib : $(LIBDIR)/lib$(MG5AMC_COMMONLIB).so
$(LIBDIR)/lib$(MG5AMC_COMMONLIB).so: ../src/*.h ../src/*.cc $(BUILDDIR)/.build.$(TAG)
	$(MAKE) -C ../src $(MAKEDEBUG) -f $(CUDACPP_SRC_MAKEFILE)

#-------------------------------------------------------------------------------

#HERE LOOP MAKE OVER P DIRECTORIES AND ADD RWGT_RUNNER_LIBS
# Ensure each librwgt.a depends on its directory being built
$(rwgtlib): $(commonlib)
	@$(MAKE) -C $(@D) VARIABLE=true

# Target (and build rules): C++ and CUDA/HIP standalone executables
$(cxx_rwgt): LIBFLAGS += $(CXXLIBFLAGSRPATH) # avoid the need for LD_LIBRARY_PATH
$(cxx_rwgt): $(BUILDDIR)/rwgt_driver.o $(rwgtlib)
	$(CXX) -o $@ $(BUILDDIR)/rwgt_driver.o $(OMPFLAGS) -ldl -pthread $(LIBFLAGS) -L$(LIBDIR) $(rwgtlib) 

ifneq ($(GPUCC),)
ifneq ($(shell $(CXX) --version | grep ^Intel),)
$(gpu_rwgt): LIBFLAGS += -lintlc # compile with icpx and link with GPUCC (undefined reference to `_intel_fast_memcpy')
$(gpu_rwgt): LIBFLAGS += -lsvml # compile with icpx and link with GPUCC (undefined reference to `__svml_cos4_l9')
else ifneq ($(shell $(CXX) --version | grep ^nvc++),) # support nvc++ #531
$(gpu_rwgt): LIBFLAGS += -L$(patsubst %%bin/nvc++,%%lib,$(subst ccache ,,$(CXX))) -lnvhpcatm -lnvcpumath -lnvc
endif
$(gpu_rwgt): LIBFLAGS += $(GPULIBFLAGSRPATH) # avoid the need for LD_LIBRARY_PATH
$(gpu_rwgt): $(BUILDDIR)/rwgt_driver.o $(rwgtlib)
	$(GPUCC) -o $@ $(BUILDDIR)/rwgt_driver.o $(CUARCHFLAGS) $(LIBFLAGS) -L$(LIBDIR) $(rwgtlib)
endif

#-------------------------------------------------------------------------------

# Target: build all targets in all BACKEND modes (each BACKEND mode in a separate build directory)
# Split the bldall target into separate targets to allow parallel 'make -j bldall' builds
# (Obsolete hack, no longer needed as there is no INCDIR: add a fbridge.inc dependency to bldall, to ensure it is only copied once for all BACKEND modes)
bldcuda:
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cuda -f $(CUDACPP_MAKEFILE)

bldhip:
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=hip -f $(CUDACPP_MAKEFILE)

bldnone:
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cppnone -f $(CUDACPP_MAKEFILE)

bldsse4:
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cppsse4 -f $(CUDACPP_MAKEFILE)

bldavx2:
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cppavx2 -f $(CUDACPP_MAKEFILE)

bld512y:
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cpp512y -f $(CUDACPP_MAKEFILE)

bld512z:
	@echo
	$(MAKE) USEBUILDDIR=1 BACKEND=cpp512z -f $(CUDACPP_MAKEFILE)

ifeq ($(UNAME_P),ppc64le)
###bldavxs: $(INCDIR)/fbridge.inc bldnone bldsse4
bldavxs: bldnone bldsse4
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
###bldavxs: $(INCDIR)/fbridge.inc bldnone bldsse4
bldavxs: bldnone bldsse4
else
###bldavxs: $(INCDIR)/fbridge.inc bldnone bldsse4 bldavx2 bld512y bld512z
bldavxs: bldnone bldsse4 bldavx2 bld512y bld512z
endif

ifneq ($(HIP_HOME),)
ifneq ($(CUDA_HOME),)
bldall: bldhip bldcuda bldavxs
else
bldall: bldhip bldavxs
endif
else
ifneq ($(CUDA_HOME),)
bldall: bldcuda bldavxs
else
bldall: bldavxs
endif
endif

#-------------------------------------------------------------------------------

# Target: clean the builds
.PHONY: clean clean-rwgtlib

clean: clean-rwgtlib
ifeq ($(USEBUILDDIR),1)
	rm -rf $(BUILDDIR)
else
	rm -f $(BUILDDIR)/.build.* $(BUILDDIR)/*.o $(BUILDDIR)/*.exe
	rm -f $(LIBDIR)/lib*.so
endif
	$(MAKE) -C ../src clean -f $(CUDACPP_SRC_MAKEFILE)
###	rm -rf $(INCDIR)

clean-rwgtlib:
	@for dir in $(DIRS); do $(MAKE) -C $$dir clean; done

cleanall:
	@echo
	$(MAKE) USEBUILDDIR=0 clean -f $(CUDACPP_MAKEFILE)
	@echo
	$(MAKE) USEBUILDDIR=0 -C ../src cleanall -f $(CUDACPP_SRC_MAKEFILE)
	rm -rf build.*

# Target: clean the builds as well as the gtest installation(s)
distclean: cleanall
ifneq ($(wildcard $(TESTDIRCOMMON)),)
	$(MAKE) -C $(TESTDIRCOMMON) clean
endif
	$(MAKE) -C $(TESTDIRLOCAL) clean

#-------------------------------------------------------------------------------

# Target: show system and compiler information
info:
	@echo ""
	@uname -spn # e.g. Linux nodename.cern.ch x86_64
ifeq ($(UNAME_S),Darwin)
	@sysctl -a | grep -i brand
	@sysctl -a | grep machdep.cpu | grep features || true
	@sysctl -a | grep hw.physicalcpu:
	@sysctl -a | grep hw.logicalcpu:
else
	@cat /proc/cpuinfo | grep "model name" | sort -u
	@cat /proc/cpuinfo | grep "flags" | sort -u
	@cat /proc/cpuinfo | grep "cpu cores" | sort -u
	@cat /proc/cpuinfo | grep "physical id" | sort -u
endif
	@echo ""
ifneq ($(shell which nvidia-smi 2>/dev/null),)
	nvidia-smi -L
	@echo ""
endif
	@echo USECCACHE=$(USECCACHE)
ifeq ($(USECCACHE),1)
	ccache --version | head -1
endif
	@echo ""
	@echo GPUCC=$(GPUCC)
ifneq ($(GPUCC),)
	$(GPUCC) --version
endif
	@echo ""
	@echo CXX=$(CXX)
ifneq ($(shell $(CXX) --version | grep ^clang),)
	@echo $(CXX) -v
	@$(CXX) -v |& egrep -v '(Found|multilib)'
	@readelf -p .comment `$(CXX) -print-libgcc-file-name` |& grep 'GCC: (GNU)' | grep -v Warning | sort -u | awk '{print "GCC toolchain:",$$5}'
else
	$(CXX) --version
endif
	@echo ""
	@echo FC=$(FC)
	$(FC) --version

#-------------------------------------------------------------------------------

# Target: 'make test' (execute runTest.exe, and compare check.exe with fcheck.exe)
# [NB: THIS IS WHAT IS TESTED IN THE GITHUB CI!]
# [NB: This used to be called 'make check' but the name has been changed as this has nothing to do with 'check.exe']
test: runTest cmpFcheck

# Target: runTest (run the C++ or CUDA/HIP test executable runTest.exe)
runTest: all.$(TAG)
ifeq ($(GPUCC),)
	$(RUNTIME) $(BUILDDIR)/runTest_cpp.exe
else
	$(RUNTIME) $(BUILDDIR)/runTest_$(GPUSUFFIX).exe
endif

# Target: runCheck (run the C++ or CUDA/HIP standalone executable check.exe, with a small number of events)
runCheck: all.$(TAG)
ifeq ($(GPUCC),)
	$(RUNTIME) $(BUILDDIR)/check_cpp.exe -p 2 32 2
else
	$(RUNTIME) $(BUILDDIR)/check_$(GPUSUFFIX).exe -p 2 32 2
endif

# Target: runFcheck (run the Fortran standalone executable - with C++ or CUDA/HIP MEs - fcheck.exe, with a small number of events)
runFcheck: all.$(TAG)
ifeq ($(GPUCC),)
	$(RUNTIME) $(BUILDDIR)/fcheck_cpp.exe 2 32 2
else
	$(RUNTIME) $(BUILDDIR)/fcheck_$(GPUSUFFIX).exe 2 32 2
endif

# Target: cmpFcheck (compare ME results from the C++/CUDA/HIP and Fortran with C++/CUDA/HIP MEs standalone executables, with a small number of events)
cmpFcheck: all.$(TAG)
	@echo
ifeq ($(GPUCC),)
	@echo "$(BUILDDIR)/check_cpp.exe --common -p 2 32 2"
	@echo "$(BUILDDIR)/fcheck_cpp.exe 2 32 2"
	@me1=$(shell $(RUNTIME) $(BUILDDIR)/check_cpp.exe --common -p 2 32 2 | grep MeanMatrix | awk '{print $$4}'); me2=$(shell $(RUNTIME) $(BUILDDIR)/fcheck_cpp.exe 2 32 2 | grep Average | awk '{print $$4}'); echo "Avg ME (C++/C++)    = $${me1}"; echo "Avg ME (F77/C++)    = $${me2}"; if [ "$${me2}" == "NaN" ]; then echo "ERROR! Fortran calculation (F77/C++) returned NaN"; elif [ "$${me2}" == "" ]; then echo "ERROR! Fortran calculation (F77/C++) crashed"; else python3 -c "me1=$${me1}; me2=$${me2}; reldif=abs((me2-me1)/me1); print('Relative difference =', reldif); ok = reldif <= 2E-4; print ( '%%s (relative difference %%s 2E-4)' %% ( ('OK','<=') if ok else ('ERROR','>') ) ); import sys; sys.exit(0 if ok else 1)"; fi
else
	@echo "$(BUILDDIR)/check_$(GPUSUFFIX).exe --common -p 2 32 2"
	@echo "$(BUILDDIR)/fcheck_$(GPUSUFFIX).exe 2 32 2"
	@me1=$(shell $(RUNTIME) $(BUILDDIR)/check_$(GPUSUFFIX).exe --common -p 2 32 2 | grep MeanMatrix | awk '{print $$4}'); me2=$(shell $(RUNTIME) $(BUILDDIR)/fcheck_$(GPUSUFFIX).exe 2 32 2 | grep Average | awk '{print $$4}'); echo "Avg ME (C++/GPU)   = $${me1}"; echo "Avg ME (F77/GPU)   = $${me2}"; if [ "$${me2}" == "NaN" ]; then echo "ERROR! Fortran calculation (F77/GPU) crashed"; elif [ "$${me2}" == "" ]; then echo "ERROR! Fortran calculation (F77/GPU) crashed"; else python3 -c "me1=$${me1}; me2=$${me2}; reldif=abs((me2-me1)/me1); print('Relative difference =', reldif); ok = reldif <= 2E-4; print ( '%%s (relative difference %%s 2E-4)' %% ( ('OK','<=') if ok else ('ERROR','>') ) ); import sys; sys.exit(0 if ok else 1)"; fi
endif

# Target: cuda-memcheck (run the CUDA standalone executable gcheck.exe with a small number of events through cuda-memcheck)
cuda-memcheck: all.$(TAG)
	$(RUNTIME) $(CUDA_HOME)/bin/cuda-memcheck --check-api-memory-access yes --check-deprecated-instr yes --check-device-heap yes --demangle full --language c --leak-check full --racecheck-report all --report-api-errors all --show-backtrace yes --tool memcheck --track-unused-memory yes $(BUILDDIR)/check_$(GPUSUFFIX).exe -p 2 32 2

#-------------------------------------------------------------------------------
