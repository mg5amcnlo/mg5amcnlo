# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: S. Roiser (Feb 2020) for the MadGraph7 CUDACPP plugin.
# Further modified by: S. Hageboeck, D. Massaro, O. Mattelaer, S. Roiser, J. Teig, A. Valassi (2020-2025).
# Integrated with the MadGraph7 project in Feb 2026.

#=== Check that the user-defined choices of BACKEND, FPTYPE, HELINL, HRDCOD are supported
#=== Configure default values for these variables if no user-defined choices exist

# Set the default BACKEND (CUDA, HIP or C++/SIMD) choice
ifeq ($(BACKEND),)
  override BACKEND = auto
endif

# Set the default FPTYPE (floating point type) choice
# NB: this only affects manual 'make' builds (madevent 'launch' builds are controlled by floating_type in run_card.dat)
ifeq ($(FPTYPE),)
  # OLD DEFAULT UP TO v1.00.00 INCLUDED (inconsistent with default floating_type='m' in run_card.dat)
  ###override FPTYPE = d
  # NEW DEFAULT (#995) AS OF v1.00.01 (now consistent with default floating_type='m' in run_card.dat)
  override FPTYPE = m
endif

# Set the default HELINL (inline helicities?) choice
ifeq ($(HELINL),)
  override HELINL = 0
endif

# Set the default HRDCOD (hardcode cIPD physics parameters?) choice
ifeq ($(HRDCOD),)
  override HRDCOD = 0
endif

# default USEBUILDDIR = 1
ifeq ($(USEBUILDDIR),)
  override USEBUILDDIR = 1
endif

# Check that the user-defined choices of BACKEND, FPTYPE, HELINL, HRDCOD are supported
# (NB: use 'filter' and 'words' instead of 'findstring' because they properly handle whitespace-separated words)
# NB: the CPU backends are the values of the 'cpu_mode' run_card entry; 'auto' picks the
# widest SIMD flavour supported by the host (see the resolution block further down).
override SUPPORTED_CPU_BACKENDS = scalar simd_128 simd_256 avx512y simd_512
override SUPPORTED_BACKENDS = cuda hip $(SUPPORTED_CPU_BACKENDS) auto
ifneq ($(words $(filter $(BACKEND), $(SUPPORTED_BACKENDS))),1)
  $(error Invalid backend BACKEND='$(BACKEND)': supported backends are $(foreach backend,$(SUPPORTED_BACKENDS),'$(backend)'))
endif

# 3 precision macros: amp (MGONGPU_FPTYPE_*), colour (MGONGPU_FPTYPE2_*), momenta/denom (MGONGPU_FPTYPE_MOMENTA_*)
# 5 modes: d=all64, f=all32, m=color32 (colour FP32, momenta+amp FP64), v=denom64 (momenta/denom FP64, colour+amp FP32), e=doubleword expansion (compensated FP64-in-FP32 denom, see MADARITH_DOUBLEEXPANSION below)
override SUPPORTED_FPTYPES = d f m e v
ifneq ($(words $(filter $(FPTYPE), $(SUPPORTED_FPTYPES))),1)
  $(error Invalid fptype FPTYPE='$(FPTYPE)': supported fptypes are $(foreach fptype,$(SUPPORTED_FPTYPES),'$(fptype)'))
endif

override SUPPORTED_HELINLS = 0 1
ifneq ($(words $(filter $(HELINL), $(SUPPORTED_HELINLS))),1)
  $(error Invalid helinl HELINL='$(HELINL)': supported helinls are $(foreach helinl,$(SUPPORTED_HELINLS),'$(helinl)'))
endif

override SUPPORTED_HRDCODS = 0 1
ifneq ($(words $(filter $(HRDCOD), $(SUPPORTED_HRDCODS))),1)
  $(error Invalid hrdcod HRDCOD='$(HRDCOD)': supported hrdcods are $(foreach hrdcod,$(SUPPORTED_HRDCODS),'$(hrdcod)'))
endif

# Stop immediately if BACKEND=cuda but nvcc is missing
ifeq ($(BACKEND),cuda)
  ifeq ($(shell which nvcc 2>/dev/null),)
    $(error BACKEND=$(BACKEND) but nvcc was not found)
  endif
endif

# Stop immediately if BACKEND=hip but hipcc is missing
ifeq ($(BACKEND),hip)
  ifeq ($(shell which hipcc 2>/dev/null),)
    $(error BACKEND=$(BACKEND) but hipcc was not found)
  endif
endif

#=== Configure MADMATRIX_BUILDDIR

# Build directory "full" tag (used for build lockfiles to prevent mixing builds with different options)
# NB: the backend name is used as-is ('simd_128', 'scalar', 'cuda', 'hip'...), see TAG below.
override DIRTAG := $(BACKEND)_$(FPTYPE)_inl$(HELINL)_hrd$(HRDCOD)

# Build directory: build.<BACKEND> by default (USEBUILDDIR=1), or current directory if USEBUILDDIR=0
# NB: using '=' (not ':=') ensures BACKEND is evaluated lazily after the potential 'auto' resolution
ifeq ($(USEBUILDDIR),1)
  override MADMATRIX_BUILDDIR = build.$(BACKEND)
else
  override MADMATRIX_BUILDDIR = .
endif

# Export MADMATRIX_BUILDDIR to src/Makefile
export MADMATRIX_BUILDDIR

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
# src folder
override SRC := ../../src

#-------------------------------------------------------------------------------

# Library output directory: libraries are ALWAYS placed in LIBDIR, independently of USEBUILDDIR.
# Can be overridden on the command line (e.g. make LIBDIR=/abs/path); defaults to ../../lib.
# NB: LIBDIR is resolved to an absolute path so it can be passed unchanged to sub-makes via export.
LIBDIR ?= ../../lib
# Keep LIBDIR as given (usually relative to this P* directory): it is used below to
# build loader-relative rpaths, which survive a move of the process directory.
override LIBDIR_AS_GIVEN := $(LIBDIR)
override LIBDIR := $(abspath $(LIBDIR))

$(info Building objects in BUILDDIR=$(BUILDDIR), libraries in LIBDIR=$(LIBDIR))

#-------------------------------------------------------------------------------

#=== Redefine BACKEND if the current value is 'auto'

# Set the BACKEND choice corresponding to 'auto' (the 'best' C++ vectorization available)
BACKEND_ORIG := $(BACKEND)
ifeq ($(BACKEND),auto)
  ifeq ($(UNAME_P),ppc64le)
    override BACKEND = simd_128
  else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
    override BACKEND = simd_128
  else ifeq ($(wildcard /proc/cpuinfo),)
    override BACKEND = scalar
    ###$(warning Using BACKEND='$(BACKEND)' because host SIMD features cannot be read from /proc/cpuinfo)
  else ifeq ($(shell grep -m1 -c avx512vl /proc/cpuinfo)$(shell $(CXX) --version | grep ^clang),1)
    override BACKEND = avx512y
  else ifeq ($(shell grep -m1 -c avx2 /proc/cpuinfo),1)
    override BACKEND = simd_256
    ###ifneq ($(shell grep -m1 -c avx512vl /proc/cpuinfo),1)
    ###  $(warning Using BACKEND='$(BACKEND)' because host does not support avx512vl)
    ###else
    ###  $(warning Using BACKEND='$(BACKEND)' because this is faster than avx512vl for clang)
    ###endif
  else ifeq ($(shell grep -m1 -c sse4_2 /proc/cpuinfo),1)
    override BACKEND = simd_128
  else
    override BACKEND = scalar
  endif
  $(info BACKEND=$(BACKEND) (was auto))
else
  $(info BACKEND='$(BACKEND)')
endif

# Create file with the resolved backend in case user chooses 'auto'
BACKEND_LOG ?= .resolved-backend
ifneq ($(BACKEND_ORIG),$(BACKEND))
  $(file >$(BACKEND_LOG),$(BACKEND))
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

# Export CXXFLAGS (so that there is no need to check/define it again in src/Makefile)
export CXXFLAGS

#-------------------------------------------------------------------------------

#=== Configure the GPU compiler (CUDA or HIP)
#=== (note, this is done also for C++, as NVTX and CURAND/ROCRAND are also needed by the C++ backends)

# Set CUDA_HOME from the path to nvcc, if it exists
override CUDA_HOME = $(patsubst %%/bin/nvcc,%%,$(shell which nvcc 2>/dev/null))

# Set HIP_HOME from the path to hipcc, if it exists
override HIP_HOME = $(shell hipconfig --rocmpath 2>/dev/null)

# Configure CUDA_INC (for CURAND and NVTX) and NVTX if a CUDA installation exists (see #965)
ifeq ($(CUDA_HOME),)
  # CUDA_HOME is empty (nvcc not found)
  override CUDA_INC=
else ifeq ($(wildcard $(CUDA_HOME)/include/),)
  # CUDA_HOME is defined (nvcc was found) but $(CUDA_HOME)/include/ does not exist?
  override CUDA_INC=
else
  CUDA_INC = -I$(CUDA_HOME)/include/
endif
###$(info CUDA_INC=$(CUDA_INC))

# Configure NVTX if a CUDA include directory exists and NVTX headers exist (see #965)
ifeq ($(CUDA_INC),)
  # $(CUDA_HOME)/include/ does not exist
  override USE_NVTX=
else ifeq ($(wildcard $(CUDA_HOME)/include/nvtx3/nvToolsExt.h),)
  # $(CUDA_HOME)/include/ exists but NVTX headers do not exist?
  override USE_NVTX=
else
  # $(CUDA_HOME)/include/nvtx.h exists: use NVTX
  # (NB: the option to disable NVTX if 'USE_NVTX=' is defined has been removed)
  override USE_NVTX=-DUSE_NVTX
endif
###$(info USE_NVTX=$(USE_NVTX))

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
# Note also that a preliminary check for nvcc and hipcc if BACKEND is cuda or hip is performed at the top of this Makefile.
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

  # Optimization flags
  GPUFLAGS = $(foreach opt, $(OPTFLAGS), $(XCOMPILERFLAG) $(opt))

  # NVidia CUDA architecture flags
  # See https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
  # See https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
  # Default: detect all compute capability (e.g., "8.0", "8.6", "9.0"), unique and sorted from lowest to higherst
  # then we embed device code for each compute capability, and for the highest PTX (forward-compatible)
  # use nvidia-smi and validate output with grep before going forward
  DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+\.[0-9]+$$' | tr -d '.' | sort -un)
  # One may pass MADGRAPH_CUDA_ARCHITECTURE (comma-separated list) to the make command to use another value or list of values (see #533).
  # Examples: use 60 for P100 (Piz Daint), 80 for A100 (Juwels Booster, NVidia raplab/Curiosity).
  comma:=,
  MADGRAPH_CUDA_ARCHITECTURE ?= $(foreach arch,$(DETECTED_CC),$(arch)$(comma))
  # Convert to space-separated list for looping
  MADGRAPH_CUDA_ARCH_LIST ?= $(subst $(comma), ,$(MADGRAPH_CUDA_ARCHITECTURE))

  # Fallback if detection failed (box has CUDA selected but probe failed)
  ifeq ($(strip $(MADGRAPH_CUDA_ARCH_LIST)),)
	# Default: use compute capability 70 for V100 (CERN lxbatch, CERN itscrd, Juwels Cluster)
	# This will embed device code for 70, and PTX for 70+
    MADGRAPH_CUDA_ARCHITECTURE := 70
    MADGRAPH_CUDA_ARCH_LIST := 70
    $(info Automatic compute capability detection failed; defaulting to $(MADGRAPH_CUDA_ARCHITECTURE))
    $(info Override with: make MADGRAPH_CUDA_ARCHITECTURE=<comma-separated list of architectures>)
  endif

  # Build for every detected SM, and add one PTX for the highest SM (forward-compatibility)
  HIGHEST_SM    := $(lastword $(MADGRAPH_CUDA_ARCH_LIST))
  GENCODE_FLAGS := $(foreach arch,$(MADGRAPH_CUDA_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch))
  GENCODE_PTX   := -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
  GPUARCHFLAGS  := $(GENCODE_FLAGS) $(GENCODE_PTX)
  GPUFLAGS += $(GPUARCHFLAGS)

  # Other NVidia-specific flags
  CUDA_OPTFLAGS = -lineinfo
  GPUFLAGS += $(CUDA_OPTFLAGS)

  # NVCC version
  ###GPUCC_VERSION = $(shell $(GPUCC) --version | grep 'Cuda compilation tools' | cut -d' ' -f5 | cut -d, -f1)

  # Fast math
  GPUFLAGS += -use_fast_math

  # Extra build warnings
  GPUFLAGS += $(XCOMPILERFLAG) -Wunused-parameter
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

  # example architecture values MI200:gfx90a, MI350X:gfx942
  # auto detect arch if not set; if fail gfx942 default
  ifeq ($(origin MADGRAPH_HIP_ARCHITECTURE),undefined)
    MADGRAPH_HIP_ARCHITECTURE := $(shell $(HIP_HOME)/bin/rocm_agent_enumerator 2>/dev/null | grep -v gfx000 | sort -u | head -1)
  endif
  ifeq ($(MADGRAPH_HIP_ARCHITECTURE),)
    MADGRAPH_HIP_ARCHITECTURE := gfx942
  endif
  # Set GPUCC as $(HIP_HOME)/bin/hipcc (it was already checked above that this exists)
  GPUCC = $(HIP_HOME)/bin/hipcc
  XCOMPILERFLAG =
  GPULANGUAGE = hip
  GPUSUFFIX = hip

  # Optimization flags (HIP -O2 workaround applied after OPTFLAGS = -O3 below, see #806)
  GPUFLAGS = $(foreach opt, $(OPTFLAGS), $(XCOMPILERFLAG) $(opt))

  # DEBUG FLAGS (for #806: see https://hackmd.io/@gmarkoma/lumi_finland)
  ###GPUFLAGS += -ggdb # FOR DEBUGGING ONLY

  # AMD HIP architecture flags
  GPUARCHFLAGS = --offload-arch=${MADGRAPH_HIP_ARCHITECTURE}
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
  # In practice, in the following, "ifeq ($(GPUCC),)" is equivalent to "BACKEND is one of SUPPORTED_CPU_BACKENDS"
  # (NB: do NOT test the backend name for a 'cpu' substring: none of the CPU backend names contains one).
  # Conversely, note that GPUFLAGS is non-empty also for C++ builds, but it is never used in that case.
  ifneq ($(words $(filter $(BACKEND), $(SUPPORTED_CPU_BACKENDS))),1)
    $(error INTERNAL ERROR! Unknown backend BACKEND='$(BACKEND)': supported backends are $(foreach backend,$(SUPPORTED_BACKENDS),'$(backend)'))
  endif

endif

# Export GPUCC, GPUFLAGS, GPULANGUAGE, GPUSUFFIX (so that there is no need to check/define them again in src/Makefile)
export GPUCC
export GPUFLAGS
export GPULANGUAGE
export GPUSUFFIX

# Export BACKEND (resolved from auto above if needed; used e.g. to name the common library)
export BACKEND

# Map BACKEND to its backend/<variant> source subdirectory
ifneq ($(GPUCC),)
  override BACKENDDIR = gpu
else ifeq ($(BACKEND),scalar)
  override BACKENDDIR = cpu
else
  override BACKENDDIR = simd
endif
export BACKENDDIR

# Export FPTYPE (used e.g. to name the common library, see MADMATRIX_COMMONLIB below)
export FPTYPE

#-------------------------------------------------------------------------------

#=== Configure ccache for C++ and CUDA/HIP builds

# Enable ccache only if USECCACHE=1
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

INCFLAGS = -I. -I../../backend/$(BACKENDDIR) -I../../backend/common
OPTFLAGS = -O3

# HIP requires -O2 to avoid "Memory access fault" in gq_ttq (#806)
ifeq ($(BACKEND),hip)
  override OPTFLAGS = -O2
endif

# PROFILE=1: unchanged optimisation (-O3) + symbols and frame pointers for profilers (perf, Adaptyst, gprof, valgrind...)
# DEBUG=1  : no optimisation + full debug symbols
# Both flags propagate automatically to src/ sub-makes via MAKEFLAGS.
# DEBUG overrides the HIP workaround above intentionally: it must always win.
ifeq ($(DEBUG),1)
  override OPTFLAGS = -O0
endif

# PCH=1: precompile the standard headers once per build instead of once per file.
# Every translation unit of a subprocess reparses the same standard headers (~0.3 s
# each with clang, i.e. most of what a small process costs, six files per
# subprocess), so for many small processes -- the decay widths, for instance --
# this makes the build ~3x faster. The precompiled header is shared by all the P*
# directories (it lives in LIBDIR, tagged like the build) and is made up front by
# 'commonlib', like the common library. Opt-in, and clang with a C++ (CPU) backend
# only: other compilers and the GPU compilers have their own requirements. The mg7
# launcher asks for it from 4 subprocess directories to build (PCH_MIN_SUBPROCESSES):
# below, the header costs about what it saves.
PCHFLAGS =
ifeq ($(PCH),1)
  ifeq ($(GPUCC),)
    ifneq ($(shell $(CXX) --version | grep -i clang),)
      override PCHFILE = $(LIBDIR)/pch_std_$(TAG).h.pch
      override PCHFLAGS = -include-pch $(PCHFILE)
    endif
  endif
endif

# Dependency on src directory
# The common library name carries the model name and the full BACKEND suffix -- see the
# matching comment in madmatrix_src.mk for why the model name is required (dlopen
# collisions across process directories regenerated in different models) -- and, purely
# for naming consistency with MADMATRIX_LIB, the FPTYPE too; unlike MADMATRIX_LIB this
# isn't consumed by anything outside these madmatrix makefiles (subprocesses.json never
# references it directly), so no further propagation is needed for this one.
# LIBTAG (empty by default) goes into the name of every library of this output: the
# dynamic loader identifies a library by its name, so two outputs loaded into one
# process (MadSpin's production and decay matrix elements) need different ones.
MADMATRIX_COMMONLIB = madmatrix_common$(LIBTAG)_%(model)s_$(BACKEND)_$(FPTYPE)
LIBFLAGS = -L$(LIBDIR) -l$(MADMATRIX_COMMONLIB)
INCFLAGS += -I$(SRC) -I$(SRC)/rambo

#-------------------------------------------------------------------------------

#=== Configure PowerPC-specific compiler flags for C++ and CUDA/HIP

# PowerPC-specific CXX compiler flags (being reviewed)
ifeq ($(UNAME_P),ppc64le)
  CXXFLAGS+= -mcpu=power9 -mtune=power9 # gains ~2-3%% both for scalar and simd_128
  # Throughput references without the extra flags below: scalar=1.41-1.42E6, simd_128=2.15-2.19E6
  ###CXXFLAGS+= -DNO_WARN_X86_INTRINSICS # no change
  ###CXXFLAGS+= -fpeel-loops # no change
  ###CXXFLAGS+= -funroll-loops # gains ~1%% for scalar, loses ~1%% for simd_128
  ###CXXFLAGS+= -ftree-vectorize # no change
  ###CXXFLAGS+= -flto # would increase to scalar=4.08-4.12E6, simd_128=4.99-5.03E6!
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
    $(error OpenMP is not supported by MadMatrix on clang16 - issue #904)
  else ifneq ($(shell $(CXX) --version | egrep '^clang version 17'),)
    ###override OMPFLAGS = # disable OpenMP on clang17 #904
    $(error OpenMP is not supported by MadMatrix on clang17 - issue #904)
  else ifneq ($(shell $(CXX) --version | egrep '^(clang)'),)
    override OMPFLAGS = -fopenmp
    ###override OMPFLAGS = # disable OpenMP MT on clang (was not ok without or with nvcc before #578)
  ###else ifneq ($(shell $(CXX) --version | egrep '^(Apple clang)'),) # AV for Mac (Apple clang compiler)
  else ifeq ($(UNAME_S),Darwin) # OM for Mac (any compiler)
    override OMPFLAGS = # AV disable OpenMP MT on Apple clang (builds fail in the CI #578)
    ###override OMPFLAGS = -fopenmp # OM re-enable OpenMP MT on Apple clang? (AV Oct 2023: this still fails in the CI)
  else
    override OMPFLAGS = -fopenmp # enable OpenMP MT by default on all other platforms
    ###override OMPFLAGS = # disable OpenMP MT on all other platforms (default before #575)
  endif
else
  ###$(info USEOPENMP!=1: will build without OpenMP)
  override OMPFLAGS =
endif

#-------------------------------------------------------------------------------

#=== Configure defaults and check if user-defined choices exist for HASBLAS

# Set the default HASBLAS (cuBLAS/hipBLAS) choice and check prior choices for HASBLAS

ifeq ($(HASBLAS),)
  ifeq ($(GPUCC),) # CPU-only build
    override HASBLAS = hasNoBlas
  else ifeq ($(findstring nvcc,$(GPUCC)),nvcc) # Nvidia GPU build
    ifeq ($(wildcard $(CUDA_HOME)/include/cublas_v2.h),)
      # cuBLAS headers do not exist??
      override HASBLAS = hasNoBlas
    else
      override HASBLAS = hasBlas
    endif
  else ifeq ($(findstring hipcc,$(GPUCC)),hipcc) # AMD GPU build
    ifeq ($(wildcard $(HIP_HOME)/include/hipblas/hipblas.h),)
      # hipBLAS headers do not exist??
      override HASBLAS = hasNoBlas
    else
      override HASBLAS = hasBlas
    endif
  else
    override HASBLAS = hasNoBlas
  endif
endif

#-------------------------------------------------------------------------------

#=== Set the CUDA/HIP/C++ compiler flags appropriate to user-defined choices of AVX, FPTYPE, HELINL, HRDCOD

# Set the build flags appropriate to OMPFLAGS
$(info OMPFLAGS=$(OMPFLAGS))
CXXFLAGS += $(OMPFLAGS)

# Set the build flags appropriate to each BACKEND choice (example: "make BACKEND=scalar")
# [NB MGONGPU_PVW512 is needed because "-mprefer-vector-width=256" is not exposed in a macro]
# [See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96476]
# [Use 'g++ <buildflags> -E -dM - < /dev/null' to check which #define's are enabled]
ifeq ($(UNAME_P),ppc64le)
  ifeq ($(BACKEND),simd_128)
    override AVXFLAGS = -D__SSE4_2__ # Power9 VSX with 128 width (VSR registers)
  else ifeq ($(BACKEND),simd_256)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on PowerPC for the moment)
  else ifeq ($(BACKEND),avx512y)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on PowerPC for the moment)
  else ifeq ($(BACKEND),simd_512)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on PowerPC for the moment)
  endif
else ifeq ($(UNAME_M),arm64) # ARM on Apple silicon
  ifeq ($(BACKEND),scalar) # this internally undefines __ARM_NEON
    override AVXFLAGS = -DMGONGPU_NOARMNEON
  else ifeq ($(BACKEND),simd_128) # __ARM_NEON is always defined on Apple silicon
    override AVXFLAGS =
  else ifeq ($(BACKEND),simd_256)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on ARM for the moment)
  else ifeq ($(BACKEND),avx512y)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on ARM for the moment)
  else ifeq ($(BACKEND),simd_512)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on ARM for the moment)
  endif
else ifeq ($(UNAME_M),aarch64) # ARM on Linux
  ifeq ($(BACKEND),scalar) # +nosimd ensures __ARM_NEON is absent
    override AVXFLAGS = -march=armv8-a+nosimd
  else ifeq ($(BACKEND),simd_128) # +simd ensures __ARM_NEON is present (128 width Q/quadword registers)
    override AVXFLAGS = -march=armv8-a+simd
  else ifeq ($(BACKEND),simd_256)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on aarch64 for the moment)
  else ifeq ($(BACKEND),avx512y)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on aarch64 for the moment)
  else ifeq ($(BACKEND),simd_512)
    $(error Invalid SIMD BACKEND='$(BACKEND)': only 'scalar' and 'simd_128' are supported on aarch64 for the moment)
  endif
else ifneq ($(shell $(CXX) --version | grep ^nvc++),) # support nvc++ #531
  ifeq ($(BACKEND),scalar)
    override AVXFLAGS = -mno-sse3 # no SIMD
  else ifeq ($(BACKEND),simd_128)
    override AVXFLAGS = -mno-avx # SSE4.2 with 128 width (xmm registers)
  else ifeq ($(BACKEND),simd_256)
    override AVXFLAGS = -march=haswell # AVX2 with 256 width (ymm registers) [DEFAULT for clang]
  else ifeq ($(BACKEND),avx512y)
    override AVXFLAGS = -march=skylake -mprefer-vector-width=256 # AVX512 with 256 width (ymm registers) [DEFAULT for gcc]
  else ifeq ($(BACKEND),simd_512)
    override AVXFLAGS = -march=skylake -DMGONGPU_PVW512 # AVX512 with 512 width (zmm registers)
  endif
else
  ifeq ($(BACKEND),scalar)
    override AVXFLAGS = -march=x86-64 # no SIMD (see #588)
  else ifeq ($(BACKEND),simd_128)
    override AVXFLAGS = -march=nehalem # SSE4.2 with 128 width (xmm registers)
  else ifeq ($(BACKEND),simd_256)
    override AVXFLAGS = -march=haswell # AVX2 with 256 width (ymm registers) [DEFAULT for clang]
  else ifeq ($(BACKEND),avx512y)
    override AVXFLAGS = -march=skylake-avx512 -mprefer-vector-width=256 # AVX512 with 256 width (ymm registers) [DEFAULT for gcc]
  else ifeq ($(BACKEND),simd_512)
    override AVXFLAGS = -march=skylake-avx512 -DMGONGPU_PVW512 # AVX512 with 512 width (zmm registers)
  endif
endif
# For the moment, use AVXFLAGS everywhere (in C++ builds): eventually, use them only in encapsulated implementations?
ifeq ($(GPUCC),)
  CXXFLAGS+= $(AVXFLAGS)
endif

# Set the build flags appropriate to each FPTYPE choice (example: "make FPTYPE=f")
# 3 precision macros: amp (MGONGPU_FPTYPE_*), colour (MGONGPU_FPTYPE2_*), momenta/denom (MGONGPU_FPTYPE_MOMENTA_*)
# 5 modes: d=all64, f=all32, m=color32 (colour FP32, momenta+amp FP64), v=denom64 (momenta/denom FP64, colour+amp FP32), e=doubleword expansion (compensated FP64-in-FP32 denom, see MADARITH_DOUBLEEXPANSION below)
$(info FPTYPE='$(FPTYPE)')
ifeq ($(FPTYPE),d) # all64
  CXXFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_DOUBLE
  GPUFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_DOUBLE
else ifeq ($(FPTYPE),f) # all32
  CXXFLAGS += -DMGONGPU_FPTYPE_FLOAT -DMGONGPU_FPTYPE2_FLOAT
  GPUFLAGS += -DMGONGPU_FPTYPE_FLOAT -DMGONGPU_FPTYPE2_FLOAT
else ifeq ($(FPTYPE),m) # color32: colour FP32, momenta+amp FP64
  CXXFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_FLOAT
  GPUFLAGS += -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_FLOAT
else ifeq ($(FPTYPE),v) # denom64: momenta/denom FP64, colour+amp FP32
  # cppnone/cuda/hip trivial, SIMD denom ex. twice, rest narrowed 
  GPUFLAGS += -DMGONGPU_FPTYPE_FLOAT -DMGONGPU_FPTYPE2_FLOAT -DMGONGPU_FPTYPE_MOMENTA_DOUBLE
  CXXFLAGS += -DMGONGPU_FPTYPE_FLOAT -DMGONGPU_FPTYPE2_FLOAT -DMGONGPU_FPTYPE_MOMENTA_DOUBLE
else ifeq ($(FPTYPE),e)
  CXXFLAGS += -DMADARITH_DOUBLEEXPANSION -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_FLOAT
  GPUFLAGS += -DMADARITH_DOUBLEEXPANSION -DMGONGPU_FPTYPE_DOUBLE -DMGONGPU_FPTYPE2_FLOAT
else
  $(error Unknown FPTYPE='$(FPTYPE)': supported fptypes are $(foreach fptype,$(SUPPORTED_FPTYPES),'$(fptype)'))
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

#=== Set the CUDA/HIP/C++ compiler and linker flags appropriate to user-defined choices of HASBLAS

$(info HASBLAS=$(HASBLAS))
override BLASCXXFLAGS=
override BLASLIBFLAGS=

# Set the BLASCXXFLAGS and BLASLIBFLAGS build flags appropriate to each HASBLAS choice (example: "make HASBLAS=hasNoBlas")
ifeq ($(HASBLAS),hasNoBlas)
  override BLASCXXFLAGS += -DMGONGPU_HAS_NO_BLAS
else ifeq ($(HASBLAS),hasBlas)
  ifeq ($(findstring nvcc,$(GPUCC)),nvcc) # Nvidia GPU build
    override BLASLIBFLAGS = -L$(CUDA_HOME)/lib64/ -lcublas
  else ifeq ($(findstring hipcc,$(GPUCC)),hipcc) # AMD GPU build
    override BLASLIBFLAGS = -L$(HIP_HOME)/lib/ -lhipblas
  endif
else
  $(error Unknown HASBLAS='$(HASBLAS)': only 'hasBlas' and 'hasNoBlas' are supported)
endif
CXXFLAGS += $(BLASCXXFLAGS)
GPUFLAGS += $(BLASCXXFLAGS)

#$(info BLASCXXFLAGS=$(BLASCXXFLAGS))
#$(info BLASLIBFLAGS=$(BLASLIBFLAGS))

#-------------------------------------------------------------------------------

#=== Configure defaults and check if user-defined choices exist for CPPBLAS

# HASBLAS above is about cuBLAS/hipBLAS, which only a GPU build can use. CPPBLAS
# is the separate question of whether the C++ color sum goes through a host BLAS:
# the color matrix does not depend on the helicity, so all the good helicities of
# one event page are the columns of one SYMM call. Whether a BLAS carrying SYMM
# could be linked was settled when this directory was written out; it is only
# taken for processes whose color matrix is large enough to be worth it, and for
# those the generated color_sum.cc carries both paths (example: "make CPPBLAS=hasNoBlas").
ifeq ($(CPPBLAS),)
  ifeq ($(GPUCC),) # CPU-only build
    override CPPBLAS = %(cpp_blas_default)s
  else # the GPU build does its color sum on the device
    override CPPBLAS = hasNoBlas
  endif
endif

override CPPBLASCXXFLAGS=
override CPPBLASLIBFLAGS=

ifeq ($(CPPBLAS),hasBlas)
  override CPPBLASCXXFLAGS += -DMGONGPU_CPP_HAS_BLAS
  override CPPBLASLIBFLAGS += %(cpp_blas_libflags)s
else ifneq ($(CPPBLAS),hasNoBlas)
  $(error Unknown CPPBLAS='$(CPPBLAS)': only 'hasBlas' and 'hasNoBlas' are supported)
endif
CXXFLAGS += $(CPPBLASCXXFLAGS)

#$(info CPPBLAS=$(CPPBLAS))
#$(info CPPBLASLIBFLAGS=$(CPPBLASLIBFLAGS))

#-------------------------------------------------------------------------------

#=== Configure Position-Independent Code
CXXFLAGS += -fPIC
GPUFLAGS += $(XCOMPILERFLAG) -fPIC

#-------------------------------------------------------------------------------

#=== Configure channelid debugging
ifneq ($(MADMATRIX_CHANNELID_DEBUG),)
  CXXFLAGS += -DMGONGPU_CHANNELID_DEBUG
  GPUFLAGS += -DMGONGPU_CHANNELID_DEBUG
endif

#=== Configure profiling/debug symbols
ifneq ($(filter 1,$(PROFILE) $(DEBUG)),)
  CXXFLAGS += -g -fno-omit-frame-pointer
  GPUFLAGS += $(XCOMPILERFLAG) -g $(XCOMPILERFLAG) -fno-omit-frame-pointer
  # Frame pointers in leaf functions too, for complete stacks (option only exists on x86 and AArch64).
  ifneq ($(filter x86_64 aarch64 arm64,$(UNAME_M)),)
    CXXFLAGS += -mno-omit-leaf-frame-pointer
    GPUFLAGS += $(XCOMPILERFLAG) -mno-omit-leaf-frame-pointer
  endif
endif

#-------------------------------------------------------------------------------

#=== Configure build directories and build lockfiles ===

# Build lockfile "full" tag (defines full specification of object-file builds that cannot be intermixed)
# NB: the backend name is used as-is, so it matches DIRTAG above (there is no common prefix to strip)
override TAG = $(BACKEND)_$(FPTYPE)_inl$(HELINL)_hrd$(HRDCOD)

# Export TAG (so that there is no need to check/define it again in src/Makefile)
export TAG

# Build directory for object files: build.<BACKEND> by default, or current directory if USEBUILDDIR=0
override BUILDDIR = $(MADMATRIX_BUILDDIR)

###override INCDIR = ../../include

# Embed rpaths so that LD_LIBRARY_PATH/DYLD_LIBRARY_PATH is not needed. They are
# expressed relative to the loading binary ($ORIGIN on Linux, @loader_path on
# Darwin) so that the whole process directory can be moved after the build.
ifeq ($(UNAME_S),Darwin)
  override RPATHORIGIN := @loader_path
else
  override RPATHORIGIN := $$ORIGIN
endif

# RPATH for executables (built in this P* directory): find the process lib and its
# common-lib dependency in LIBDIR. Only an absolute LIBDIR forces an absolute rpath.
ifeq ($(filter /%%,$(LIBDIR_AS_GIVEN)),)
  override RPATHLIBDIR := '$(RPATHORIGIN)/$(LIBDIR_AS_GIVEN)'
else
  override RPATHLIBDIR := $(LIBDIR)
endif
override CXXLIBFLAGSRPATH = -Wl,-rpath,$(RPATHLIBDIR)
override GPULIBFLAGSRPATH = -Xlinker -rpath -Xlinker $(RPATHLIBDIR)
# RPATH for the process shared lib itself: the common lib sits next to it in LIBDIR
override CXXLIBFLAGSRPATH2 = -Wl,-rpath,'$(RPATHORIGIN)'
override GPULIBFLAGSRPATH2 = -Xlinker -rpath -Xlinker '$(RPATHORIGIN)'

# Name recorded in the process lib itself, and hence in whatever links against it.
# Without it the linker bakes in the absolute build path of the library.
ifeq ($(UNAME_S),Darwin)
  override CXXLIBFLAGSNAME = -Wl,-install_name,@rpath/lib$(MADMATRIX_LIB).so
else
  override CXXLIBFLAGSNAME = -Wl,-soname,lib$(MADMATRIX_LIB).so
endif
override GPULIBFLAGSNAME = -Xlinker -soname -Xlinker lib$(MADMATRIX_LIB).so

# Setting LD_LIBRARY_PATH or DYLD_LIBRARY_PATH in the RUNTIME is no longer necessary (neither on Linux nor on Mac)
override RUNTIME =

#===============================================================================
#=== Makefile TARGETS and build rules below
#===============================================================================

processid_short=$(shell basename $(CURDIR))
###$(info processid_short=$(processid_short))

# The process library name carries LIBTAG (see MADMATRIX_COMMONLIB above), the model
# name (same %(model)s substitution, and the same dlopen-collision rationale, as
# MADMATRIX_COMMONLIB above), the process id, the hardware backend, and the
# floating-point type -- so that libraries built for different models/precisions/backends
# out of the same process directory never collide, and so that a loader can tell, from
# the filename alone, exactly which of those a given library on disk was built with
# (subprocesses.json itself stays agnostic to BACKEND and FPTYPE -- see me_lib_format in
# madmatrix/output.py -- since a given process directory can be compiled for several of
# each without regenerating the JSON).
MADMATRIX_LIB = madmatrix$(LIBTAG)_%(model)s_$(processid_short)_$(BACKEND)_$(FPTYPE)
objects_lib=$(BUILDDIR)/CPPProcess.o $(BUILDDIR)/color_sum.o $(BUILDDIR)/MatrixElementKernels.o $(BUILDDIR)/CrossSectionKernels.o $(BUILDDIR)/umami.o $(BUILDDIR)/SigmaKin.o
# The spin-density matrices for MadSpin (backend/common/Density.h): C++ backends,
# and only in the builds madmatrix/density.py asks for (DENSITY=1, the scalar
# libraries MadSpin loads), not in every library of every output
ifeq ($(DENSITY)$(GPUCC),1)
  objects_lib += $(BUILDDIR)/Density.o
endif

# Backend-owned sources (the backend variant first, then what all of them share)
vpath %%.cc ../../backend/$(BACKENDDIR)
vpath %%.cc ../../backend/common

# Rambo/random-number sources
vpath %%.cc ../../src/rambo

# Explicitly define the default goal (this is not necessary as it is the first target, which is implicitly the default goal)
.DEFAULT_GOAL := all.$(TAG)

# First target (default goal): build the process library (which also builds the common library as a dependency)
all.$(TAG): $(BUILDDIR)/.build.$(TAG) $(LIBDIR)/lib$(MADMATRIX_LIB).so

# Target: the process library alone, whatever the default goal of the makefile
# including this one (the standalone one adds check_sa.exe to it)
.PHONY: lib
lib: all.$(TAG)

# Target (and build options): address sanitizer #207
###CXXLIBFLAGSASAN =
###GPULIBFLAGSASAN =
###asan: OPTFLAGS = -g -O0 -fsanitize=address -fno-omit-frame-pointer
###asan: CUDA_OPTFLAGS = -G $(XCOMPILERFLAG) -fsanitize=address $(XCOMPILERFLAG) -fno-omit-frame-pointer
###asan: CXXLIBFLAGSASAN = -fsanitize=address
###asan: GPULIBFLAGSASAN = -Xlinker -fsanitize=address -Xlinker -shared
###asan: all.$(TAG)
###asan: all.$(TAG)

# Target: tag-specific build lockfiles
override oldtagsb=`if [ -d $(BUILDDIR) ]; then find $(BUILDDIR) -maxdepth 1 -name '.build.*' ! -name '.build.$(TAG)' -exec echo $(shell pwd)/{} \; ; fi`
$(BUILDDIR)/.build.$(TAG):
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	@if [ "$(oldtagsb)" != "" ]; then echo "Cannot build for tag=$(TAG) as old builds exist for other tags:"; echo "  $(oldtagsb)"; echo "Please run 'make clean' first"; echo "If 'make clean' is not enough: run 'make clean USEBUILDDIR=1 BACKEND=$(BACKEND) FPTYPE=$(FPTYPE)' or 'make cleanall'"; exit 1; fi
	@touch $(BUILDDIR)/.build.$(TAG)

# Apply special build flags only to CrossSectionKernels.o (no fast math, see #117 and #516)
ifeq ($(shell $(CXX) --version | grep ^nvc++),)
$(BUILDDIR)/CrossSectionKernels.o: CXXFLAGS := $(filter-out -ffast-math,$(CXXFLAGS))
$(BUILDDIR)/CrossSectionKernels.o: CXXFLAGS += -fno-fast-math
# the precompiled header was made with -ffast-math: clang refuses it otherwise
$(BUILDDIR)/CrossSectionKernels.o: PCHFLAGS :=
$(BUILDDIR)/CrossSectionKernels.o: GPUFLAGS += $(XCOMPILERFLAG) -fno-fast-math
endif

# Avoid "warning: builtin __has_trivial_... is deprecated; use __is_trivially_... instead" in GPUCC with icx2023 (#592)
ifneq ($(shell $(CXX) --version | egrep '^(Intel)'),)
ifneq ($(GPUCC),)
GPUFLAGS += -Wno-deprecated-builtins
endif
endif

#### Apply special build flags only to CPPProcess.o (-flto)
###$(BUILDDIR)/CPPProcess.o: CXXFLAGS += -flto

# Generic target and build rules: objects from C++ or CUDA/HIP compilation.
# Object files use a plain .o suffix; the TAG lockfile in BUILDDIR prevents silent mixing of
# incompatible backends (different BACKEND, FPTYPE, etc.) in the same directory.
# Use USEBUILDDIR=1 to build for multiple backends simultaneously without cleaning.
ifeq ($(GPUCC),)
$(BUILDDIR)/%%.o : %%.cc *.h ../../backend/$(BACKENDDIR)/*.h ../../backend/common/*.h $(SRC)/*.h $(SRC)/rambo/*.h $(BUILDDIR)/.build.$(TAG) $(PCHFILE)
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	$(CXX) $(CPPFLAGS) $(INCFLAGS) $(CXXFLAGS) $(PCHFLAGS) -c $< -o $@
else
$(BUILDDIR)/%%.o : %%.cc *.h ../../backend/$(BACKENDDIR)/*.h ../../backend/common/*.h $(SRC)/*.h $(SRC)/rambo/*.h $(BUILDDIR)/.build.$(TAG)
	@if [ ! -d $(BUILDDIR) ]; then echo "mkdir -p $(BUILDDIR)"; mkdir -p $(BUILDDIR); fi
	$(GPUCC) $(CPPFLAGS) $(INCFLAGS) $(GPUFLAGS) -c -x $(GPULANGUAGE) $< -o $@
endif

#-------------------------------------------------------------------------------

# Target (and build rules): common (src) library
commonlib : $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so $(PCHFILE)

# The precompiled standard headers (PCH=1), see above. Like the common library
# they are made once, up front, by whoever owns the build.
ifneq ($(PCHFILE),)
ifeq ($(MADMATRIX_COMMONLIB_EXTERNAL),1)
$(PCHFILE): ;
else
$(PCHFILE): ../../backend/common/pch_std.h $(BUILDDIR)/.build.$(TAG)
	@mkdir -p $(LIBDIR)
	$(CXX) $(CPPFLAGS) $(INCFLAGS) $(CXXFLAGS) -x c++-header $< -o $@
endif
endif

# The common library is shared by every P* directory. When several of them are
# built concurrently (SubProcesses/makefile fanning out under 'make -j N'), they
# must not all recurse into src/ and race on the same objects: the dispatcher
# builds the common library once, up front, and sets MADMATRIX_COMMONLIB_EXTERNAL=1
# to say so. Building a P* directory on its own (the default) is unaffected.
ifeq ($(MADMATRIX_COMMONLIB_EXTERNAL),1)
$(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so: ;
else
$(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so: $(SRC)/*.h $(SRC)/*.cc $(BUILDDIR)/.build.$(TAG)
	$(MAKE) -C $(SRC) BACKEND=$(BACKEND) FPTYPE=$(FPTYPE) LIBDIR=$(LIBDIR)
endif

#-------------------------------------------------------------------------------

# Target (and build rules): process shared library (C++ or CUDA/HIP, selected by GPUCC)
ifeq ($(GPUCC),)
$(LIBDIR)/lib$(MADMATRIX_LIB).so: $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so $(objects_lib)
	$(CXX) -shared -o $@ $(objects_lib) $(CXXLIBFLAGSNAME) $(CXXLIBFLAGSRPATH2) -L$(LIBDIR) -l$(MADMATRIX_COMMONLIB) $(CPPBLASLIBFLAGS)
else
$(LIBDIR)/lib$(MADMATRIX_LIB).so: $(LIBDIR)/lib$(MADMATRIX_COMMONLIB).so $(objects_lib)
	$(GPUCC) --shared -o $@ $(objects_lib) $(GPULIBFLAGSNAME) $(GPULIBFLAGSRPATH2) -L$(LIBDIR) -l$(MADMATRIX_COMMONLIB) $(BLASLIBFLAGS)
endif

#-------------------------------------------------------------------------------

# Target: build all targets in all BACKEND modes (each BACKEND mode in a separate build directory)
# Split the bldall target into separate targets to allow parallel 'make -j bldall' builds.
# Pass LIBDIR explicitly so 'make bldall LIBDIR=/path' works end-to-end.
# (GNU make auto-forwards command-line vars via MAKEFLAGS, but being explicit is safer and self-documenting.)
# (LIBDIR is forwarded as given, not resolved, to keep the rpaths loader-relative:
# the sub-make runs in this same directory, so the relative form still resolves.)
_BLDFLAGS = USEBUILDDIR=1 LIBDIR=$(LIBDIR_AS_GIVEN)

bldcuda:
	@echo
	$(MAKE) $(_BLDFLAGS) BACKEND=cuda

bldhip:
	@echo
	$(MAKE) $(_BLDFLAGS) BACKEND=hip

bldnone:
	@echo
	$(MAKE) $(_BLDFLAGS) BACKEND=scalar

bldsse4:
	@echo
	$(MAKE) $(_BLDFLAGS) BACKEND=simd_128

bldavx2:
	@echo
	$(MAKE) $(_BLDFLAGS) BACKEND=simd_256

bld512y:
	@echo
	$(MAKE) $(_BLDFLAGS) BACKEND=avx512y

bld512z:
	@echo
	$(MAKE) $(_BLDFLAGS) BACKEND=simd_512

# The C++/SIMD backends that are worth building on this machine...
ifeq ($(UNAME_P),ppc64le)
  BLDAVXS = scalar simd_128
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
  BLDAVXS = scalar simd_128
else
  BLDAVXS = scalar simd_128 simd_256 avx512y simd_512
endif

# ...plus the GPU backends whose compiler was found. This is the single place
# where the list of 'bldall' backends is defined: SubProcesses/makefile drives
# multi-backend builds through the bldall/bldcommonlib targets below.
BLDBACKENDS = $(BLDAVXS)
ifneq ($(CUDA_HOME),)
  BLDBACKENDS := cuda $(BLDBACKENDS)
endif
ifneq ($(HIP_HOME),)
  BLDBACKENDS := hip $(BLDBACKENDS)
endif

# The bld* targets keep their historical short names, which do not follow the
# backend names, so map them explicitly rather than editing the backend string.
BLDTARGET_scalar   = bldnone
BLDTARGET_simd_128 = bldsse4
BLDTARGET_simd_256 = bldavx2
BLDTARGET_avx512y  = bld512y
BLDTARGET_simd_512 = bld512z
BLDTARGET_cuda     = bldcuda
BLDTARGET_hip      = bldhip

bldavxs: $(foreach backend,$(BLDAVXS),$(BLDTARGET_$(backend)))
bldall: $(foreach backend,$(BLDBACKENDS),$(BLDTARGET_$(backend)))

# Target: the common (src) library in all BACKEND modes. Used by the
# SubProcesses dispatcher, which owns the common library during a 'bldall'
# (see MADMATRIX_COMMONLIB_EXTERNAL above).
.PHONY: bldcommonlib $(addprefix commonlib.,$(BLDBACKENDS))
bldcommonlib: $(addprefix commonlib.,$(BLDBACKENDS))

$(addprefix commonlib.,$(BLDBACKENDS)): commonlib.%%:
	+$(MAKE) $(_BLDFLAGS) BACKEND=$* commonlib

#-------------------------------------------------------------------------------

# Target: clean the builds
.PHONY: clean cleanall cleancommon cleanallcommon

# clean: remove objects and libraries for the selected BACKEND only.
clean:
ifeq ($(USEBUILDDIR),1)
	rm -rf $(BUILDDIR)
ifneq ($(MADMATRIX_COMMONLIB_EXTERNAL),1)
	$(MAKE) -C $(SRC) clean BACKEND=$(BACKEND) FPTYPE=$(FPTYPE) LIBDIR=$(LIBDIR)
endif
else
	rm -f $(BUILDDIR)/.build.* $(BUILDDIR)/*.o
	rm -f $(LIBDIR)/lib$(MADMATRIX_LIB).so
	rm -f $(BACKEND_LOG)
ifneq ($(MADMATRIX_COMMONLIB_EXTERNAL),1)
	$(MAKE) -C $(SRC) clean BACKEND=$(BACKEND) FPTYPE=$(FPTYPE) LIBDIR=$(LIBDIR)
endif
endif

# cleanall: remove objects and libraries for ALL backends and FPTYPEs.
# (NB: MADMATRIX_LIB itself only names one BACKEND/FPTYPE combination -- see
# its own definition above -- so this globs for all of them explicitly,
# rather than just removing $(MADMATRIX_LIB) as 'clean' above does.)
cleanall:
	rm -rf build.*
	rm -f ./.build.* ./*.o
	rm -f $(LIBDIR)/libmadmatrix_%(model)s_$(processid_short)_*.so
ifneq ($(MADMATRIX_COMMONLIB_EXTERNAL),1)
	$(MAKE) -C $(SRC) cleanall LIBDIR=$(LIBDIR)
endif

# Clean only the common (src) part. These are the counterparts of 'commonlib'
# for the SubProcesses dispatcher, which owns the common library while the P*
# directories are cleaned in parallel with MADMATRIX_COMMONLIB_EXTERNAL=1.
cleancommon:
	$(MAKE) -C $(SRC) clean BACKEND=$(BACKEND) FPTYPE=$(FPTYPE) LIBDIR=$(LIBDIR)

cleanallcommon:
	$(MAKE) -C $(SRC) cleanall LIBDIR=$(LIBDIR)

#-------------------------------------------------------------------------------

# Detect backend (to be used in case of 'auto' to give info to the user)
.PHONY: detect-backend
detect-backend:
	@echo "Resolved backend has already been written to $(BACKEND_LOG) at parse time."

#-------------------------------------------------------------------------------
