STDLIB = -lc++
STDLIB_FLAG = -stdlib=libc++
MACFLAG = -mmacosx-version-min=10.7
DEFAULT_F_COMPILER = gfortran
DEFAULT_F2PY_COMPILER = f2py
DEFAULT_CPP_COMPILER = g++
#end_of_make_opts_variables

# Rest of the makefile
ifeq ($(origin FFLAGS),undefined)
  #FFLAGS= -O -w
  FFLAGS = -O -fno-automatic
  #FFLAGS+= -g -fbounds-check -ffpe-trap=invalid,zero,overflow,underflow,denormal -Wall
endif

# REMOVE MACFLAG IF NOT ON MAC 
UNAME := $(shell uname -s)
ifneq ($(UNAME), Darwin)
    MACFLAG=  
endif


ifeq ($(origin CXXFLAGS),undefined)
  #CXXFLAGS= -g 
  CXXFLAGS = -O $(STDLIB_FLAG) $(MACFLAG)
endif

ifeq ($(origin CFLAGS),undefined)
  #CFLAGS= -g 
  CFLAGS = -O  $(STDLIB_FLAG) $(MACFLAG)
endif

# REMOVE MACFLAG IF IN F2PY mode
ifdef f2pymode
    MACFLAG= 
endif

# Increase the number of allowed charcters in a Fortran line
FFLAGS+= -ffixed-line-length-132

# Set FC unless it's defined by an environment variable
ifeq ($(origin FC),default)
#  FC=g77
  FC=$(DEFAULT_F_COMPILER)
endif
# Set F2PY unless it's defined by an environment variable
ifeq ($(origin F2PY),undefined)
  F2PY=$(DEFAULT_F2PY_COMPILER)
endif

# Set CXX unless it's defined by an environment variable
ifeq ($(origin CXX),default)
  CXX=$(DEFAULT_CPP_COMPILER)
endif

UNAME := $(shell uname -s)
ifeq ($(origin LDFLAGS), undefined)
     LDFLAGS=$(STDLIB) $(MACFLAG)
endif
# Options: dynamic, lhapdf

UNAME := $(shell uname)

# Option dynamic

ifdef dynamic
  ifeq ($(UNAME), Darwin)
    libext=dylib
    FFLAGS+= -fno-common
    LDFLAGS += -bundle
    define CREATELIB
      $(FC) -dynamiclib -undefined dynamic_lookup -o $(1) $(2)
    endef
  else
    libext=so
    FFLAGS+= -fPIC
    LDFLAGS += -shared
    define CREATELIB
      $(FC) $(FFLAGS) $(LDFLAGS) -o $(1) $(2)
    endef
  endif
else
  libext=a
  define CREATELIB
    $(AR) cru $(1) $(2)
    ranlib $(1)
  endef
endif

# Option lhapdf

ifneq ($(lhapdf),)
  CXXFLAGS += $(shell $(lhapdf) --cppflags)
  alfas_functions=alfas_functions_lhapdf
  llhapdf = $(shell $(lhapdf) --libs)
  reweight_xsec_events_pdf_dummy=
else
  alfas_functions=alfas_functions
  llhapdf=
  reweight_xsec_events_pdf_dummy=reweight_xsec_events_pdf_dummy.o
endif

# Option APPLGrid
ifneq ($(applgrid),)
  APPLLIBS=$(shell applgrid-config --ldcflags) $(shell amcfast-config --ldflags)
  applgrid_interface=appl_interface.o
else
  APPLLIBS=
  applgrid_interface=appl_interface_dummy.o
endif

# Madloop
ifdef madloop
  ifeq (,$(wildcard parton_lum_0.f)) # this is to skip ML for LOonly
   libmadloop=libMadLoop.a
   libcuttools=-lcts %(link_tir_libs)s
  endif 
else
  libmadloop=
  libcuttools=
endif

# OLP (virtuals) necessary link
libOLP=
