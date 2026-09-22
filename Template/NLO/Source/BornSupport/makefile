include ../make_opts
.DEFAULT_GOAL := all
include providers.mk

PICFLAGS ?= -fPIC
PYTHON ?= python3
SUPPORT_FFLAGS = $(FFLAGS) $(PICFLAGS) -ffixed-line-length-none -ffree-line-length-none
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SUPPORT_EXT = dylib
  SUPPORT_SHARED_FLAGS = -dynamiclib -Wl,-install_name,@rpath/libmc_born_support.dylib -Wl,-exported_symbols_list,exports.list
else
  SUPPORT_EXT = so
  SUPPORT_SHARED_FLAGS = -shared -Wl,-z,defs -Wl,--version-script=exports.map -Wl,-soname,libmc_born_support.so
endif
TARGET = ../../lib/libmc_born_support.$(SUPPORT_EXT)
MODULE_TARGETS = ../../lib/mc_born_types.mod ../../lib/mc_born_support.mod
OBJECTS = mc_born_types.o $(DEPENDENCY_OBJECTS) $(PROVIDER_OBJECTS) mc_born_support.o

.PHONY: all clean
all: $(TARGET) $(MODULE_TARGETS)

$(TARGET): $(OBJECTS) makefile exports.map exports.list
	$(FC) $(SUPPORT_SHARED_FLAGS) -o $@ $(OBJECTS)

$(MODULE_TARGETS): $(TARGET)
	cp $(notdir $@) $@

%.o: %.f ../make_opts
	$(FC) $(SUPPORT_FFLAGS) -I. -c $< -o $@

%.o: %.f90 ../make_opts
	$(FC) $(SUPPORT_FFLAGS) -I. -c $< -o $@

$(PROVIDER_OBJECTS) mc_born_support.o: mc_born_types.o

clean:
	rm -f *.o *.mod p*/*.o c*/*.o deps/*.o $(TARGET) ../../lib/mc_born_types.mod ../../lib/mc_born_support.mod
