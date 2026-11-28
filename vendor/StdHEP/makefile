#
# main StdHep makefile
#
# to build debug libraries: make DEBUG="-g" all
# to build shared libraries: make BUILD_SHARED="true" all
#
MAKE=make

ifdef DEBUG
  DMAKE = "DEBUG=$(DEBUG)"
endif

STDHEP_DIR = .
FFLAGS += -std=legacy
CFLAGS += -Wno-implicit-function-declaration

# The cern libraries are needed to build the Phase and Space executables.
# There is no point in building the libraries unless the executables will 
# be built.
ifdef CRNLIB
   BUILD_DISPLAY = spin util display
endif
#------------------------------------------

all: binlib mcfio stdhep $(BUILD_DISPLAY)

binlib:
	-test -d $(STDHEP_DIR)/bin || mkdir -p $(STDHEP_DIR)/bin
	-test -d $(STDHEP_DIR)/lib || mkdir -p $(STDHEP_DIR)/lib

mcfio:
	($(MAKE) -C mcfio/src $(DMAKE) all) > log.mcfio.$$$$ 2>&1

stdhep:
	($(MAKE) -C src/stdhep $(DMAKE) all) > log.stdhep.$$$$ 2>&1

spin:
	($(MAKE) -C src/display/spin $(DMAKE) all) > log.spin.$$$$ 2>&1

util:
	($(MAKE) -C src/display/util $(DMAKE) all) > log.util.$$$$ 2>&1

display:
	($(MAKE) -C src/display $(DMAKE) all) > log.display.$$$$ 2>&1

clean:
	rm -f log.*
	$(MAKE) -C mcfio/src clean
	$(MAKE) -C src/stdhep $(DMAKE) clean
	$(MAKE) -C src/display $(DMAKE) clean
	$(MAKE) -C src/display/spin $(DMAKE) clean
	$(MAKE) -C src/display/util $(DMAKE) clean

realclean:
	rm -f log.*
	$(MAKE) -C mcfio/src realclean
	$(MAKE) -C src/stdhep realclean
	$(MAKE) -C src/display realclean
	$(MAKE) -C src/display/spin realclean
	$(MAKE) -C src/display/util realclean


#------------------------------------------

.PHONY: all realclean stdhep spin util display mcfio
