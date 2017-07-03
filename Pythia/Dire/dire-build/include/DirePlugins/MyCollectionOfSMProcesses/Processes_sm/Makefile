CXX       = g++
LIBDIR    = ../lib
INCDIR    = ../include
MODELNAME = sm
PROCLIB   = PY8MEs_$(MODELNAME)
CXXFLAGS  = -O -I$(INCDIR) -I.
LIBFLAGS  = -L$(LIBDIR) -l$(PROCLIB)

main      = check

define CREATELIB
$(AR) cru $(1) $(2)
ranlib $(1)
endef

all: $(main)

objects = HelAmps_$(MODELNAME).o Parameters_$(MODELNAME).o \
	PY8MEs.o rambo.o read_slha.o \
	$(patsubst %.cc,%.o,$(wildcard PY8MEs_R*_P*.cc)) \

$(main): check_sa.o $(LIBDIR)/lib$(PROCLIB).a
	$(CXX) -o $@ check_sa.o $(LIBFLAGS)

$(LIBDIR)/lib$(PROCLIB).a: $(objects)
	$(call CREATELIB, $@, $^)

modellib : $(LIBDIR)/lib$(PROCLIB).a

.PHONY: clean

clean:
	rm -f $(main)
	rm -f $(objects)

