ifeq ($(wildcard ../make_opts), ../make_opts)
  include ../make_opts
else
  FFLAGS+= -ffixed-line-length-132
  FC=gfortran
endif

FFLAGS+= -O3

LIBDIR=../../lib/
LIBRARY=librunning.$(libext)

MODEL = matrix_exponential.o c8lib.o r8lib.o


all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(MODEL)
	$(call CREATELIB, $@, $^)

$(LIBRARY): $(MODEL)
	ar cru $(LIBRARY) $(MODEL)
	ranlib $(LIBRARY)