include ../make_opts

LIBRARY = libeatom.$(libext)
LIBRARY_SHARED  = libdeatom.$(dylibext)
LIBDIR  = ../../lib/

OBJ=interpolation.o spherical_harmonic.o C12_gftilde.o W186_gftilde.o Pb208_gftilde.o atomic_electron_wavefunctions.o eatom_external.o eatom_ps.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(OBJ)
	$(call CREATELIB, $@, $^)

# Shared library (optional)
$(LIBDIR)$(LIBRARY_SHARED): $(OBJ)
	$(FC) -shared -o $@ $^ $(LDFLAGS)

shared: $(LIBDIR)$(LIBRARY_SHARED)

# One target per rule: the .mod files are side products of the .o files, and
# module users depend on the .o of the module (safe with make -j).
%.o: %.f90
	$(FC) $(FFLAGS) -c $<

atomic_electron_wavefunctions.o: interpolation.o spherical_harmonic.o C12_gftilde.o W186_gftilde.o Pb208_gftilde.o
eatom_external.o: atomic_electron_wavefunctions.o
eatom_ps.o: atomic_electron_wavefunctions.o eatom_external.o

clean:
	$(RM) *~ *.o *.mod $(LIBDIR)$(LIBRARY)
