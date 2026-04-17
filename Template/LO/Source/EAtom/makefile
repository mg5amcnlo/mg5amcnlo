
include ../make_opts

LIBRARY = libeatom.$(libext)
LIBRARY_SHARED  = libdeatom.$(dylibext)
LIBDIR  = ../../lib/

OBJ=interpolation.o spherical_harmonic.o C12_gftilde.o W186_gftilde.o Pb208_gftilde.o atomic_electron_wavefunctions.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(OBJ)
	$(call CREATELIB, $@, $^)

# Shared library (optional)
$(LIBDIR)$(LIBRARY_SHARED): $(OBJ)
	$(FC) -shared -o $@ $^ $(LDFLAGS)

shared: $(LIBDIR)$(LIBRARY_SHARED)

spherical_harmonic.o spherical_harmonic.mod: spherical_harmonic.f90
	$(FC) $(FFLAGS) -c $<

atomic_electron_wavefunctions.o atomic_electron_wavefunctions.mod: atomic_electron_wavefunctions.f90 interpolation.mod spherical_harmonic.mod C12_gftilde.mod W186_gftilde.mod Pb208_gftilde.mod
	$(FC) $(FFLAGS) -c $<

C12_gftilde.o C12_gftilde.mod: C12_gftilde.f90
	$(FC) $(FFLAGS) -c $<

W186_gftilde.o W186_gftilde.mod: W186_gftilde.f90
	$(FC) $(FFLAGS) -c $<

Pb208_gftilde.o Pb208_gftilde.mod: Pb208_gftilde.f90
	$(FC) $(FFLAGS) -c $<

interpolation.o interpolation.mod: interpolation.f90
	$(FC) $(FFLAGS) -c $<

clean:
	$(RM) *~ *.o *.mod $(LIBDIR)$(LIBRARY)
