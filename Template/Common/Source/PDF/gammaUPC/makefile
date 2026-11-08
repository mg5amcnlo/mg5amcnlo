
LIBRARY       = libgammaUPC.$(libext)
LIBDIR        = ../../../lib/

include ../../make_opts

#FC=gfortran

EXE=test

photonflux=tbessk.o tbessj.o elliptic_integral.o nielsen_generalized_polylog.o photoabsorption.o ElasticPhotonPhotonFlux.o OpticalGlauber_Geometry.o nintlib.o interpolation.o fmin_module.o photonpdfsquare.o

all: $(LIBDIR)$(LIBRARY)

# A module and its .mod file come from one compile of its
# source: only the .o is a target, so a parallel make never
# runs that compile twice (the two would race on the .mod).
# Modules a file USEs are prerequisites through their .o.

$(LIBDIR)$(LIBRARY): $(photonflux)
	$(call CREATELIB, $@, $^)

$(EXE): test.o $(photonflux)
	$(FC) $(FFLAGS) -o $(EXE) test.o $(photonflux)

test.o: test.f90 ElasticPhotonPhotonFlux.o OpticalGlauber_Geometry.o
	$(FC) $(FFLAGS) -c $<

ElasticPhotonPhotonFlux.o: ElasticPhotonPhotonFlux.f90 OpticalGlauber_Geometry.o nintlib.o interpolation.o fmin_module.o photoabsorption.o
	$(FC) $(FFLAGS) -c $<

photoabsorption.o: photoabsorption.f90
	$(FC) $(FFLAGS) -c $<

photonpdfsquare.o : photonpdfsquare.f ElasticPhotonPhotonFlux.o
	$(FC) $(FFLAGS) -c $<

tbessk.o: tbessk.f90
	$(FC) $(FFLAGS) -c $<

tbessj.o: tbessj.f90
	$(FC) $(FFLAGS) -c $<

elliptic_integral.o: elliptic_integral.f90
	$(FC) $(FFLAGS) -c $<

nielsen_generalized_polylog.o: nielsen_generalized_polylog.f90
	$(FC) $(FFLAGS) -c $<

interpolation.o: interpolation.f90
	$(FC) $(FFLAGS) -c $<

fmin_module.o: fmin_module.f90
	$(FC) $(FFLAGS) -c $<

OpticalGlauber_Geometry.o: OpticalGlauber_Geometry.f90 nintlib.o interpolation.o nielsen_generalized_polylog.o
	$(FC) $(FFLAGS) -c $<

nintlib.o: nintlib.f90
	$(FC) $(FFLAGS) -c $<

clean_all:
	$(RM) *.o *~ $(BIN)$(EXE) *.mod $(LIBDIR)$(LIBRARY)

clean:
	$(RM) *.o *~ *.mod $(BIN)$(EXE) $(LIBDIR)$(LIBRARY)
