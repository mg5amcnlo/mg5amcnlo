
LIBRARY       = libgammaUPC.$(libext)
LIBDIR        = ../../../lib/

include ../../make_opts

#FC=gfortran

EXE=test

photonflux=tbessk.o tbessj.o nielsen_generalized_polylog.o ElasticPhotonPhotonFlux.o OpticalGlauber_Geometry.o nintlib.o interpolation.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(photonflux)
	$(call CREATELIB, $@, $^)

$(EXE): test.o $(photonflux)
	$(FC) $(FFLAGS) -o $(EXE) test.o $(photonflux)

test.o: test.f90 ElasticPhotonPhotonFlux.mod OpticalGlauber_Geometry.mod
	$(FC) $(FFLAGS) -c $<

ElasticPhotonPhotonFlux.o ElasticPhotonPhotonFlux.mod: ElasticPhotonPhotonFlux.f90 OpticalGlauber_Geometry.mod nintlib.mod interpolation.mod
	@($(FC) $(FFLAGS) -c $<)
	@(cp ElasticPhotonPhotonFlux.mod $(LIBDIR))

tbessk.o: tbessk.f90
	$(FC) $(FFLAGS) -c $<

tbessj.o: tbessj.f90
	$(FC) $(FFLAGS) -c $<

nielsen_generalized_polylog.o nielsen_generalized_polylog.mod: nielsen_generalized_polylog.f90
	$(FC) $(FFLAGS) -c $<

interpolation.o interpolation.mod: interpolation.f90
	$(FC) $(FFLAGS) -c $<

OpticalGlauber_Geometry.o OpticalGlauber_Geometry.mod: OpticalGlauber_Geometry.f90 nintlib.mod interpolation.mod nielsen_generalized_polylog.mod
	$(FC) $(FFLAGS) -c $<

nintlib.o nintlib.mod: nintlib.f90
	$(FC) $(FFLAGS) -c $<

clean_all:
	$(RM) *.o *~ $(BIN)$(EXE) *.mod $(LIBDIR)$(LIBRARY) $(LIBDIR)ElasticPhotonPhotonFlux.mod

clean:
	$(RM) *.o *~ *.mod $(BIN)$(EXE) $(LIBDIR)$(LIBRARY) $(LIBDIR)ElasticPhotonPhotonFlux.mod
