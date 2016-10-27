HEPMC  = $(HEPMCPATH)
INCDIR = ../include

-include ../../Source/make_opts

DEBUG=
FF=$(FC)   $(FFLAGS)   $(DEBUG)
CPP=$(CXX) $(CXXFLAGS) $(DEBUG)
CC=$(CXX)  $(CFLAGS)   $(DEBUG)
CXXFLAGS += $(shell $(HWPPPATH)/bin/herwig-config --cppflags )

all : HepMCFortran.so

%.o: ../srcCommon/%.f 
	$(FF) -c $< -fPIC -I$(INCDIR) -o $@	
%.o: %.f
	$(FF) -c $< -fPIC -I$(INCDIR) -o $@   
	
%.o: ../srcCommon/%.cc
	$(CPP) -c $< -fPIC $(INCLOPTION) -I$(INCDIR) -o $@
%.o: %.cc
	$(CPP) -c $< -fPIC $(INCLOPTION) -I$(INCDIR) -o $@

HepMCFortran.so : HepMCFortran.cc HepMCFortran.h $(FORFILE)
	$(CPP) -shared -fPIC $(CXXFLAGS) $(LDFLAGS) $(EXTRAPATHS) $(EXTRALIBS) \
	-I$(HEPMC)/include -L$(HEPMC)/lib -lHepMC -lHepMCfio $< -o $@ $(FORFILE)

clean:
	rm -f *.o *.so
