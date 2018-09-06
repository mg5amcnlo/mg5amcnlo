WORK=/home/prestel/work/2018/MG5/PY8meetsMG5aMC
PYTHIA8INCLUDE=$(WORK)/Pythia/Pythia/pythia-src/include
PYTHIA8LIB=$(WORK)/Pythia/Pythia/pythia-src/lib
DIRELIB=$(WORK)/Pythia/Pythia/pythia-src/../../Dire/dire-src/lib
PYTHIA8FLAGS=-lstdc++ -lz -ldl -fPIC

ps_interface.o: ps_interface.cc
	gcc $(PYTHIA8FLAGS) -c -I. -I$(PYTHIA8INCLUDE) -I$(DIREINCLUDE) $^

testgrid: testgrid.f dfint.f kerset.f ps_interface.cc sudakov.f 
	gfortran -o testgrid testgrid.f dfint.f kerset.f ps_interface.cc sudakov.f -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

gridsudgen: gridsudgen.f dfint.f kerset.f ps_interface.cc 
	gfortran -o gridsudgen gridsudgen.f dfint.f kerset.f ps_interface.cc -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

clean:
	@rm -f *.o testgrid  gridsudgen

