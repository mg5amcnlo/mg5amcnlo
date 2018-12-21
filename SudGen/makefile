-include makefile.inc

MPI_PATH=/usr/local

ps_interface.o: ps_interface.cc
	gcc $(PYTHIA8FLAGS) -c -I. -I$(PYTHIA8INCLUDE) -I$(DIREINCLUDE) $^

testgrid: testgrid.f dfint.f kerset.f ps_interface.cc sudakov.f 
	gfortran -o testgrid testgrid.f dfint.f kerset.f ps_interface.cc sudakov.f -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

gridsudgen: gridsudgen.f dfint.f kerset.f ps_interface.cc 
	gfortran -o gridsudgen gridsudgen.f dfint.f kerset.f ps_interface.cc -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

gridsudgen_MPI: gridsudgen_MPI.f dfint.f kerset.f ps_interface.cc 
	mpif77 -o gridsudgen_MPI gridsudgen_MPI.f dfint.f kerset.f ps_interface.cc  -L$(MPI_PATH)/lib/ -lmpi -I$(MPI_PATH)/include/ -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

clean:
	@rm -f *.o testgrid gridsudgen gridsudgen_MPI

