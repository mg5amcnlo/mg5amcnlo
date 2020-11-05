-include makefile.inc

MPI_PATH=/usr/local

pythia_interface_for_sudgen.o: pythia_interface_for_sudgen.cc
	gcc $(PYTHIA8FLAGS) -c -I. -I$(PYTHIA8INCLUDE) $^

dire_interface_for_sudgen.o: dire_interface_for_sudgen.cc
	gcc $(PYTHIA8FLAGS) -c -I. -I$(PYTHIA8INCLUDE) -I$(DIREINCLUDE) $^

testgrid: testgrid.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc sudakov.f 
	gfortran -o testgrid testgrid.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc sudakov.f -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

gridsudgen: gridsudgen.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc 
	gfortran -o gridsudgen gridsudgen.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

gridsudgen_sep: gridsudgen_sep.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc 
	gfortran -o gridsudgen_sep gridsudgen_sep.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

gridsudgen_fromlog: gridsudgen_fromlog.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc 
	gfortran -o gridsudgen_fromlog gridsudgen_fromlog.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

gridsudgen_MPI: gridsudgen_MPI.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc 
	mpif77 -o gridsudgen_MPI gridsudgen_MPI.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc  -L$(MPI_PATH)/lib/ -lmpi -I$(MPI_PATH)/include/ -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

PDF_test: PDF_test.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc 
	gfortran -o PDF_test PDF_test.f NNPDFDriver.f opendata.f pythia_interface_for_sudgen.cc dire_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) -I$(PYTHIA8INCLUDE)/../../../Dire/dire-src/include $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8  -L$(DIRELIB) -Wl,-rpath $(DIRELIB) -ldire $(PYTHIA8FLAGS)

clean:
	@rm -f *.o testgrid gridsudgen gridsudgen_MPI

