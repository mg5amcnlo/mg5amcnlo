PYTHIA_PATH:=$(shell python3 pythia_path.py)

-include makefile.inc

MPI_PATH=/usr/local

pythia_interface_for_sudgen.o: pythia_interface_for_sudgen.cc
	gcc $(PYTHIA8FLAGS) -c -I. -I$(PYTHIA8INCLUDE) $^

testgrid: testgrid.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc sudakov.f 
	gfortran -o testgrid testgrid.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc sudakov.f -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudgen: gridsudgen.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	gfortran -o gridsudgen gridsudgen.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudcomb: gridsudcomb.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	gfortran -o gridsudcomb gridsudcomb.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudgen2: gridsudgen2.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	gfortran -o gridsudgen2 gridsudgen2.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudcomb2: gridsudcomb2.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	gfortran -o gridsudcomb2 gridsudcomb2.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudgen_sep: gridsudgen_sep.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	gfortran -o gridsudgen_sep gridsudgen_sep.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudgen_fromlog: gridsudgen_fromlog.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	gfortran -o gridsudgen_fromlog gridsudgen_fromlog.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudgen_MPI: gridsudgen_MPI.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	mpif77 -o gridsudgen_MPI gridsudgen_MPI.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc  -L$(MPI_PATH)/lib/ -lmpi -I$(MPI_PATH)/include/ -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

gridsudgen_clust: gridsudgen_clust.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc 
	gfortran -o gridsudgen_clust gridsudgen_clust.f dfint.f abend.f kerset.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

PDF_test: PDF_test.f pythia_interface_for_sudgen.cc 
	gfortran -o PDF_test PDF_test.f NNPDFDriver.f opendata.f pythia_interface_for_sudgen.cc -I$(PYTHIA8INCLUDE) $(LDFLAGS) -L$(PYTHIA8LIB) -Wl,-rpath $(PYTHIA8LIB) -lpythia8 $(PYTHIA8FLAGS)

clean:
	@rm -f *.o testgrid gridsudgen gridsudgen_MPI

