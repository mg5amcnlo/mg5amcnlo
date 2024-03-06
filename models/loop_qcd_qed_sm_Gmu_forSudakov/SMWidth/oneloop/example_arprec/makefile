OLODIR = ../src
MPHOME = $(HOME)/fortran/arprec/arprec-2.2.9/fortran
FC=gfortran
CXX = g++
FCFLAGS = -O2 -ffree-form
FCLIBS = -L/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6 -L/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -L/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6/../../.. -lgfortranbegin -lgfortran -lm

ARPLIB = \
	$(MPHOME)/libarprec_f_main.a \
	$(MPHOME)/libarprecmod.a \
	$(MPHOME)/../src/libarprec.a

f_test.exe: avh_olo.f90 avh_olo.o f_test.o
	$(CXX) -o $@ avh_olo.o f_test.o $(ARPLIB) $(FCLIBS)

%.o: %.f
	$(FC) -c -I$(MPHOME) $(FCFLAGS) $<
%.o: %.f90
	$(FC) -c -I$(MPHOME) $(FCFLAGS) $<

avh_olo.f90: 
	$(OLODIR)/avh_olo.py -dpkind="kind(1d0)" -mptype="arprec"

clean:
	 rm -f *.exe *.o *.mod avh_olo.f90
