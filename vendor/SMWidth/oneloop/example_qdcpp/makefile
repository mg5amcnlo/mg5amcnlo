OLODIR = ../src
QDHOME = $(HOME)/fortran/qd-2.3.13
FC=gfortran
CXX = g++
FCFLAGS = -O2 -ffree-form
FCLIBS = -L/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6 -L/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -L/usr/lib/gcc/x86_64-redhat-linux6E/4.4.6/../../.. -lrt -lgfortranbegin -lgfortran -lm

QDLIB = \
	$(QDHOME)/fortran/.libs/libqdmod.a \
	$(QDHOME)/fortran/.libs/libqd_f_main.a \
	$(QDHOME)/src/.libs/libqd.a

f_test.exe: avh_olo.f90 avh_olo.o f_test.o
	$(CXX) -o $@ avh_olo.o f_test.o $(QDLIB) $(FCLIBS)

%.o: %.f
	$(FC) -c -I$(QDHOME)/fortran $(FCFLAGS) $<
%.o: %.f90
	$(FC) -c -I$(QDHOME)/fortran $(FCFLAGS) $<

avh_olo.f90: 
	$(OLODIR)/avh_olo.py -dpkind="kind(1d0)" -qdtype="qdcpp"

clean:
	 rm -f *.exe *.o *.mod avh_olo.f90
