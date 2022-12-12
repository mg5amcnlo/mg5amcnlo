FC=gfortran
FFLAGS = -O -fPIC
HERE   = $(PWD)
QLDIR  = $(HERE)/ql
FFDIR  = $(HERE)/ff

# -----------------------------------------------------------------------------
# Specify the object files. 

TEST= \
test.o 

.f.o:	
		$(FC) $(F77FLAGS) -c $*.f


# -----------------------------------------------------------------------------

all: test

test: qldir ffdir 		\
	$(TEST)
	$(FC) $(FFLAGS) -L$(QLDIR) -L$(FFDIR) -o $@ \
            $(TEST) -lqcdloop -lff 

ffdir:
	(cd $(FFDIR); make; ranlib libff.a) 

qldir:
	(cd $(QLDIR); make) 


# -----------------------------------------------------------------------------

clean:
	- rm -f *.o *.s *.prj *~ core test

# -----------------------------------------------------------------------------

# DO NOT DELETE


