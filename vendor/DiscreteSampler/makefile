
PROG = check

FC   = gfortran
DOTO = %.o
DOTF = %.f

# I wish I would do that but Olivier would kill me
#FFLAGS = -ffree-form -ffree-line-length-300
FFLAGS = -g 

SOURCE = DiscreteSampler.o StringCast.o tester.o

all : $(PROG)

#########
# Additional rules to make sure the module dependence is correctly accounted for
#########

%: %.o
	$(FC) $(FFLAGS) -o $@ $^

DiscreteSampler.o : StringCast.o
tester.o : DiscreteSampler.o

########

%(DOTO): $(DOTF)
	$(FC) $(FFLAGS) -c $<

$(PROG): $(SOURCE) makefile
	$(FC) $(FFLAGS) -o $(PROG) $(SOURCE)

clean:
	@rm -f *.o
	@rm -f *.mod
