all: testmp$(EXE) example0$(EXE) example1$(EXE) 

include ../makefile

NUMS0= numerators.o mpnumerators.o 

NUMS = numerators.o mpnumerators.o rambo.o random.o 

mpcheck:
	./testmp

test:
	cd testdir; cp input* ../; cp do ../; cp cdiff ../; cd ../; sh do; \
 cp out* testdir; rm out*; rm input* ; sh cdiff > testoutput; rm cdiff; rm do

testmp$(EXE): $(NUMS) libcts.a testmp.o
	$(FC) $(FFLAGS) -Iincludects -o testmp$(EXE) \
 testmp.f $(NUMS) libcts.a

example0$(EXE): $(NUMS0) libcts.a example0.o
	$(FC) $(FFLAGS) -Iincludects -o example0$(EXE) \
 example0.f $(NUMS0) libcts.a

example1$(EXE): $(NUMS) libcts.a example1.o 
	$(FC) $(FFLAGS) -Iincludects -o example1$(EXE) \
 example1.f $(NUMS) libcts.a

libcts.a:
	cd .. ; $(MAKE)

.SUFFIXES: .f90

.f90.o:
	$(FC) $(FFLAGS) -Iincludects -c $<

clean:
	$(RM) testmp$(EXE) example0$(EXE) example1$(EXE)  *.o testoutput

