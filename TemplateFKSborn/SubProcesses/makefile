F77   = gfortran
#FFLAGS= -O -ffixed-line-length-132
#FFLAGS = -g -ffixed-line-length-132 -Wall -fno-automatic
FFLAGS = -O -ffixed-line-length-132 -fno-automatic
LIBDIR = ../../lib/
BINDIR = ../../bin/
PROG   = madevent
LINKLIBS =  -L../../lib/ -ldhelas3 -ldsample_fks -lmodel -lgeneric -lpdf
LIBS = $(LIBDIR)libdhelas3.a $(LIBDIR)libdsample_fks.a $(LIBDIR)libgeneric.a $(LIBDIR)libpdf.a $(LIBDIR)libmodel.a


PROCESS= driver.o myamp.o auto_dsig.o genps_fks.o matrix.o unwgt.o setcuts.o cuts.o cluster.o reweight.o initcluster.o addmothers.o setscales.o fks_singular.o born.o sborn_sf.o $(patsubst %.f,%.o,$(wildcard b_sf_???.f)) fks_Sij.o

SYMMETRY = myamp.o genps_fks.o matrix.o  initcluster.o setscales.o reweight.o fks_singular.o born.o sborn_sf.o $(patsubst %.f,%.o,$(wildcard b_sf_???.f)) fks_Sij.o

TEST = myamp.o genps_fks.o matrix.o  initcluster.o setscales.o reweight.o fks_singular.o born.o sborn_sf.o $(patsubst %.f,%.o,$(wildcard b_sf_???.f)) fks_Sij.o trapfpe.o ktclusdble.o
# $B$ obj_for_MW $E$ !this is a tag for MadWeight. Don't edit this line

%.o : %.f
	$(F77) $(FFLAGS) -c $<
%.o: %.c
	$(CC) -g -c $^

# $B$ main_make $B$ !this is a tag for MadWeight. Don't edit this line
$(PROG): $(PROCESS)  makefile $(LIBS)
	$(F77) $(FFLAGS) -o $(PROG) $(PROCESS) $(LINKLIBS)
# $E$ main_make $E$ !this is a tag for MadWeight. Don't edit this line

symmetry.dat: mg.sym gensym
	./gensym

gensym: symmetry_fks.o genps.inc setcuts.o cuts.o cluster.o $(SYMMETRY) configs.inc  
	$(F77) $(FFLAGS) -o gensym symmetry_fks.o setcuts.o cuts.o cluster.o $(SYMMETRY) $(LINKLIBS)

test: symmetry_fks_test.o genps.inc setcuts.o cuts.o cluster.o $(TEST) configs.inc  
	$(F77) $(FFLAGS) -o test symmetry_fks_test.o setcuts.o cuts.o cluster.o $(TEST) $(LINKLIBS)

driver.f: genps.inc 

genps.inc: nexternal.inc
	touch genps.inc

symmetry_fks.o: genps.inc configs.inc
symmetry_fks_test.o: genps.inc configs.inc
genps.o: genps.inc configs.inc 
cuts.o: genps.inc pmass.inc
invarients.o: genps.inc 
myamp.o: props.inc genps.inc
reweight.o: message.inc sudakov.inc
cluster.o: cluster.inc genps.inc
addmothers.o: genps.inc symswap.inc message.inc
unwgt.o: genps.inc symswap.inc message.inc
