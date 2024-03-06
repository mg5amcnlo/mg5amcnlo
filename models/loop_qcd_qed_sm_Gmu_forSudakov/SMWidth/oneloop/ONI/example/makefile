FC=gfortran
FFLAGS = 

WITHLT = yes
#WITHLT = no

LTHOME = $(HOME)/loopcal/LoopTools-2.6/build
LTVERSIONge26 = yes
#LTVERSIONge26 = no
LTLIB = ooptools

CUBAHOME = $(HOME)/Cuba-2.0
CUBALIB = cuba

OLOHOME = ../..
OLOLIB = avh_olo

ONIHOME = ..
ONILIB = avh_oni

########################################################################
FILE = example

all: edit a.out

.f.o: 
	$(FC) $(FFLAGS) -I$(OLOHOME) -c $*.f -o $*.o 

OBJECTS = \
          ./$(FILE).o

ifeq ($(WITHLT),yes)

a.out: $(OBJECTS) 
	$(FC) $(FFLAGS) -I$(OLOHOME) -o $(FILE).exe $(OBJECTS) \
	-L$(LTHOME) -l$(LTLIB) \
	-L$(OLOHOME) -l$(OLOLIB) \
	-L$(ONIHOME) -l$(ONILIB) \
	-L$(CUBAHOME) -l$(CUBALIB) -lm
ifeq ($(LTVERSIONge26),yes)
edit:
	 sed -i -e's/^!LTno//' $(FILE).f
	 sed -i -e's/!LTno/!LTyes/' $(FILE).f
	 sed -i -e's/ffini/ltini/' $(FILE).f
else
edit:
	 sed -i -e's/^!LTno//' $(FILE).f
	 sed -i -e's/!LTno/!LTyes/' $(FILE).f
	 sed -i -e's/ltini/ffini/' $(FILE).f
endif

else

a.out: $(OBJECTS) 
	$(FC) $(FFLAGS) -o $(FILE).exe $(OBJECTS) \
	-L$(OLOHOME) -l$(OLOLIB) \
	-L$(ONIHOME) -l$(ONILIB) \
	-L$(CUBAHOME) -l$(CUBALIB) -lm
edit:
	 sed -i -e's/^\( .*\) !LTyes/!LTno\1 !LTno/' $(FILE).f

endif


clean:;	rm example.o example.exe
