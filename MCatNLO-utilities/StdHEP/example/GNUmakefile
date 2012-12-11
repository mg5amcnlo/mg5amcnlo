#
# stdhep examples makefile
#

# define STDHEP_DIR and MCFIO_DIR only if they are not already defined
# note that testcsvpdg works only if STDHEP_DIR is a defined environmental variable
# readPDG uses PDG_MASS_TBL
STDHEP_DIR ?= ..
MCFIO_DIR ?= ..

include $(STDHEP_DIR)/src/stdhep_arch

SINC=$(STDHEP_DIR)/src/inc
INCS = -I$(SINC) -I$(STDHEP_DIR)/mcfio/src \
       -I$(SINC)/isajet -I$(SINC)/pythia -I$(SINC)/herwig \
       -I$(SINC)/qq -I$(SINC)/dpm
CPPFLGS = $(KPPFLAGS) $(INCS)

EXAMS =  standalone
RUNLIST = run_standalone

ifdef CRNLIB
  EXAMS += inputExample multipleInputExample qqExample pythiaExampleRead
  EXAMS += readPythia  readPythiaLH
  RUNLIST += run_qq run_io
  ifdef HERWIG_DIR
    EXAMS += herwigExample
    RUNLIST += run_herwig
  endif
  ifdef ISAJET_DIR
    EXAMS += isajetExample
    RUNLIST += run_isajet
  endif
  ifdef PYLIB
    EXAMS += listPythia pythiaExample pythiaLHExample
    RUNLIST += run_pythia
  endif
endif


#------------------------------------------

all:  $(EXAMS)

standalone: listStdHep listStdHepConv readPDG \
            testCInOut testpdg testcsvpdg

madgraph: pythiaMadGraph madToXDR dumpMadGraph

testC: testCInOut

listStdHep: listStdHep$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o listStdHep listStdHep$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a 

listStdHepConv: listStdHepConv$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o listStdHepConv listStdHepConv$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a 

readPDG: readPDG$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o readPDG readPDG$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a 

herwigExample: herwigExample$(SFX).o testhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o herwigExample herwigExample$(SFX).o testhst$(SFX).o \
		$(HERWIG_DIR)/lib/hwudat$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		-L$(HERWIG_DIR)/lib -lherwig -lherdum \
		$(CRNLIB)/libpacklib.a $(LOCAL)

isajetExample: isajetExample$(SFX).o testhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o isajetExample isajetExample$(SFX).o testhst$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(ISAJET_DIR)/libisajet.a \
		$(CRNLIB)/libmathlib.a $(CRNLIB)/libpacklib.a $(LOCAL)

pythiaExample: pythiaExample$(SFX).o testhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o pythiaExample pythiaExample$(SFX).o testhst$(SFX).o \
		$(PYLIB)/pydata.o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(PYLIB)/libpythia.a $(PYLIB)/libpydum.a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(CRNLIB)/libpacklib.a $(LOCAL)

pythiaExampleRead: pythiaExampleRead$(SFX).o testhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o pythiaExampleRead pythiaExampleRead$(SFX).o testhst$(SFX).o \
		$(PYLIB)/pydata.o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(PYLIB)/libpythia.a $(PYLIB)/libpydum.a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(CRNLIB)/libpacklib.a $(LOCAL)

inputExample: inputExample$(SFX).o testhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o inputExample inputExample$(SFX).o testhst$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(CRNLIB)/libpacklib.a $(LOCAL)

multipleInputExample: multipleInputExample$(SFX).o tstfliom$(SFX).o \
                     $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o multipleInputExample multipleInputExample$(SFX).o tstfliom$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(CRNLIB)/libpacklib.a $(LOCAL)

listPythia: listPythia$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o listPythia listPythia$(SFX).o \
		$(PYLIB)/pydata.o \
		$(PYLIB)/libpythia.a $(LOCAL)

qqExample: qqExample$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o qqExample qqExample$(SFX).o \
		$(QQ_DIR)/lib/ranp.o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(QQLIB) $(TYPSCN) \
		$(CRNLIB)/libpacklib.a $(LOCAL)

pythiaLHExample: pythiaLHExample$(SFX).o pythiaLHhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o pythiaLHExample \
		pythiaLHExample$(SFX).o pythiaLHhst$(SFX).o \
		$(PYLIB)/pydata.o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(PYLIB)/libpythia.a $(PYLIB)/libpydum.a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(CRNLIB)/libpacklib.a $(LOCAL)

pythiaMadGraph: pythiaMadGraph$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o pythiaMadGraph \
		pythiaMadGraph$(SFX).o  \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(PYLIB)/libpythia.a $(PYLIB)/libpydum.a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a 

madToXDR: madToXDR$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o madToXDR madToXDR$(SFX).o  \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a 

dumpMadGraph: dumpMadGraph$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o dumpMadGraph dumpMadGraph$(SFX).o  \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a 

readPythia: readPythia$(SFX).o testhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o readPythia readPythia$(SFX).o testhst$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(CRNLIB)/libpacklib.a $(LOCAL)

readPythiaLH: readPythiaLH$(SFX).o pythiaLHhst$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o readPythiaLH readPythiaLH$(SFX).o pythiaLHhst$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a \
		$(CRNLIB)/libpacklib.a $(LOCAL)

testCInOut: testCInOut$(SFX).o $(STDHEP_DIR)/lib/libstdhepC$(SFX).a
	$(CC) $(CFLAGS) -g -o testCInOut testCInOut$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhepC$(SFX).a \
		$(MCFIO_DIR)/lib/libFmcfio$(SFX).a $(LOCAL)

testpdg:     testpdgtran$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o $@  testpdgtran$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a

testcsvpdg:     testcsvpdg$(SFX).o $(STDHEP_DIR)/lib/libstdhep$(SFX).a
	$(F77) $(FFLAGS) -o $@  \
	        testcsvpdg$(SFX).o \
		$(STDHEP_DIR)/lib/libstdhep$(SFX).a

clean:
	rm -f *.o *.f *.map

realclean:
	rm -f listStdHep listStdHepConv readPDG herwigExample isajetExample pythiaExample
	rm -f inputExample multipleInputExample testCInOut
	rm -f testpdg madToXDR qqExample listPythia pythiaMadGraph pythiaExampleRead
	rm -f pythiaLHExample readPythia readPythiaLH dumpMadGraph testcsvpdg
	rm -f *.bak *.o *.f *.map
	rm -f log.*

#------------------------------------------
# run examples

##run: run_list run_gen run_misc run_io
run: $(RUNLIST)

run_gen: run_herwig run_isajet run_pythia  run_qq

run_standalone: listStdHep listStdHepConv readPDG \
                testCInOut testpdg testcsvpdg
	./listStdHep.sh
	./listStdHepConv.sh
	./testCInOut.sh
	./testpdgtran.sh
	./readPDG.sh
	
run_broken: testcsvpdg
	./testcsvpdg.sh

run_list: listStdHep listStdHepConv  readPDG
	./listStdHep.sh
	./listStdHepConv.sh
	./readPDG.sh

run_pythia:  listPythia pythiaExample  pythiaLHExample readPythiaLH readPythia
	./listPythia.sh
	./pythiaExample.sh
	./pythiaLHExample.sh
	./readPythia.sh
	./readPythiaLH.sh

run_isajet:  isajetExample 
	./isajetExample.sh

run_herwig: herwigExample 
	./herwigExample.sh

run_qq: qqExample 
	./qqExample.sh

run_misc: testCInOut testpdg testcsvpdg
	./testCInOut.sh
	./testcsvpdg.sh
	./testpdgtran.sh

runC: testCInOut
	./testCInOut.sh

run_io: inputExample multipleInputExample run_gen
	./inputExample.sh
	./multipleInputExample.sh

#------------------------------------------

.SUFFIXES : 			# cancel implicit rules
.SUFFIXES : .o .c .F .inc .h

%.o: %.F
	$(F77) $(INCS) $(FFLAGS) -c $*.F

%_g.o: %.F
	$(F77) $(INCS) $(FFLAGS) -c -o $@ $*.F

%.o: %.c
	$(CC) $(INCS) $(CFLAGS) -c $<


%_g.o: %.c
	$(CC) $(INCS) $(CFLAGS) -c -o $@ $<
