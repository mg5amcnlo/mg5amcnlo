FC=gfortran
FFLAGS =

FILE = example.f

#WITHLT = yes
WITHLT = no

LTHOME = $(HOME)/loopcal/LoopTools-2.6/build
LTLIB = ooptools
LTVSNge26 = yes
#LTVSNge26 = no

OLOHOME = ../
OLOLIB = avh_olo

#######################################################################
all: copy edit compile rmtmp

copy:
	cp $(FILE) tmp_$(FILE)

edit:
	../src/avh_pc_exe.py "case" "LT" "$(WITHLT)" "tmp_$(FILE)"
	../src/avh_pc_exe.py "case" "LTVSNge26" "$(WITHLT)$(LTVSNge26)" "tmp_$(FILE)"

ifeq ($(WITHLT),yes)
compile: 
	$(FC) $(FFLAGS) -I$(OLOHOME) -o example.exe tmp_$(FILE) \
	-L$(LTHOME) -l$(LTLIB) \
	-L$(OLOHOME) -l$(OLOLIB)
else
compile: 
	$(FC) $(FFLAGS) -I$(OLOHOME) -o example.exe tmp_$(FILE) \
	-L$(OLOHOME) -l$(OLOLIB)
endif

rmtmp:
	rm -f tmp_$(FILE)

clean:
	rm -f *.exe *.mod
