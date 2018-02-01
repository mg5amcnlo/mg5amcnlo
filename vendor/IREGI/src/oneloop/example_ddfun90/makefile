FC=gfortran
FFLAGS = -O

DDF90MOD = $(HOME)/fortran/ddfun90/
DDF90LIB = $(HOME)/fortran/ddfun90/
DDF90 = ddfun90

OLOMOD = ../
OLOLIB = ../
OLO = avh_olo

FILE = example.f

WITHLT = yes
#WITHLT = no

LTHOME = $(HOME)/loopcal/LoopTools-2.6/build
LTLIB = ooptools
LTVSNge26 = yes
#LTVSNge26 = no

#######################################################################
all: copy edit compile rmtmp

copy:
	cp $(FILE) tmp_$(FILE)

edit:
	../src/avh_pc_exe.py "case" "LT" "$(WITHLT)" "tmp_$(FILE)"
	../src/avh_pc_exe.py "case" "LTVSNge26" "$(WITHLT)$(LTVSNge26)" "tmp_$(FILE)"

ifeq ($(WITHLT),yes)
compile: 
	$(FC) $(FFLAGS) -I$(DDF90MOD) -I$(OLOMOD) -o example.exe tmp_$(FILE) \
	-L$(DDF90LIB) -l$(DDF90) -L$(OLOLIB) -l$(OLO) -L$(LTHOME) -l$(LTLIB)
else
compile: 
	$(FC) $(FFLAGS) -I$(DDF90MOD) -I$(OLOMOD) -o example.exe tmp_$(FILE) \
	-L$(DDF90LIB) -l$(DDF90) -L$(OLOLIB) -l$(OLO)
endif

rmtmp:
	 rm -f tmp_$(FILE)

clean:
	 rm -f *.exe *.mod
