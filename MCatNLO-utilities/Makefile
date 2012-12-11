# makefile for the aMCatNLO-utilities package
# compile the different packages and copies the libraries 
# in the MCatNLO/lib directory
#

STDHEP_PATH=StdHEP
LIB_STDHEP=$(STDHEP_PATH)/lib/libstdhep.a \
	   $(STDHEP_PATH)/lib/libFmcfio.a

MCATNLO_LIBPATH=MCatNLO/lib/

all: $(LIB_STDHEP)
	cp $(LIB_STDHEP) $(MCATNLO_LIBPATH)

clean:
	cd $(STDHEP_PATH); make clean
	rm -f $(MCATNLO_LIBPATH)/*

$(LIB_STDHEP):
	cd $(STDHEP_PATH); make 
