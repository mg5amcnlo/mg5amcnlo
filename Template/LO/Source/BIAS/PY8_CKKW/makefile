BIASDIR   = ../
SOURCEDIR = ../..
include $(SOURCEDIR)/make_opts

include ./bias_dependencies

all: PY8_CKKW 

clean:
	$(RM) *.o $(BIASLIBDIR)$(BIASLIBRARY)

INCLUDES = $(BIASDIR)/bias.inc \
	 	   $(SOURCEDIR)/maxparticles.inc \
	 	   $(SOURCEDIR)/nexternal.inc \
		   $(SOURCEDIR)/run_config.inc \
		   $(SOURCEDIR)/lhe_event_infos.inc \
		   $(SOURCEDIR)/run.inc \
		   $(SOURCEDIR)/cuts.inc

#
# Compilation of the module ptj_bias
#
PY8_CKKW.o: PY8_CKKW.f $(INCLUDES) 
	$(FC) $(FFLAGS) $(LDFLAGS) -c -o PY8_CKKW.o PY8_CKKW.f

py8_mg5amc_bias_interface.o: py8_mg5amc_bias_interface.cc
	$(DEFAULT_CPP_COMPILER) $(CXXFLAGS) $(PY8CXXFLAGS) $(LDFLAGS) $(PY8LIBS) -c -o py8_mg5amc_bias_interface.o py8_mg5amc_bias_interface.cc

PY8_CKKW: PY8_CKKW.o py8_mg5amc_bias_interface.o
	$(call CREATELIB, $(BIASLIBDIR)$(BIASLIBRARY), $^)

#
# List of the requirements for this module.
# 'VALID' is the keyword that *must* be returned if everything is in order.
#
requirements:	
ifeq ($(shell $(call CHECK_MG5AMC_VERSION, 2, 4, 2)),OK)
ifeq ($(PYTHIA8_PATH),NotInstalled)
	@echo "Error: This module requires Pythia8. Consider installing it with the command 'install pythia8'."
else
	@echo "VALID"
endif
else
	@echo "Error:: MG5aMC is not recent enough (found "$(MG5AMC_VERSION)")"
	@echo "FAIL"
endif
