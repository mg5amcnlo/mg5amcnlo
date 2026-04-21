include ../../make_opts

all: ptj_bias

clean:
	$(RM) *.o $(BIASLIBDIR)$(BIASLIBRARY)

#
# Compilation of the module ptj_bias
#
ptj_bias.o: ptj_bias.f ../bias.inc
	$(FC) $(FFLAGS) $(LDFLAGS) -c -o ptj_bias.o ptj_bias.f

ptj_bias: ptj_bias.o
	$(call CREATELIB, $(BIASLIBDIR)$(BIASLIBRARY), $^)

#
# List of the requirements for this module.
# 'VALID' is the keyword that *must* be returned if everything is in order.
#
requirements:	
	@echo "VALID"

