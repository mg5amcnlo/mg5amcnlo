
include ../make_opts

LIBRARY	= libbias.$(libext)
LIBDIR  = ../../lib/

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)

#
# Compilation of the module ptj_bias
#

ptj_bias: ptj_bias.o
	$(call CREATELIB, $(LIBDIR)$(LIBRARY), $^)

#
# Compilation of the module dummy
#

None: dummy.o
	$(call CREATELIB, $(LIBDIR)$(LIBRARY), $^)
