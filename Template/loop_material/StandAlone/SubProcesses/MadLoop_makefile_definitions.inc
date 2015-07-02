LINK_LOOP_LIBS = -L$(LIBDIR) -lcts %(link_tir_libs)s
LOOP_LIBS = $(LIBDIR)libcts.$(libext) %(tir_libs)s
LOOP_INCLUDE = %(tir_include)s
LOOP_PREFIX = %(prefix)s
DOTO = %(doto)s
DOTF = %(dotf)s
LINK_MADLOOP_LIB = -L$(LIBDIR) -lMadLoop
MADLOOP_LIB = $(LIBDIR)libMadLoop.$(libext)

$(MADLOOP_LIB):
	cd ..; make -f makefile_MadLoop OLP_static
