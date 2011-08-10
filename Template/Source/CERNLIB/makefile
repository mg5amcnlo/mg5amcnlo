include ../make_opts

LIBRARY	= libcernlib.$(libext)
LIBDIR  = ../../lib/
SOURCES = abend.o dlsqp2.o lenocc.o mtlprt.o mtlset.o radmul.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(SOURCES)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
