# Note that optimization doesn't work on all machines

include ../make_opts

LIBRARY	= libcernlib.a
LIBDIR  = ../../lib/
SOURCES = abend.o dlsqp2.o lenocc.o mtlprt.o mtlset.o radmul.o

all: $(LIBDIR)$(LIBRARY)

%.o : %.f makefile
	$(F77) $(FFLAGS) -c $<

$(LIBDIR)$(LIBRARY): $(SOURCES)
	ar cru $(LIBRARY) $^
	ranlib $(LIBRARY)
	mv $(LIBRARY) $(LIBDIR)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
