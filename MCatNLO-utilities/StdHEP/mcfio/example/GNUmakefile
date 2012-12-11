#
# make procedure for mcfio examples - use gmake
#

SRCDIR = ../src
LIBDIR = ../../lib

include ../arch_mcfio


FINC = -I$(SRCDIR) -I$(STDHEP_DIR)/src/inc
CINC = -I$(SRCDIR)
               
#--------------------------

all: t1 t2 t3

examples: tN1 tN2

oldexamples: t1 t2 t3 t4 t5 \
             t6 t7

tN1:  tN1.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ tN1.o \
		$(LIBDIR)/libFmcfio.a

tN2:  tN2.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ tN2.o \
		$(LIBDIR)/libFmcfio.a

t1:  t1.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@  t1.o \
		$(LIBDIR)/libFmcfio.a \
		$(LIBDIR)/libstdhep.a
	       
t2:  t2.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ t2.o \
		$(LIBDIR)/libFmcfio.a \
		$(LIBDIR)/libstdhep.a
	       
t3:  t3.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ t3.o \
		$(LIBDIR)/libFmcfio.a \
		$(LIBDIR)/libstdhep.a
	       
t4:  t4.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ t4.o \
		$(LIBDIR)/libFmcfio.a \
		$(LIBDIR)/libstdhep.a
	       
t5:  t5.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ t5.o \
		$(LIBDIR)/libFmcfio.a \
		$(LIBDIR)/libstdhep.a
	       
t6:  t6.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ t6.o \
		$(LIBDIR)/libFmcfio.a \
		$(LIBDIR)/libstdhep.a
	       
t7:  t7.o $(LIBDIR)/libFmcfio.a
	$(F77) $(FFLAGS) -o $@ t7.o \
		$(LIBDIR)/libFmcfio.a \
		$(LIBDIR)/libstdhep.a

clean:
	rm -f *.o
	       
realclean:
	rm -f *.o
	rm -f tN1
	rm -f tN2
	rm -f t1
	rm -f t2
	rm -f t3
	rm -f t4
	rm -f t5
	rm -f t6
	rm -f t7

#--------------------------

%.o: %.F
	$(F77) -c -o $*.o -DUNIX $(FINC) $(FFLAGS) $<

%.o: %.c
	$(CC) -c -o $*.o $(CFLAGS) $(CINC) $*.c

#--------------------------

.PHONY: realclean clean

.SUFFIXES : 		# cancel implicit rules
.SUFFIXES : .o .c .F

#--------------------------

# Fortran dependencies
tN1: mcfio.inc glob22.inc glo23.inc
tN2: mcfio.inc glob22.inc glo23.inc


# DO NOT DELETE THIS LINE -- make depend depends on it.

