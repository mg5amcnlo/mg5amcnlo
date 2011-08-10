# ----------------------------------------------------------------------------
#
# Makefile for TOOLS library
# Apr. 25 2003
#
# ----------------------------------------------------------------------------


gfortran           = f77
FFLAGS        = -O -ffixed-line-length-132
LIBDIR        = ../../../lib/
CLASS = class_a.o class_b.o class_c.o class_d.o class_e.o class_f.o class_g.o
BLOKS= block_a.o block_b.o block_c.o block_d.o block_e.o
OBJS= $(CLASS) $(BLOKS) fuse.o


.f.o: ; $(gfortran) $(FFLAGS) -c $*.f

all: $(LIBDIR)libblocks.a

$(LIBDIR)libblocks.a: $(OBJS)
	ar cru libblocks.a $(OBJS)
	ranlib libblocks.a
	mv libblocks.a  $(LIBDIR)

class_a.f: ../../nexternal.inc ../../genps.inc 
block_a.f: ../../nexternal.inc ../../genps.inc
class_b.f: ../../nexternal.inc ../../genps.inc
block_b.f: ../../nexternal.inc ../../genps.inc
class_c.f: ../../nexternal.inc ../../genps.inc
block_c.f: ../../nexternal.inc ../../genps.inc
class_d.f: ../../nexternal.inc ../../genps.inc
block_d.f: ../../nexternal.inc ../../genps.inc
class_e.f: ../../nexternal.inc ../../genps.inc
block_e.f: ../../nexternal.inc ../../genps.inc
class_f.f: ../../nexternal.inc ../../genps.inc
class_g.f: ../../nexternal.inc ../../genps.inc
