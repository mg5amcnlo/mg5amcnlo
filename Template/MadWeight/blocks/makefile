# ----------------------------------------------------------------------------
#
# Makefile for blocks
#
# ----------------------------------------------------------------------------



LIBRARY       = libblocks.$(libext)
LIBDIR        = ../../../lib/

include ../../make_opts

CLASS = class_a.o class_b.o class_c.o class_d.o class_e.o class_f.o class_g.o class_h.o
BLOKS= block_a.o block_b.o block_c.o block_d.o block_e.o
OBJS= $(CLASS) $(BLOKS) fuse.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(OBJS)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)

