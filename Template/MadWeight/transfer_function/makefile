# ----------------------------------------------------------------------------
#
# Makefile for the transfer function
#
# ----------------------------------------------------------------------------



LIBRARY       = libTF.$(libext)
LIBDIR        = ../../../lib/

include ../../make_opts

OBJS           =  transfer_function.o

all: $(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(OBJS)
	$(call CREATELIB, $@, $^)

transfer_function.o: nb_tf.inc

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)


