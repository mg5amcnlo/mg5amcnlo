#
# Platform independent part of make procedure for Nirvana spin widget, 
# included by machine specific makefiles.
#
include ../../stdhep_arch

ifeq "$(UNAME)" "AIX"
    CFLAGS += -D_BSD
endif
ifeq "$(UNAME)" "IRIX"
    CFLAGS += -DSYSV
endif
ifeq "$(UNAME)" "IRIX64"
    CFLAGS += -DSYSV
endif
ifeq "$(UNAME)" "Linux"
    CFLAGS += -DFUNCPROTO -DXTFUNCPROTO -I/usr/X11/include
endif
ifeq "$(UNAME)" "Linux22"
    CFLAGS += -DFUNCPROTO -DXTFUNCPROTO -I/usr/X11/include -I$(OPENMOTIF_DIR)/include
endif
ifeq "$(UNAME)" "Linux24"
    CFLAGS += -DFUNCPROTO -DXTFUNCPROTO -I/usr/X11R6/include
endif
ifeq "$(UNAME)" "OSF1"
    CFLAGS += -DUSE_DIRENT -DFUNCPROTO -DXTFUNCPROTO
endif
ifeq "$(UNAME)" "SunOS"
    CFLAGS += -Xa -I/usr/openwin/include -I/usr/dt/include -DMOTIF12
endif

OBJS = Spin$(SFX).o geometry$(SFX).o
SRCS = Spin.c geometry.c
INCLS= Spin.h SpinP.h geometry.h

NLIBDIR = ../../../lib

all: $(NLIBDIR)/libSpin$(SFX).a

$(NLIBDIR)/libSpin$(SFX).a: $(OBJS)
	$(AR) -urs $(NLIBDIR)/libSpin$(SFX).a $(OBJS)

clean:
	rm -f *.o

realclean:
	rm -f *.o $(NLIBDIR)/libSpin*.a

#------------------------------------------

.SUFFIXES : 		# cancel implicit rules
.SUFFIXES : .o .c

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%_g.o: %.c
	$(CC) $(CFLAGS) $(DFLG) -c -o $@ $<

# DO NOT DELETE THIS LINE -- make depend depends on it.

Spin.o: geometry.h
Spin.o: SpinP.h
Spin.o: Spin.h
geometry.o: geometry.h
Spin_g.o: geometry.h
Spin_g.o: SpinP.h
Spin_g.o: Spin.h
geometry_g.o: geometry.h
