#
# Platform independent part of make procedure for Nirvana utilities directory, 
# included by machine specific makefiles.
#
include ../../stdhep_arch

ifeq "$(UNAME)" "AIX"
    CFLAGS += -O -DUSE_DIRENT
endif
ifeq "$(UNAME)" "IRIX"
    CFLAGS += -O -DFUNCPROTO -DXTFUNCPROTO
endif
ifeq "$(UNAME)" "IRIX64"
     CFLAGS += -O -n32 -mips3 -DFUNCPROTO -DXTFUNCPROTO
endif
ifeq "$(UNAME)" "Linux"
    CFLAGS += -O -DFUNCPROTO -DXTFUNCPROTO -I/usr/X11/include
endif
ifeq "$(UNAME)" "Linux22"
    CFLAGS += -O -DFUNCPROTO -DXTFUNCPROTO -I/usr/X11/include -I$(OPENMOTIF_DIR)/include
endif
ifeq "$(UNAME)" "Linux24"
    CFLAGS += -O -DFUNCPROTO -DXTFUNCPROTO -I/usr/X11R6/include
endif
ifeq "$(UNAME)" "OSF1"
     CFLAGS += -DUSE_DIRENT -DFUNCPROTO -DXTFUNCPROTO
endif
ifeq "$(UNAME)" "SunOS"
    CFLAGS += -Xa -I/usr/openwin/include -I/usr/dt/include -DUSE_DIRENT \
	    -DMOTIF12
endif

NLIBDIR = ../../../lib

SRCS = DialogF.c stringUtils.c getfiles.c printUtils.c psUtils.c \
	misc.c fileUtils.c help.c prefFile.c fontsel.c
OBJS = $(patsubst %.c,%$(SFX).o,$(SRCS))
INCLS= DialogF.h getfiles.h stringUtils.h printUtils.h psUtils.h \
	misc.h fileUtils.h help.h prefFile.h fontsel.h
VMSSRCS = vmsUtils.c vmsUtils.h vmsparam.h

all: $(NLIBDIR)/libNUtil$(SFX).a 

$(NLIBDIR)/libNUtil$(SFX).a: $(OBJS)
	$(AR) -urs $(NLIBDIR)/libNUtil$(SFX).a $(OBJS)

printUtils$(SFX).o: printUtils.c
	$(CC) -c $(CFLAGS) $(DFLG) $(PRINTFLAGS) -o $@ printUtils.c

clean:
	rm -f *.o

realclean:
	rm -f *.o $(NLIBDIR)/libNUtil*.a

depend:
	$(MAKEDEPEND) -fMakefile.common -- $(CFLAGS) -- $(SRCS)
	
#------------------------------------------

.SUFFIXES : 		# cancel implicit rules
.SUFFIXES : .o .c

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%_g.o: %.c
	$(CC) $(CFLAGS) $(DFLG) -c -o $@ $<

# DO NOT DELETE THIS LINE -- make depend depends on it.

DialogF.o: DialogF.h
stringUtils.o: stringUtils.h
getfiles.o: fileUtils.h
getfiles.o: getfiles.h
printUtils.o: DialogF.h
printUtils.o: printUtils.h
psUtils.o: psUtils.h
misc.o: DialogF.h
misc.o: misc.h
fileUtils.o: fileUtils.h
help.o: misc.h
help.o: help.h
DialogF_g.o: DialogF.h
stringUtils_g.o: stringUtils.h
getfiles_g.o: fileUtils.h
getfiles_g.o: getfiles.h
printUtils_g.o: DialogF.h
printUtils_g.o: printUtils.h
psUtils_g.o: psUtils.h
misc_g.o: DialogF.h
misc_g.o: misc.h
fileUtils_g.o: fileUtils.h
help_g.o: misc.h
help_g.o: help.h
