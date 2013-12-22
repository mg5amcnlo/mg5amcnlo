FFLAGS        = 

LFLAGS	      = $(FFLAGS)

DEST	      = /usr/local/lib

HDRS	      = aa.h \
		ff.h

LIBRARY	      = ff.a

MAKEFILE      = Makefile

AAOBJS	      = aaxbx.o \
		aaxcx.o \
		aaxdx.o \
		aaxex.o \
		aaxinv.o \
		aacbc.o \
		aaccc.o \
		aacinv.o \
		spence.o \
		npoin.o

OBJS	      = ff2dl2.o \
		ffabcd.o \
		ffca0.o \
		ffcb0.o \
		ffcb1.o \
		ffcb2.o \
		ffcb2p.o \
		ffcdb0.o \
		ffcc0.o \
		ffcc0p.o \
		ffcc1.o \
		ffcel2.o \
		ffcel3.o \
		ffcel4.o \
		ffcel5.o \
		ffceta.o \
		ffcli2.o \
		ffcrr.o \
		ffcxr.o \
		ffcxs3.o \
		ffcxs4.o \
		ffcxyz.o \
		ffdcc0.o \
		ffdcxs.o \
		ffdel2.o \
		ffdel3.o \
		ffdel4.o \
		ffdel5.o \
		ffdel6.o \
		ffdl2i.o \
		ffdl5p.o \
		ffdxc0.o \
		ffinit.o \
		ffrcvr.o \
		fftran.o \
		ffxb0.o \
		ffxb1.o \
		ffxb2p.o \
		ffxc0.o \
		ffxc0i.o \
		ffxc0p.o \
		ffxc1.o \
		ffxd0.o \
		ffxd0h.o \
		ffxd0i.o \
		ffxd0p.o \
		ffxd1.o \
		ffxdb0.o \
		ffxdbd.o \
		ffxdi.o \
		ffxdpv.o \
		ffxe0.o \
		ffxe1.o \
		ffxf0.o \
		ffxf0h.o \
		ffxli2.o \
		ffxxyz.o

# older software can be made to run with this dummy routine
#		ffini.o

PRINT	      = pr

AASRCS	      = aaxbx.f \
		aaxcx.f \
		aaxdx.f \
		aaxex.f \
		aaxinv.f \
		aacbc.f \
		aaccc.f \
		aacinv.f \
		spence.f \
		npoin.f

SRCS	      = ff2dl2.f \
		ffabcd.f \
		ffca0.f \
		ffcb0.f \
		ffcb1.f \
		ffcb2.f \
		ffcb2p.f \
		ffcdb0.f \
		ffcc0.f \
		ffcc0p.f \
		ffcc1.f \
		ffcdb0.f \
		ffcel2.f \
		ffcel3.f \
		ffcel4.f \
		ffcel5.f \
		ffceta.f \
		ffcli2.f \
		ffcrr.f \
		ffcxr.f \
		ffcxs3.f \
		ffcxs4.f \
		ffcxyz.f \
		ffdcc0.f \
		ffdcxs.f \
		ffdel2.f \
		ffdel3.f \
		ffdel4.f \
		ffdel5.f \
		ffdel6.f \
		ffdl2i.f \
		ffdl5p.f \
		ffdxc0.f \
		ffinit.f \
		ffrcvr.f \
		fftran.f \
		ffxb0.f \
		ffxb1.f \
		ffxb2p.f \
		ffxc0.f \
		ffxc0i.f \
		ffxc0p.f \
		ffxc1.f \
		ffxd0.f \
		ffxd0h.f \
		ffxd0i.f \
		ffxd0p.f \
		ffxd1.f \
		ffxdb0.f \
		ffxdbd.f \
		ffxdi.f \
		ffxdpv.f \
		ffxe0.f \
		ffxe1.f \
		ffxf0.f \
		ffxf0h.f \
		ffxli2.f \
		ffxxyz.f

all:		$(LIBRARY)

test:		npointes
		npointes > npointes.out
		diff -w npointes.out npointes.sun

npointes:	npointes.o $(LIBRARY)
		f77 $(LFLAGS) -o npointes npointes.o $(LIBRARY)

$(LIBRARY):	$(OBJS) $(AAOBJS)
		ar cru $(LIBRARY) $(OBJS) $(AAOBJS)
		ranlib $(LIBRARY)

clean:;		rm -f $(OBJS) $(AAOBJS) $(LIBRARY) npointes.o npointes

depend:;	mkmf -f $(MAKEFILE) LIBRARY=$(LIBRARY) DEST=$(DEST)

extract:;	ar xo $(DEST)/$(LIBRARY)
		@rm -f __.SYMDEF

index:;		ctags -wx $(HDRS) $(SRCS)

install:	$(LIBRARY)
		install $(LIBRARY) $(DEST)
		ranlib $(DEST)/$(LIBRARY)

library:	$(LIBRARY)

print:;		$(PRINT) $(HDRS) $(SRCS)

update:		$(DEST)/$(LIBRARY)

$(DEST)/$(LIBRARY): $(SRCS) $(HDRS) $(EXTHDRS)
		@-ar xo $(DEST)/$(LIBRARY)
		@make -f $(MAKEFILE) DEST=$(DEST) install clean
###
aaxbx.o: ff.h aa.h
aaxcx.o: ff.h aa.h
aaxdx.o: ff.h aa.h
aaxex.o: ff.h aa.h
aaxinv.o: ff.h aa.h
ff2dl2.o: ff.h
ffabcd.o: ff.h
ffca0.o: ff.h
ffcb0.o: ff.h
ffcb1.o: ff.h
ffcc0.o: ff.h
ffcc0p.o: ff.h
ffcc1.o: ff.h
ffcdb0.o: ff.h
ffcel2.o: ff.h
ffcel3.o: ff.h
ffcel4.o: ff.h
ffcel5.o: ff.h
ffceta.o: ff.h
ffcli2.o: ff.h
ffcrr.o: ff.h
ffcxr.o: ff.h
ffcxs3.o: ff.h
ffcxs4.o: ff.h
ffcxyz.o: ff.h
ffdcc0.o: ff.h
ffdcxs.o: ff.h
ffdel2.o: ff.h
ffdel3.o: ff.h
ffdel4.o: ff.h
ffdel5.o: ff.h
ffdel6.o: ff.h
ffdl2i.o: ff.h
ffdl5p.o: ff.h
ffdxc0.o: ff.h
ffinit.o: ff.h aa.h
ffrcvr.o: ff.h
fftran.o: ff.h
ffxb0.o: ff.h
ffxb1.o: ff.h
ffxb2p.o: ff.h
ffxc0.o: ff.h
ffxc0i.o: ff.h
ffxc0p.o: ff.h
ffxc1.o: ff.h
ffxd0.o: ff.h
ffxd0h.o: ff.h
ffxd0i.o: ff.h
ffxd0p.o: ff.h
ffxd1.o: ff.h aa.h
ffxdb0.o: ff.h
ffxdbd.o: ff.h
ffxdi.o: ff.h
ffxdpv.o: ff.h
ffxe0.o: ff.h
ffxe1.o: ff.h
ffxf0.o: ff.h
ffxf0h.o: ff.h
ffxli2.o: ff.h
ffxxyz.o: ff.h
npoin.o: ff.h aa.h
npointes.o: ff.h
