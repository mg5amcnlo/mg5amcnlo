# ----------------------------------------------------------------------------
#
# Makefile for DHELAS 3.0 library
# Feb. 28, 2001
#
# ----------------------------------------------------------------------------
#
# Use a TAB to precede shell commands (e.g., f90, ar, etc).
# Note: this Makefile uses features that *may not*
# be supported by make utilities other than GNU make.
#
# ----------------------------------------------------------------------------
# coupsm.F and coupsm.inc are not to be included if HELAS is used with 
# the 2003 version of MadEvent
# ----------------------------------------------------------------------------

FC            = g77

FFLAGS        = -O -I.

LD	      = ld

DEST	      = ./lib/

EXTHDRS	      =

HDRS	      =

INSTALL	      = /etc/install

LIBRARY	      = ./lib/libdhelas3.a

MAKEFILE      = Makefile

OBJS	      =	httsxx.o hstlxx.o hvvshx.o \
		vvvtlx.o jvvtlx.o uvvvlx.o sstlxx.o usslxx.o \
		hvvhxx.o hvvvxx.o httaxx.o \
		vvsshx.o jvsshx.o ttssxx.o utssxx.o \
		ttsaxx.o jvvsxx.o vvvsxx.o jvshxx.o utsaxx.o \
		vvshxx.o jvtaxx.o vvtaxx.o uvvaxx.o \
		momntx.o mom2cx.o boostx.o rotxxx.o \
		ixxxxx.o oxxxxx.o vxxxxx.o sxxxxx.o \
		iovxxx.o fvixxx.o fvoxxx.o jioxxx.o j3xxxx.o \
		iosxxx.o fsixxx.o fsoxxx.o hioxxx.o \
		vvvxxx.o jvvxxx.o gggxxx.o jggxxx.o \
		vvsxxx.o jvsxxx.o hvvxxx.o \
		vssxxx.o jssxxx.o hvsxxx.o \
		sssxxx.o hssxxx.o \
		wwwwxx.o jwwwxx.o w3w3xx.o jw3wxx.o \
		ggggxx.o jgggxx.o \
		vvssxx.o jvssxx.o hvvsxx.o \
		ssssxx.o hsssxx.o \
		eaixxx.o eaoxxx.o jeexxx.o \
		ioscxx.o fsicxx.o fsocxx.o hiocxx.o \
		iovcxx.o fvicxx.o fvocxx.o jiocxx.o \
		iovgox.o fvigox.o fvogox.o jiogox.o \
		iovdmx.o fvidmx.o fvodmx.o jiodmx.o \
		iosgld.o fsigld.o fsogld.o hiogld.o \
		iovgld.o fvigld.o fvogld.o jiogld.o \
		txxxxx.o txxxx2.o \
		iotxkk.o iovtkk.o vvtxkk.o ftixkk.o ftoxkk.o \
                iovkxx.o vvvkxx.o jiokxx.o jvvkxx.o \
                sstxxx.o hstxxx.o ussxxx.o \
                iotxxx.o ftixxx.o ftoxxx.o uioxxx.o \
                vvtxxx.o uvvxxx.o jvtxxx.o \
                iovtxx.o fvtixx.o fvtoxx.o jiotxx.o uiovxx.o \
                vvvtxx.o jvvtxx.o uvvvxx.o \
                ggggtx.o jgggtx.o uggggx.o \
                w3w3nx.o jw3wnx.o jwwwnx.o wwwwnx.o

PRINT	      = pr

SHELL	      = /bin/sh

SYSHDRS	      =

MFLAGS        = -e

#.F.o:
%.o : %.F
	$(FC) $(FFLAGS) -c $<

%.o : %.f
	$(FC) $(FFLAGS) -c $<

all:		$(LIBRARY)

helas:
		-rm -f *.o
		export FFLAGS='-O +cpp' && \
		export LIBRARY=libdhelas3.a && \
		$(MAKE) $(MFLAGS)

helas_check:
		-rm -f *.o
		export FFLAGS='-O +cpp -DHELAS_CHECK' && \
		export LIBRARY=libdhelas3_check.a && \
		$(MAKE) $(MFLAGS)

install-helas:
		export LIBRARY=libdhelas3.a && \
		$(MAKE) $(MFLAGS) install	

install-helas_check:
		export LIBRARY=libdhelas3.a && \
		$(MAKE) $(MFLAGS) install	

$(LIBRARY):	$(OBJS)
		@echo  "Loading $(LIBRARY) ... "
		@ar cru $(LIBRARY) $(OBJS)
		@echo "done"

clean:;		@rm -f $(OBJS) core

clobber:;	@rm -f $(OBJS) $(LIBRARY) core tags

install:	$(LIBRARY)
	        @echo Installing $(LIBRARY) in $(DEST)
	        @if [ $(DEST) != . ]; then \
	        (rm -f $(DEST)/$(LIBRARY); $(INSTALL) -f $(DEST) $(LIBRARY)); fi

$(DEST)/$(LIBRARY): $(SRCS) $(HDRS) $(EXTHDRS)
	        @$(MAKE) -f $(MAKEFILE) ROOT=$(ROOT) DEST=$(DEST) install
