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

include ../make_opts

LIBRARY	= libdhelas.$(libext)
LIBDIR  = ../../lib/

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

all:		$(LIBDIR)$(LIBRARY)

$(LIBDIR)$(LIBRARY): $(OBJS)
	$(call CREATELIB, $@, $^)

clean:
	$(RM) *.o $(LIBDIR)$(LIBRARY)
