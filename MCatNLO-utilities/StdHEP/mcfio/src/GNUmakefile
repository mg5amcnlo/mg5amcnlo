#
# make procedure for mcfio  - use gmake
# This makefile will only build libFmcfio by default
#      -- gmake ntu if you want to build ntuBrowser and ntuBuild
#      -- ntuBrowser is built with histoscope if $HISTO_DIR is defined.
#
#
UNAME := $(shell uname)

LIBDIR = ../../lib
BINDIR = ../../bin

include ../arch_mcfio
 
FINC = -I.
CINC = -I.
HLIB =
ifdef HISTO_DIR
   CFLAGS += -DHISTO
   CINC += -I$(HISTO_DIR)/include
   HLIB += $(HISTO_DIR)/lib/libCHisto.a
endif

CL_F_SRC    = mcfio_FPrintDictionary \
               mcf_evt_xdr mcfio_FBinding mcfio_Util1 \
               mcfio_Direct mcfio_SeqDummy mcfio_Block \
               mcf_ntubldInit mcf_ntuBldDbinc \
               mcf_NTuIOFiles  mcf_NTuIOUtils \
	       mcfio_UserDictionary
CL_F_OBJS   = $(addsuffix $(SFX).o,$(CL_F_SRC))

CL_NTUBLD_SRC  = \
               mcf_NTuBuildWindow mcf_NTuBldMenu \
               mcf_NTuBldHelp mcf_NTuBldFiles \
               mcf_ntubldInit mcf_ntuBldDbinc DialogF misc help \
               getfiles fileUtils
CL_NTUBLD_OBJS   = $(addsuffix $(SFX).o,$(CL_NTUBLD_SRC))

CL_BROWSER_SRC = mcf_BrowseMainPanel mcf_BrowseMainMenu \
               mcf_BrowseHelp  mcf_BrowseMainProgram \
               mcf_BrowseUtil1 mcf_BrowseUtil2
CL_BROWSER_OBJS   = $(addsuffix $(SFX).o,$(CL_BROWSER_SRC))
               
LIBLIST = $(LIBDIR)/libFmcfio$(SFX).a
ifdef BUILD_SHARED
  LIBLIST += $(LIBDIR)/libFmcfio$(SFX).$(SHEXT)
  SHCL_F_OBJS   = $(patsubst %$(SFX).o,%$(SHX)$(SFX).o,$(CL_F_OBJS))
endif
               
#--------------------------

all: $(LIBLIST) 

ntu: $(BINDIR)/ntuBuild$(SFX) $(BINDIR)/ntuBrowser$(SFX)

$(LIBDIR)/libFmcfio$(SFX).a: $(CL_F_OBJS) 
	$(AR) -urs $(LIBDIR)/libFmcfio$(SFX).a $(CL_F_OBJS) 

$(LIBDIR)/libFmcfio$(SFX).$(SHEXT): $(SHCL_F_OBJS)
	@echo Creating shared library $@ ...
	@$(RM) $@
	$(CC) -Wl,-soname,$(@F) -shared -o $@ \
		$(SHCL_F_OBJS)

$(BINDIR)/ntuBuild$(SFX): $(CL_NTUBLD_OBJS) $(LIBDIR)/libFmcfio$(SFX).a \
                    mcf_BrowseUtil1$(SFX).o mcf_nTupleBuild$(SFX).o
	$(CC) $(CFLAGS) -o $@ \
	mcf_nTupleBuild$(SFX).o mcf_BrowseUtil1$(SFX).o \
	$(CL_NTUBLD_OBJS) \
	$(LIBDIR)/libFmcfio$(SFX).a $(HLIB) \
        $(MXTLIBS)

$(BINDIR)/ntuBrowser$(SFX): $(CL_BROWSER_OBJS) $(CL_NTUBLD_OBJS) $(LIBDIR)/libFmcfio$(SFX).a
	$(CC) $(CFLAGS) -o $@ \
	$(CL_BROWSER_OBJS) \
	$(CL_NTUBLD_OBJS) \
	$(LIBDIR)/libFmcfio$(SFX).a $(HLIB) \
        $(MXTLIBS)
        
clean:
	rm -f *.o 
	       
realclean:
	rm -f *.o 
	rm -f $(LIBDIR)/libFmcfio*
	rm -f $(BINDIR)/ntuBuild $(BINDIR)/ntuBuild_g
	rm -f $(BINDIR)/ntuBrowser $(BINDIR)/ntuBrowser_g

glib:
	@echo "libraries are debug by default"

#--------------------------

%.o: %.F
	$(F77) -c -o $*.o -DUNIX $(FINC) $(FFLAGS) $<

%_g.o: %.F
	$(F77) -c -o $*_g.o -DUNIX $(FINC) $(FFLAGS) $(DFLG) $<

%.o: %.c
	$(CC) -c -o $*.o $(CFLAGS) $(CINC) $*.c

%_g.o: %.c
	$(CC) -c -o $*_g.o $(CFLAGS) $(DFLG) $(CINC) $*.c

%_s.o: %.F
	$(F77) -c -o $@ -DUNIX $(FINC) $(FFLAGS) $(SHFLAG) $<

%_s_g.o: %.F
	$(F77) -c -o $@ -DUNIX $(FINC) $(FFLAGS) $(SHFLAG) $(DFLG) $<

%_s.o: %.c
	$(CC) -c -o $@ $(CFLAGS) $(CINC) $(SHFLAG) $*.c

%_s_g.o: %.c
	$(CC) -c -o $@ $(CFLAGS) $(DFLG) $(CINC) $(SHFLAG) $*.c

#--------------------------

.PHONY: realclean clean glib

.SUFFIXES : 		# cancel implicit rules
.SUFFIXES : .o .c .F

#--------------------------


# DO NOT DELETE THIS LINE -- make depend depends on it.

mcf_evt_xdr.o: mcf_xdr.h
mcfio_Direct.o: mcf_xdr.h mcfio_Dict.h mcfio_Util1.h mcfio_Direct.h
mcfio_SeqDummy.o:mcfio_Sequential.h
mcfio_Util1.o: mcf_xdr.h mcfio_Dict.h mcfio_Util1.h mcfio_Direct.h \
                mcfio_Sequential.h
mcfio_block.o: mcf_xdr.h mcfio_Dict.h mcfio_Util1.h mcfio_Block.h              
mcf_evt_xdr_g.o: mcf_xdr.h
mcfio_Direct_g.o: mcf_xdr.h mcfio_Dict.h mcfio_Util1.h mcfio_Direct.h
mcfio_SeqDummy_g.o:mcfio_Sequential.h
mcfio_Util1_g.o: mcf_xdr.h mcfio_Dict.h mcfio_Util1.h mcfio_Direct.h \
                mcfio_Sequential.h
mcfio_block_g.o: mcf_xdr.h mcfio_Dict.h mcfio_Util1.h mcfio_Block.h              
