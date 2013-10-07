#
# StdHep makefile
#
include ../stdhep_arch

STDHEP_DIR = ../..

#this has been added by MZ
FFLAGS+= -fd-lines-as-code

SLIB = $(STDHEP_DIR)/lib
SBIN = $(STDHEP_DIR)/bin
SINC = $(STDHEP_DIR)/src/inc

INCS = -I$(SINC) -I$(STDHEP_DIR)/mcfio/src \
       -I$(SINC)/isajet -I$(SINC)/pythia -I$(SINC)/herwig \
       -I$(SINC)/qq -I$(SINC)/dpm
CFLAGS += -I$(SINC) -I$(STDHEP_DIR)/mcfio/src
CPPFLGS = $(KPPFLAGS) $(INCS)

FOBJS =	$(patsubst %.F,%$(SFX).o,$(wildcard *.F))
COBJS =	$(patsubst %.c,%$(SFX).o,$(wildcard *.c))
XDROBJS = $(patsubst %.c,%$(SFX).o,$(wildcard mcf*.c))
UOBJS = copy_stdevent$(SFX).o sum_4vec_d$(SFX).o sum_4vec_f$(SFX).o

LIBLIST = $(SLIB)/libstdhep$(SFX).a $(SLIB)/libstdhepC$(SFX).a
ifdef BUILD_SHARED
  LIBLIST += $(SLIB)/libstdhep$(SFX).$(SHEXT) $(SLIB)/libstdhepC$(SFX).$(SHEXT)
  SHFOBJS   = $(patsubst %$(SFX).o,%$(SHX)$(SFX).o,$(FOBJS))
  SHCOBJS   = $(patsubst %$(SFX).o,%$(SHX)$(SFX).o,$(COBJS))
  SHXDROBJS = $(patsubst %$(SFX).o,%$(SHX)$(SFX).o,$(XDROBJS))
  SHUOBJS   = $(patsubst %$(SFX).o,%$(SHX)$(SFX).o,$(UOBJS))
endif


#------------------------------------------

all: $(LIBLIST)

$(SLIB)/libstdhep$(SFX).a:	$(FOBJS) $(XDROBJS) $(UOBJS)
	$(AR) -urs $(SLIB)/libstdhep$(SFX).a $(FOBJS) \
		$(XDROBJS) \
                $(UOBJS)

$(SLIB)/libstdhepC$(SFX).a:	$(COBJS)
	$(AR) -urs $(SLIB)/libstdhepC$(SFX).a $(COBJS)

$(SLIB)/libstdhep$(SFX).$(SHEXT): $(SHFOBJS) $(SHXDROBJS) $(SHUOBJS)
	@echo Creating shared library $@ ...
	@$(RM) $@
	$(CC) -Wl,-soname,$(@F) -shared -o $@ \
		$(SHFOBJS) $(SHXDROBJS) $(SHUOBJS)

$(SLIB)/libstdhepC$(SFX).$(SHEXT): $(SHCOBJS)
	@echo Creating shared library $@ ...
	@$(RM) $@
	$(CC) -Wl,-soname,$(@F) -shared -o $@ \
		$(SHCOBJS)

clean:
	rm -f *.o *.a *.f

realclean:
	rm -f $(SLIB)/libstdhep*.a
	rm -f $(SLIB)/libstdhep*.$(SHEXT)
	rm -f *.o *.a *.f

#------------------------------------------

.SUFFIXES : 		# cancel implicit rules
.SUFFIXES : .o .c .F

%.o: %.F
	$(F77) $(INCS) $(FFLAGS) -c $*.F

%_g.o: %.F
	$(F77) $(INCS) $(FFLAGS) -c -o $@ $*.F

%.o: %.c
	$(CC) $(CFLAGS) -c $<

%_g.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%_s.o: %.F
	$(F77) $(INCS) $(FFLAGS) $(SHFLAG) -c -o $@ $*.F

%_s_g.o: %.F
	$(F77) $(INCS) $(FFLAGS) $(SHFLAG) -c -o $@ $*.F

%_s.o: %.c
	$(CC) $(CFLAGS) $(SHFLAG) -c -o $@ $<

%_s_g.o: %.c
	$(CC) $(CFLAGS) $(SHFLAG) -c -o $@ $<
