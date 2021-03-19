FC=gfortran
FC=gfortran
FC=gfortran
FC=gfortran

CDFUN = yes
#CDFUN = no

CUBAVERS = 2.x

LIBRARY = libavh_oni.a

FLAGS = -O

SRCDIR = ./src
BUILD = ./build

OBJS = \
	$(BUILD)/avh_oni_hello.o \
	$(BUILD)/avh_oni_3div.o \
	$(BUILD)/avh_oni_3fin.o \
	$(BUILD)/avh_oni_dfam.o \
	$(BUILD)/avh_oni_4div.o \
	$(BUILD)/avh_oni_cmplx.o \
	$(BUILD)/avh_oni_func.o \
	$(BUILD)/avh_oni_cuba.o \
	$(BUILD)/avh_oni_dqagpe.o \
	$(BUILD)/avh_oni_d1mach.o \
	$(BUILD)/avh_oni_real.o

ifeq ($(CDFUN),yes)
all:		copy $(LIBRARY) clean
else
all:		copy edit $(LIBRARY) clean
endif

$(LIBRARY):	$(OBJS)
		ar cru $(LIBRARY) $(OBJS)
		ranlib $(LIBRARY)

edit:
	sed -i -e'/CDLOGyes/ s/cdlog/log/g' -e'/CDLOGyes/ s/CDLOGyes/CDLOGno/' $(BUILD)/*.f
	sed -i -e'/CDABSyes/ s/cdabs/abs/g' -e'/CDABSyes/ s/CDABSyes/CDABSno/' $(BUILD)/*.f

copy:
	cp $(SRCDIR)/*.f $(BUILD)
	cp $(SRCDIR)/avh_oni_cuba-$(CUBAVERS).h $(BUILD)/avh_oni_cuba.h

clean:
	rm -f $(BUILD)/*
