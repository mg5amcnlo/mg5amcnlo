default: lib

#lib: libcts.a ltcut
lib: libcts.a 

MPFUN = mpfun90.o mpmod90.o mpmodm90.o mpmodx90.o

CTS = cts_type.o cts_tensors.o cts_constants.o cts_combinatorics.o \
  cts_dynamics.o cts_kinematics.o \
  cts_loopfunctions.o cts_cuttools.o \
  cts_cutroutines.o

AVH =   avh_olo.o 

QCDLOOP = aaxbx.o aaxcx.o aaxdx.o aaxex.o aaxinv.o aacbc.o \
          aaccc.o aacinv.o spence.o \
                npoin.o \
                ff2dl2.o \
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
                ffinit_mine.o \
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
                ffxxyz.o \
                qlinit.o \
                qlI4.o qlI3.o qlI2.o qlI1.o qlI4array.o\
qlzero.o  qlI4fin.o qlI4DNS41.o qlI4sub0m.o\
qlI4sub1m.o qlI4sub2m.o qlI4sub2ma.o qlI4sub2mo.o qlI4sub3m.o qlI3fin.o \
qlI3sub.o qlI2fin.o qlfndd.o\
qlLi2omprod.o qlLi2omrat.o \
qllnomrat4.o qlLi2omx.o qlLi2omx2.o \
qltrisort.o qlsnglsort.o qltri1.o  qltri2.o\
qltri3.o  qltri4.o qltri5.o qltri6.o qlbox1.o qlbox2.o qlbox3.o\
qlbox4.o qlbox5.o qlbox6.o qlbox7.o qlbox8.o qlbox9.o \
qlbox10.o qlbox11.o qlbox12.o qlbox13.o qlbox14.o \
qlbox15.o qlbox16.o  qlcLi2omx2.o qlcLi2omx3.o auxCD.o qlspencer.o \
qlratreal.o qlratgam.o ddilog.o qllnrat.o qlfunctions.o  qlYcalc.o qlkfn.o \
qlxpicheck.o

#ALL = $(MPFUN) $(AVH) $(CTS) $(LT) $(QCDLOOP)
ALL = $(MPFUN) $(AVH) $(CTS) $(QCDLOOP)

libcts.a: libcts.a($(ALL))

.SUFFIXES: .f90

.f90.a:
	$(FC) $(FFLAGS) -c $<
	$(AR) $(ARFLAGS) $@ $*.o
