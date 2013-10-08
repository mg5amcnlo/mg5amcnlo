C...This file contains an almost complete listing of all PYTHIA 
C...commonblocks, and additionally some recommended other 
C...declarations. You may copy this to the top of your
C...mina program and then eliminate unnecessary parts.

C-----------------------------------------------------------------
 
C...All real arithmetic in double precision.
      IMPLICIT DOUBLE PRECISION(A-H, O-Z)
C...Three Pythia functions return integers, so need declaring.
      INTEGER PYK,PYCHGE,PYCOMP
C...Parameter statement to help give large particle numbers
C...(left- and righthanded SUSY, technicolor, excited fermions,
C...extra dimensions).
      PARAMETER (KSUSY1=1000000,KSUSY2=2000000,KTECHN=3000000,
     &KEXCIT=4000000,KDIMEN=5000000)

C...EXTERNAL statement links PYDATA on most machines.
      EXTERNAL PYDATA

C...Commonblocks.
C...The event record.
      COMMON/PYJETS/N,NPAD,K(4000,5),P(4000,5),V(4000,5)
C...Parameters.
      COMMON/PYDAT1/MSTU(200),PARU(200),MSTJ(200),PARJ(200)
C...Particle properties + some flavour parameters.
      COMMON/PYDAT2/KCHG(500,4),PMAS(500,4),PARF(2000),VCKM(4,4)
C...Decay information.
      COMMON/PYDAT3/MDCY(500,3),MDME(8000,2),BRAT(8000),KFDP(8000,5)
C...Particle names
      COMMON/PYDAT4/CHAF(500,2)
      CHARACTER CHAF*16
C...Random number generator information.
      COMMON/PYDATR/MRPY(6),RRPY(100)
C...Selection of hard scattering subprocesses.
      COMMON/PYSUBS/MSEL,MSELPD,MSUB(500),KFIN(2,-40:40),CKIN(200)
C...Parameters. 
      COMMON/PYPARS/MSTP(200),PARP(200),MSTI(200),PARI(200)
C...Internal variables.
      COMMON/PYINT1/MINT(400),VINT(400)
C...Process information.
      COMMON/PYINT2/ISET(500),KFPR(500,2),COEF(500,20),ICOL(40,4,2)
C...Parton distributions and cross sections.
      COMMON/PYINT3/XSFX(2,-40:40),ISIG(1000,3),SIGH(1000)
C...Resonance width and secondary decay treatment.
      COMMON/PYINT4/MWID(500),WIDS(500,5)
C...Generation and cross section statistics.
      COMMON/PYINT5/NGENPD,NGEN(0:500,3),XSEC(0:500,3)
C...Process names.
      COMMON/PYINT6/PROC(0:500)
      CHARACTER PROC*28
C...Total cross sections.
      COMMON/PYINT7/SIGT(0:6,0:6,0:5)
C...Photon parton distributions: total and valence only.
      COMMON/PYINT8/XPVMD(-6:6),XPANL(-6:6),XPANH(-6:6),XPBEH(-6:6), 
     &XPDIR(-6:6) 
      COMMON/PYINT9/VXPVMD(-6:6),VXPANL(-6:6),VXPANH(-6:6),VXPDGM(-6:6) 
C...Supersymmetry parameters.
      COMMON/PYMSSM/IMSS(0:99),RMSS(0:99)
C...Supersymmetry mixing matrices.
      COMMON/PYSSMT/ZMIX(4,4),UMIX(2,2),VMIX(2,2),SMZ(4),SMW(2),
     &SFMIX(16,4),ZMIXI(4,4),UMIXI(2,2),VMIXI(2,2)
C...R-parity-violating couplings in supersymmetry.
      COMMON/PYMSRV/RVLAM(3,3,3), RVLAMP(3,3,3), RVLAMB(3,3,3)
C...Internal parameters for R-parity-violating processes.
      COMMON/PYRVNV/AB(2,16,2),RMS(0:3),RES(6,5),IDR,IDR2,DCMASS,KFR(3)
      COMMON/PYRVPM/RM(0:3),A(2),B(2),RESM(2),RESW(2),MFLAG
      LOGICAL MFLAG
C...Parameters for Gauss integration of supersymmetric widths.
      COMMON/PYINTS/XXM(20)
      COMMON/PYG2DX/X1
C...Histogram information.
      COMMON/PYBINS/IHIST(4),INDX(1000),BIN(20000)
C...HEPEVT commonblock.
      PARAMETER (NMXHEP=4000)
      COMMON/HEPEVT/NEVHEP,NHEP,ISTHEP(NMXHEP),IDHEP(NMXHEP),
     &JMOHEP(2,NMXHEP),JDAHEP(2,NMXHEP),PHEP(5,NMXHEP),VHEP(4,NMXHEP)
      DOUBLE PRECISION PHEP,VHEP
C...User process initialization commonblock.
      INTEGER MAXPUP
      PARAMETER (MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON/HEPRUP/IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),XMAXUP(MAXPUP),
     &LPRUP(MAXPUP)
C...User process event common block.
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,IDUP(MAXNUP),
     &ISTUP(MAXNUP),MOTHUP(2,MAXNUP),ICOLUP(2,MAXNUP),PUP(5,MAXNUP),
     &VTIMUP(MAXNUP),SPINUP(MAXNUP)
