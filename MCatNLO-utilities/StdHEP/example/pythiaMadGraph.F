      PROGRAM PYXTRA
C...Long example how to interface user-defined processes to PYTHIA
C...based on the Les Houches commonblock agreement. Generates events 
C...of several different kinds, to test the new code under varied 
C...conditions, but kinematics selection and cross sections are
C...completely unphysical.

C...Double precision and integer declarations.
      IMPLICIT NONE
      INTEGER PYK,PYCHGE,PYCOMP

#include "hepeup.inc"
#include "heprup.inc"
#include "stdlun.inc"
 
C...Standard PYTHIA commonblocks.
#include "pyjets.inc"
#include "pydat1.inc"
#include "pydat3.inc"
#include "pysubs.inc"
#include "pypars.inc"

C...Extra commonblock to transfer run info.
      INTEGER MODE,NLIM
      REAL*8 SCALEF,WTMAX
      COMMON/PRIV/MODE,NLIM,SCALEF,WTMAX

C...Local arrays.

      REAL*8 XLUM
      character*72 FNAME
      character*5 cgive
      character*30 cgive0
      integer istr
      integer lok,NEV,iev

      EXTERNAL PYDATA


C...Switch process mode; agrees with IDWTUP code (+-1,+-2,+-3,+-4).
      MODE=2
C.....2 means weighted in and unweighted out
C......Events selected according to xsection

CCC   MODE=1
C.....1 means weighted in and unweighted out
C......Events selected according to maximum weight

C...Maximum number of events to generate.
      OPEN(55,FILE='pythiaMadGraph.in',status='old')
      NEV=100
      READ(55,*) MODE,XLUM,FNAME,SCALEF

C initialize HEP logical units
      lnhwrt=23
      lnhrd=0
      lnhdcy=0
      lnhpdf=77
      lnhout=22
      lnhdmp=0
      mstu(11)=lnhout
      open(unit=lnhout,file='pythiaMadGraph.lpt',status='new')
C
      WRITE(CGIVE,'(I5)') lnhout
      CGIVE0='MSTU(11)='//CGIVE
      CALL PYGIVE(CGIVE0)


C...Initialize with external process.
      CALL PYINIT('USER',' ',' ',0D0)

      NEV = INT (XLUM * XSECUP(1))
      IF(NEV.LE.0) NEV=1


      write(lnhout,*) ' requesting ',XLUM,' pbi of data '
      write(lnhout,*) ' cross section is ',xsecup(1),' pb '
      write(lnhout,*) ' events to be generated = ',nev

C.....Opening stdhep file for writing
      call stdxwinit(FNAME,'StdHep/Pythia example',
     1               nev,istr,lok)
      if(lok.ne.0) write(lnhout,*)
     1        ' Problem opening file '

C          Write Stdhep begin-run record   
      call stdxwrt(100,istr,lok)
      if(lok.ne.0) write(lnhout,*)
     1        ' Problem writing stdhep begin run record'

C...Event loop; generate event; check it was obtained or quit.
      DO 130 IEV=1,NEV
        CALL PYEVNT
        CALL LUNHEP(1)
        IF(MSTI(51).EQ.1) GOTO 140  

        IF(IEV.LT.10) THEN
         CALL PYLIST(7)
         CALL PYLIST(2)
        ELSEIF(IEV.LT.50) THEN
           IF(MOD(IEV-1,10).EQ.0) THEN
              write(lnhout,*) ' IEV = ',IEV
              CALL PYLIST(7)
              CALL PYLIST(2)
           ENDIF
        ENDIF
 
C.......Write one event
        call stdxwrt(1,istr,lok)


  130 CONTINUE

C...Statistics and histograms.
  140 CALL PYSTAT(1)


C          Fill Stdhep common block 1 with run information
      call stdflpyxsec(nev)
C          Write end-of-run record  
      call stdxwrt(200,istr,lok)
      if(lok.ne.0) write(lnhout,*) ' Problem writing end run record'


c...close event file
      call stdxend(istr)
      close(lnhpdf)
      END
   
C*********************************************************************
 
C...UPINIT
C...Routine to be called by user to set up user-defined processes.
C...Code below only intended as example, without any claim of realism.
C...Especially it shows what info needs to be put in HEPRUP.
 
      SUBROUTINE UPINIT
 
C...Double precision and integer declarations.
      IMPLICIT NONE

#include "heprup.inc"
#include "hepeup.inc"
#include "stdlun.inc"

      REAL*8 XSTOT,WTGRT
      INTEGER I,J,NPARTS,IMAX,NGRT,lok

C...Extra commonblock to transfer run info.
      INTEGER MODE,NLIM
      REAL*8 SCALEF,WTMAX
      COMMON/PRIV/MODE,NLIM,SCALEF,WTMAX
      SAVE/PRIV/

C...Set incoming beams: Tevatron Run II.
      IDBMUP(1)=2212
      IDBMUP(2)=-2212
      EBMUP(1)=980D0
      EBMUP(2)=980D0

C...Set PDF's of incoming beams: CTEQ 5L.
C...Note that Pythia will not look at PDFGUP and PDFSUP.  
      PDFGUP(1)=4
      PDFSUP(1)=46
      PDFGUP(2)=PDFGUP(1)
      PDFSUP(2)=PDFSUP(1)
      
C...If you want Pythia to use PDFLIB, you have to set it by hand.
C...(You also have to ensure that the dummy routines
C...PDFSET, STRUCTM and STRUCTP in Pythia are not linked.)      
C      MSTP(52)=2
C      MSTP(51)=1000*PDFGUP(1)+PDFSUP(1)

C...Decide on weighting strategy: unweighted on input.
      IDWTUP=MODE

C...Number of external processes. 
      NPRUP=1

C.....read through data file to find maximum weight....only first time

      open(unit=lnhpdf,file='evgg.mad',status='old')

      WTMAX=0D0
      XSTOT=0D0

      WTGRT=0D0
      NGRT=0D0
      IMAX=0
      lok = 0

      DO while( lok.eq.0 )
        call readmadgraph(1.9,lok)
	if(lok.eq.0) then
	  nparts=nup
          IF(xwgtup.GT.WTMAX) WTMAX=xwgtup
          XSTOT=XSTOT+xwgtup
          IMAX=imax+1
	endif
      ENDDO
      rewind(lnhpdf)

      IF(SCALEF.LT.1D0.AND.SCALEF.GT.0D0) THEN
       WTMAX=WTMAX*SCALEF
       DO I=1,IMAX
         call readmadgraph(1.9,lok)
	 if(lok.eq.-1) go to 778
	 nparts=nup
         IF(xwgtup.GT.WTMAX) THEN
           NGRT=NGRT+1
           WTGRT=WTGRT+xwgtup
         ENDIF
       ENDDO
 778   CONTINUE
       rewind(lnhpdf)
       write(lnhout,*) ' For unweighting, scaled maximum by ',SCALEF
       write(lnhout,*) ' Number of Events with wt > wtmax ',NGRT, 
     $ ' out of ',IMAX
       write(lnhout,*) ' Total Cross Section with wt > wtmax ',WTGRT
       write(lnhout,*) ' Relative to total ',WTGRT/XSTOT
      ENDIF

      XSECUP(1)=XSTOT
      XMAXUP(1)=WTMAX
      LPRUP(1)=661

      RETURN
      END
C*********************************************************************
 
C...UPEVNT
C...Sample routine to generate events of various kinds.
C...Not intended to be realistic, but rather to show in closed
C...and understandable form what such a routine might look like.
C...Especially it shows what info needs to be put in HEPEUP.
 
      SUBROUTINE UPEVNT

C...Double precision and integer declarations.
      IMPLICIT NONE

#include "pypars.inc"

C...Extra commonblock to transfer run info.
      INTEGER MODE,NLIM
      REAL*8 SCALEF,WTMAX
      COMMON/PRIV/MODE,NLIM,SCALEF,WTMAX
      SAVE/PRIV/

#include "hepeup.inc"
#include "stdlun.inc"

      REAL*8 TEST,QMIN,SMDOT5,QTMP
      INTEGER I,IJ,IC1,IC2,IDC1,IDC2,IDUMB,IC,J
      INTEGER MUP,IDH,IDL,ID,IDP,IDA,IDPA,IMO,IS
      integer lok

C...If PYTHIA is supposed to select event type, do not modify this choice.
      IDPRUP=661

      call readmadgraph(1.9,lok)

C.....Account for scale factor
      IF(XWGTUP.GT.WTMAX) XWGTUP=WTMAX

C...Guesses for the correct scale
C   Assumptions:
C     (1) if the initial state is a color singlet, then
C         use s-hat for the scale
C
C     (2) if color flow to the final state, use the minimum
C         of the dot products of color connected pairs
C         (times two for consistency with above)

      QMIN=SMDOT5(PUP(1,1),PUP(1,2))
C
      DO IJ=1,NUP
       IDC1=ICOLUP(1,IJ)
       IDC2=ICOLUP(2,IJ)
       IF(IDC1.EQ.0) IDC1=-1
       IF(IDC2.EQ.0) IDC2=-2
       DO IC=IJ+1,NUP
        IC1=ICOLUP(1,IC)
        IC2=ICOLUP(2,IC)
        IF(ISTUP(IC)*ISTUP(IJ).EQ.1) THEN
         IF(IDC1.EQ.IC2.OR.IDC2.EQ.IC1) THEN
          QTMP=SMDOT5(PUP(1,IJ),PUP(1,IC))
          QMIN=MIN(QMIN,QTMP)
         ENDIF
        ELSEIF(ISTUP(IC)*ISTUP(IJ).EQ.-1) THEN
         IF(IDC1.EQ.IC1.OR.IDC2.EQ.IC2) THEN
          QTMP=SMDOT5(PUP(1,IJ),PUP(1,IC))          
          QMIN=MIN(QMIN,QTMP)
         ENDIF
        ENDIF
       ENDDO
      ENDDO

      SCALUP=QMIN
      
c      write(lnhout,*) ' scalup = ',scalup
C.....Add mothers for l+ l- and l nu_l pairs
      IJ=3
 555  CONTINUE
      IS=ISTUP(IJ)
      IF(IS.NE.1) GOTO 998
      ID=IDUP(IJ)
      IDA=ABS(ID)
      write(lnhout,*) ij,ida,nup
      IF(IDA.GE.11.AND.IDA.LE.16.AND.IJ.LT.NUP) THEN
       IF(ICOLUP(1,IJ).EQ.0 .AND. ICOLUP(2,IJ).EQ.0) THEN        
        IDP=IDUP(IJ+1)
        IDPA=ABS(IDP)
        IMO=0
        IF(IDP.EQ.-ID) THEN
C.......gamma*/Z* mother
         IMO=22
        ELSE
         IDL=MIN(IDA,IDPA)
         IDH=MAX(IDA,IDPA)
         IF(MOD(IDH,2).EQ.0.AND.(IDH-IDL.EQ.1)) THEN
          IF(ABS(ID).EQ.IDL) THEN
           IMO=-SIGN(24,ID)
          ELSE
           IMO=-SIGN(24,IDP)
          ENDIF
         ENDIF
        ENDIF
        write(lnhout,*) ' IMO = ',IMO
        IF(IMO.EQ.0) GOTO 998
        MUP=NUP
        NUP=NUP+1
        DO IC=NUP,IJ+1,-1
         IDUP(IC)=IDUP(IC-1)
         ISTUP(IC)=ISTUP(IC-1)
         MOTHUP(1,IC)=MOTHUP(1,IC-1)
         MOTHUP(2,IC)=MOTHUP(2,IC-1)
         ICOLUP(1,IC)=ICOLUP(1,IC-1)
         ICOLUP(2,IC)=ICOLUP(2,IC-1)
         DO J=1,5
          PUP(J,IC)=PUP(J,IC-1)
         ENDDO
         write(lnhout,*) ' ** ',ic,idup(ic),istup(ic)
        ENDDO
C.......
        write(lnhout,*) ' ij = ',ij
        IDUP(IJ)=IMO
        ISTUP(IJ)=2
        DO J=1,4
         PUP(J,IJ)=PUP(J,IJ+1)+PUP(J,IJ+2)
        ENDDO
        MOTHUP(1,IJ+1)=IJ
        MOTHUP(2,IJ+1)=IJ
        MOTHUP(1,IJ+2)=IJ
        MOTHUP(2,IJ+2)=IJ
        TEST=PUP(1,IJ)**2+PUP(2,IJ)**2+PUP(3,IJ)**2-PUP(4,IJ)**2
        IF(TEST.LE.0D0) THEN
         PUP(5,IJ)=DSQRT(-TEST)         
        ELSEIF(DABS(TEST).LT.1D-4) THEN
         PUP(5,IJ)=0D0
        ELSE
         PUP(5,IJ)=-1D0
         write(lnhout,*) ' NEGATIVE MASS '
        ENDIF
        IJ=IJ+2
       ENDIF
      ENDIF
        
       
 998  CONTINUE
      IJ=IJ+1
      IF(IJ.GT.NUP) GOTO 999
      GOTO 555
 999  CONTINUE


      write(lnhout,*) ' event '

        DO IC=1,NUP
         write(lnhout,*) ic,idup(ic),istup(ic),mothup(1,ic),mothup(2,ic)
        ENDDO

      RETURN
      END
C
      FUNCTION SMDOT5(V1,V2)
      IMPLICIT NONE
      REAL*8 SMDOT5,TEMP
      REAL*8 V1(5),V2(5)
      INTEGER I

      SMDOT5=0D0
      TEMP=V1(4)*V2(4)
      DO I=1,3
       TEMP=TEMP-V1(I)*V2(I)
      ENDDO

      SMDOT5=SQRT(2D0*ABS(TEMP))

      RETURN
      END

