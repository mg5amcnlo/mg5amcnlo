c Analysis for ZZ->eemumu
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE PYABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      implicit none
      integer kchg
      INTEGER MAXPUP
      PARAMETER(MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON /HEPRUP/ IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &                IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),
     &                XMAXUP(MAXPUP),LPRUP(MAXPUP)
      real * 4 pi
      parameter (pi=3.14160E0)
      character*10 MonteCarlo
      parameter (MonteCarlo="PYTHIA6Q")
      integer lhefile,nevents
      parameter (lhefile=98)
      INTEGER MQQ
      COMMON/cMQQ/MQQ
      DOUBLE PRECISION EVNORM
      COMMON/CEVNORM/EVNORM
      EVNORM = 1d0/DFLOAT(MQQ)
      OPEN(UNIT=lhefile,NAME='ZZ6.lhe',STATUS='UNKNOWN')
      nevents=-MQQ
      call write_lhef_header(lhefile,nevents,MonteCarlo)
      call write_lhef_init(lhefile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
c
 999  END


C----------------------------------------------------------------------
      SUBROUTINE PYAEND(IEVT)
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      implicit none
      LOGICAL ENDOFRUN
      COMMON/CENDOFRUN/ENDOFRUN
      integer lhefile 
      parameter (lhefile=98)
c
      IF(.NOT.ENDOFRUN)RETURN
      write(lhefile,*)'</LesHouchesEvents>'
      CLOSE(lhefile)
 999  END

C----------------------------------------------------------------------
      SUBROUTINE PYANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      implicit double precision(a-h, o-z)
      implicit integer(i-n)
      DOUBLE PRECISION PSUM(4)
      include 'reweight0.inc'
      INTEGER MAXNUP_tlh
      PARAMETER (MAXNUP_tlh=500)
      INTEGER NUP_tlh,IDPRUP_tlh,IDUP_tlh(MAXNUP_tlh),
     # ISTUP_tlh(MAXNUP_tlh),MOTHUP_tlh(2,MAXNUP_tlh),
     # ICOLUP_tlh(2,MAXNUP_tlh)
      DOUBLE PRECISION XWGTUP_tlh,SCALUP_tlh,AQEDUP_tlh,AQCDUP_tlh,
     # PUP_tlh(5,MAXNUP_tlh),VTIMUP_tlh(MAXNUP_tlh),
     # SPINUP_tlh(MAXNUP_tlh)
      DOUBLE PRECISION EVNORM
      COMMON/CEVNORM/EVNORM
      character*140 buff_tlh
      common/cbuff_tlh/buff_tlh
      INTEGER KK
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
      REAL*8 WWW0
c Keep lepton pairs whose invariant masses are closer than PAIRWIDTH
c to the Z pole mass
      DOUBLE PRECISION PAIRWIDTH
      DATA PAIRWIDTH/10.D0/

      integer pychge
      external pydata

      common/pyjets/n,npad,k(4000,5),p(4000,5),v(4000,5)
      common/pydat1/mstu(200),paru(200),mstj(200),parj(200)
      common/pydat2/kchg(500,4),pmas(500,4),parf(2000),vckm(4,4)
      common/pydat3/mdcy(500,3),mdme(8000,2),brat(8000),kfdp(8000,5)
      common/pysubs/msel,mselpd,msub(500),kfin(2,-40:40),ckin(200)
      common/pypars/mstp(200),parp(200),msti(200),pari(200)

      DOUBLE PRECISION EVWEIGHT
      COMMON/CEVWEIGHT/EVWEIGHT
      INTEGER IFAIL
      COMMON/CIFAIL/IFAIL
      integer lhefile,izero
      parameter (lhefile=98)
      parameter (izero=0)
c
C--RETURN IF FAILURE
      IF(IFAIL.EQ.1) RETURN
c
      DO J=1,5
         PHEP(J,1)=P(1,J)
         PHEP(J,2)=P(2,J)
      ENDDO
c
      WWW0=EVWEIGHT
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=PYCHGE(K(1,2))+PYCHGE(K(2,2))
      ILEP(1,1)=0
      ILEP(1,2)=0
      ILEP(2,1)=0
      ILEP(2,2)=0
C
      DO 100 IHEP=1,N
        DO J=1,5
          PHEP(J,IHEP)=P(IHEP,J)
        ENDDO
        IF (K(IHEP,1).LE.10) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+PYCHGE(K(IHEP,2))
        ENDIF
        IST=K(IHEP,1)
        ID1=K(IHEP,2)
C LOOK FOR FINAL-STATE ELECTRONS OR MUONS
C (I,J)==(TYPE,CHARGE)
C TYPE=1,2 -> ELECTRON, MUONS
C CHARGE=1,2 -> NEGATIVE, POSITIVE
        IF( IST.LE.10.AND.ID1.EQ.11 )THEN
          ILEP(1,1)=ILEP(1,1)+1
          IWHERE(ILEP(1,1),1,1)=IHEP
        ELSEIF( IST.LE.10.AND.ID1.EQ.-11 )THEN
          ILEP(1,2)=ILEP(1,2)+1
          IWHERE(ILEP(1,2),1,2)=IHEP
        ELSEIF( IST.LE.10.AND.ID1.EQ.13 )THEN
          ILEP(2,1)=ILEP(2,1)+1
          IWHERE(ILEP(2,1),2,1)=IHEP
        ELSEIF( IST.LE.10.AND.ID1.EQ.-13 )THEN
          ILEP(2,2)=ILEP(2,2)+1
          IWHERE(ILEP(2,2),2,2)=IHEP
        ENDIF
  100 CONTINUE
C 
      IF( ILEP(1,1).LT.1.OR.ILEP(1,2).LT.1 .OR.
     #    ILEP(2,1).LT.1.OR.ILEP(2,2).LT.1 )THEN
         write(*,*)'error 500 in pyanal'
         write(*,*)ILEP(1,1),ILEP(1,2),ILEP(2,1),ILEP(2,2)
         stop
      ENDIF
C ORDER LEPTONS BY HARDNESS. 
C IORD(K,I,J) IS THE K^th HARDEST OF TYPE (I,J)
      DO I=1,2
        DO J=1,2
          IF(ILEP(I,J).EQ.1)THEN
            IORD(1,I,J)=1
          ELSEIF(ILEP(I,J).EQ.2)THEN
            PT1=GETPTV4(PHEP(1,IWHERE(1,I,J)))
            PT2=GETPTV4(PHEP(1,IWHERE(2,I,J)))
            IF(PT1.GT.PT2)THEN
              IORD(1,I,J)=1
              IORD(2,I,J)=2
            ELSE
              IORD(1,I,J)=2
              IORD(2,I,J)=1
            ENDIF
          ELSE
            DO KKK=1,ILEP(I,J)
              IORD(KKK,I,J)=KKK
            ENDDO
            DO KKK=ILEP(I,J),2,-1
              DO L=1,KKK-1
                L1=L+1
                PT1=GETPTV4(PHEP(1,IWHERE(IORD(L,I,J),I,J)))
                PT2=GETPTV4(PHEP(1,IWHERE(IORD(L1,I,J),I,J)))
                IF(PT1.LT.PT2)THEN
                  ITMP=IORD(L,I,J)
                  IORD(L,I,J)=IORD(L1,I,J)
                  IORD(L1,I,J)=ITMP
                ENDIF
              ENDDO
            ENDDO
          ENDIF
        ENDDO
      ENDDO
C KEEP THE HARDEST FOR EACH (TYPE,CHARGE) PAIR. CONVENTIONS:
C  1 -> E-
C  2 -> E+
C  3 -> MU-
C  4 -> MU+
      DO I=1,4
        PLEP(I,1)=PHEP(I,IWHERE(IORD(1,1,1),1,1))
        PLEP(I,2)=PHEP(I,IWHERE(IORD(1,1,2),1,2))
        PLEP(I,3)=PHEP(I,IWHERE(IORD(1,2,1),2,1))
        PLEP(I,4)=PHEP(I,IWHERE(IORD(1,2,2),2,2))
      ENDDO
c
c Begin of analysis-dependent LH stuff
c
      NUP_tlh=4
      DO I=1,NUP_tlh
        DO J=1,4
          PUP_tlh(J,I)=PLEP(J,I)
        ENDDO
        PUP_tlh(5,I)=0.D0
        ISTUP_tlh(I)=1
      ENDDO
      IDUP_tlh(1)=11
      IDUP_tlh(2)=-11
      IDUP_tlh(3)=13
      IDUP_tlh(4)=-13
c
c End of analysis-dependent LH stuff
c
      IDPRUP_tlh=izero
      XWGTUP_tlh=EVWGT*EVNORM
      SCALUP_tlh=EMSCA
      AQEDUP_tlh=ZERO
      AQCDUP_tlh=ZERO
      DO I=1,NUP_tlh
        MOTHUP_tlh(1,I)=izero
        ICOLUP_tlh(2,I)=izero
        MOTHUP_tlh(1,I)=izero
        ICOLUP_tlh(2,I)=izero
        VTIMUP_tlh(I)=ZERO
        SPINUP_tlh(I)=ZERO
      ENDDO
      call write_lhef_event(lhefile,
     #   NUP_tlh,IDPRUP_tlh,XWGTUP_tlh,SCALUP_tlh,AQEDUP_tlh,
     #   AQCDUP_tlh,IDUP_tlh,ISTUP_tlh,MOTHUP_tlh,ICOLUP_tlh,
     #   PUP_tlh,VTIMUP_tlh,SPINUP_tlh,buff_tlh)
 999  END

      function getptv4(p)
      implicit none
      real*8 getptv4,p(4)
c
      getptv4=sqrt(p(1)**2+p(2)**2)
      return
      end
