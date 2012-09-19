c Create LH ntuples for ZZ->eemumu
C----------------------------------------------------------------------
      SUBROUTINE RCLOS()
C     DUMMY IF HBOOK IS USED
C----------------------------------------------------------------------
      END


C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      INTEGER MAXPUP
      PARAMETER(MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON /HEPRUP/ IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &                IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),
     &                XMAXUP(MAXPUP),LPRUP(MAXPUP)
      INTEGER MQQ
      COMMON/cMQQ/MQQ
      DOUBLE PRECISION EVNORM
      COMMON/CEVNORM/EVNORM
      character*10 MonteCarlo
      parameter (MonteCarlo="HERWIG6")
      integer lhefile,nevents
      parameter (lhefile=98)
c
      if(MQQ.eq.0)then
        write(*,*)'MQQ=',MQQ
        CALL HWWARN('HWABEG',500)
      endif
      IF(ABS(IDWTUP).EQ.1.OR.ABS(IDWTUP).EQ.2.OR.
     #   ABS(IDWTUP).EQ.4) THEN
        EVNORM = 1.D3/DFLOAT(MQQ)
      ELSEIF(ABS(IDWTUP).EQ.3) THEN
        EVNORM = ONE/DFLOAT(MQQ)
      ELSE
        CALL HWWARN('HWABEG',501)
      ENDIF
C
      OPEN(UNIT=lhefile,NAME='ZZ6.lhe',STATUS='UNKNOWN')
c Number of events actually stored will depend on failure rate.
c Use a negative value as upper bound
      nevents=-MAXEV
      call write_lhef_header(lhefile,nevents,MonteCarlo)
      call write_lhef_init(lhefile,
     &     IDBMUP,EBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,
     &     XSECUP,XERRUP,XMAXUP,LPRUP)
c
 999  END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
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
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      INTEGER IHEP,IST,ID,ID1,K,L,L1,ITMP,
     # ILEP(2,2),IWHERE(30,2,2),IORD(30,2,2)
      DOUBLE PRECISION PT1,GETPTV4,PT2,PLEP(4,4)
      INTEGER I,J
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
      integer lhefile,izero
      parameter (lhefile=98)
      parameter (izero=0)
c
      IF (IERROR.NE.0) RETURN
c
c Begin of part extracted from mcatnlo_hwanZZ6.f
c
      ILEP(1,1)=0
      ILEP(1,2)=0
      ILEP(2,1)=0
      ILEP(2,2)=0
c
      DO 100 IHEP=1,NHEP
        IST=ISTHEP(IHEP)      
        ID=IDHW(IHEP)
        ID1=IDHEP(IHEP)
C LOOK FOR FINAL-STATE ELECTRONS OR MUONS
C (I,J)==(TYPE,CHARGE)
C TYPE=1,2 -> ELECTRON, MUONS
C CHARGE=1,2 -> NEGATIVE, POSITIVE
        IF( IST.EQ.1.AND.ID1.EQ.11 )THEN
          ILEP(1,1)=ILEP(1,1)+1
          IWHERE(ILEP(1,1),1,1)=IHEP
        ELSEIF( IST.EQ.1.AND.ID1.EQ.-11 )THEN
          ILEP(1,2)=ILEP(1,2)+1
          IWHERE(ILEP(1,2),1,2)=IHEP
        ELSEIF( IST.EQ.1.AND.ID1.EQ.13 )THEN
          ILEP(2,1)=ILEP(2,1)+1
          IWHERE(ILEP(2,1),2,1)=IHEP
        ELSEIF( IST.EQ.1.AND.ID1.EQ.-13 )THEN
          ILEP(2,2)=ILEP(2,2)+1
          IWHERE(ILEP(2,2),2,2)=IHEP
        ENDIF
  100 CONTINUE
C 
      IF( ILEP(1,1).LT.1.OR.ILEP(1,2).LT.1 .OR.
     #    ILEP(2,1).LT.1.OR.ILEP(2,2).LT.1 )THEN
        CALL HWUEPR
        CALL HWWARN('HWANAL',500)
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
            DO K=1,ILEP(I,J)
              IORD(K,I,J)=K
            ENDDO
            DO K=ILEP(I,J),2,-1
              DO L=1,K-1
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
c End of part extracted from mcatnlo_hwanZZ6.f
c
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
c
 999  END


      function getptv4(p)
      implicit none
      real*8 getptv4,p(4)
c
      getptv4=sqrt(p(1)**2+p(2)**2)
      return
      end
