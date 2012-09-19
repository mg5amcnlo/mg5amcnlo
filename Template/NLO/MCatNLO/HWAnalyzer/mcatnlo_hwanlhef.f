c Generic routines for storing events in Les Houches format
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
      OPEN(UNIT=lhefile,NAME='generic.lhe',STATUS='UNKNOWN')
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
      integer lhefile 
      parameter (lhefile=98)
c
      write(lhefile,*)'</LesHouchesEvents>'
      CLOSE(lhefile)
 999  END


C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
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
      integer lhefile,izero,icount_tlh
      parameter (lhefile=98)
      parameter (izero=0)
c
      IF (IERROR.NE.0) RETURN
c
c Begin of analysis-dependent part. This is a trivial example
      NUP_tlh=10
      icount_tlh=0
      DO I=4,3+NUP_tlh
        icount_tlh=icount_tlh+1
        DO J=1,5
          PUP_tlh(J,icount_tlh)=PHEP(J,I)
        ENDDO
        IDUP_tlh(icount_tlh)=IDHEP(I)
        IF(I.LE.5)THEN
          ISTUP_tlh(icount_tlh)=-1
        ELSE
          ISTUP_tlh(icount_tlh)=1
        ENDIF
      ENDDO
c End of analysis-dependent part
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
