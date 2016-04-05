C----------------------------------------------------------------------
      SUBROUTINE UPEVNT
C----------------------------------------------------------------------
C  Reads MC@NLO input files and fills Les Houches event common HEPEUP
C  Event file is written by MadFKS
C  Mostly derived from read_lhef_event() in handling_lhe_events.f
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
C---Les Houches Event Common Block
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &              ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &              SPINUP(MAXNUP)
      INTEGER ISORH_LHE,IFKS_LHE,JFKS_LHE,FKSFATHER_LHE,IPARTNER_LHE
      DOUBLE PRECISION SCALE1_LHE,SCALE2_LHE
      DOUBLE PRECISION WGTCENTRAL,WGTMUMIN,WGTMUMAX,WGTPDFMIN,WGTPDFMAX
      INTEGER MQQ
      COMMON/cMQQ/MQQ
      INTEGER IUNIT
      PARAMETER (IUNIT=61)
      CHARACTER*80 STRING
      CHARACTER*140 BUFF
      character*20 cidwgt
      character*12 dummy12
      character*2 dummy2
      CHARACTER*9 CH1
      INTEGER I,J,II,NPS,NNG,idwgt
      character*140 buff_tlh
      common/cbuff_tlh/buff_tlh
c evwgt_lh is meant to be passed to stdhep
      DOUBLE PRECISION EVWGT_LH
      COMMON/CEVWGT_LH/EVWGT_LH
      include 'reweight0.inc'
      integer iww
      double precision ww(max_weight_shower)
      common/cww/ww
      integer nwgt
      common/cnwgt/nwgt
C
      IF (IERROR.NE.0) RETURN
c
      ISORH_LHE=0
c Find the start of the events
      do while (.true.)
         read(iunit,'(a)')string
         if(INDEX(STRING,'<event').ne.0) exit
      enddo
      read(iunit,*)NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP
C---Les Houches expects mean weight to be the cross section in pb
      EVWGT_LH=XWGTUP
      XWGTUP=XWGTUP*MQQ
      do i=1,nup
        read(iunit,*)IDUP(I),ISTUP(I),MOTHUP(1,I),MOTHUP(2,I),
     #                 ICOLUP(1,I),ICOLUP(2,I),
     #                 PUP(1,I),PUP(2,I),PUP(3,I),PUP(4,I),PUP(5,I),
     #                 VTIMUP(I),SPINUP(I)
c Avoids rounding problems for zero-mass particles
        if(pup(5,i).eq.0.d0.and.istup(i).eq.1)then
          pup(4,i)=pup(1,i)**2+pup(2,i)**2+pup(3,i)**2
          pup(4,i)=sqrt(pup(4,i))
        endif
      enddo
      iww=1
      ww(iww)=XWGTUP/MQQ
      read(iunit,'(a)')buff
      if(buff(1:1).eq.'#')then
        buff_tlh=buff
        read(buff,*)ch1,iSorH_lhe,ifks_lhe,jfks_lhe,
     #                    fksfather_lhe,ipartner_lhe,
     #                    scale1_lhe,scale2_lhe,
     #                    jwgtinfo,mexternal,iwgtnumpartn,
     #         wgtcentral,wgtmumin,wgtmumax,wgtpdfmin,wgtpdfmax
        if(jwgtinfo.eq.9)then
          if (nwgt.gt.1) then
             read(iunit,'(a)')string ! <rwgt>
             wgtref=XWGTUP/MQQ
             do iww=2,nwgt      ! start at 2, because 'central value' is not part of the extra weights
                call read_rwgt_line_wgt(iunit,ww(iww))
             enddo
             read(iunit,'(a)')string ! </rwgt>
          endif
        else
          do while(index(string,'</event>').eq.0)
             read(iunit,'(a)')string
          enddo
          backspace(iunit)
        endif
        read(iunit,'(a)')string
      else
        string=buff(1:len_trim(buff))
        buff_tlh=' '
      endif
      if(INDEX(STRING,'</event>').eq.0)then
        write(*,*)'FATAL ERROR #2 IN UPEVNT'
        stop
      endif
c Modify what follows to set scale of H or S events in a different way
c$$$      IF(ISORH_LHE.EQ.2)THEN
c$$$c H events
c$$$        IF(SCALE2_LHE.GT.0.D0)SCALUP=SCALE2_LHE
c$$$      ENDIF
 401  format(2(1x,e14.8))
 402  format(8(1x,e14.8))
 403  format(6(1x,e14.8))
 404  format(3(1x,e14.8))
 405  format(4(1x,e14.8))
 406  format(2(1x,e14.8),2(1x,i3))
 441  format(4(1x,e16.10))
 442  format(1x,e16.10,2(1x,e14.8))
c 503  format(1x,i2,1x,i6,4(1x,d14.8))
c 504  format(1x,i8,1x,i2,4(1x,i4),5(1x,d14.8),2(1x,d10.4))
      END


C----------------------------------------------------------------------
      SUBROUTINE UPINIT
C----------------------------------------------------------------------
C  Reads MC@NLO input headers and fills Les Houches run common HEPRUP
C  Event file is written by MadFKS
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
C--Les Houches Common Blocks
      INTEGER MAXPUP
      PARAMETER(MAXPUP=100)
      INTEGER IDBMUP,PDFGUP,PDFSUP,IDWTUP,NPRUP,LPRUP
      DOUBLE PRECISION EBMUP,XSECUP,XERRUP,XMAXUP
      COMMON /HEPRUP/ IDBMUP(2),EBMUP(2),PDFGUP(2),PDFSUP(2),
     &                IDWTUP,NPRUP,XSECUP(MAXPUP),XERRUP(MAXPUP),
     &                XMAXUP(MAXPUP),LPRUP(MAXPUP)
      INTEGER MAXNUP
      PARAMETER (MAXNUP=500)
      INTEGER NUP,IDPRUP,IDUP,ISTUP,MOTHUP,ICOLUP
      DOUBLE PRECISION XWGTUP,SCALUP,AQEDUP,AQCDUP,PUP,VTIMUP,SPINUP
      COMMON/HEPEUP/NUP,IDPRUP,XWGTUP,SCALUP,AQEDUP,AQCDUP,
     &              IDUP(MAXNUP),ISTUP(MAXNUP),MOTHUP(2,MAXNUP),
     &              ICOLUP(2,MAXNUP),PUP(5,MAXNUP),VTIMUP(MAXNUP),
     &              SPINUP(MAXNUP)
c Hard event file (to be entered in Herwig driver)
      integer i
      CHARACTER*50 QQIN
      COMMON/VVJIN/QQIN
      CHARACTER*200 STRING
      include 'reweight0.inc'
      integer nwgt
      common/cnwgt/nwgt
      character*50 weights_info(max_weight_shower)
      common/cwgtsinfo/weights_info
      
      nwgt=1
      weights_info(nwgt)="central value"
      IF (IERROR.NE.0) RETURN
C--SET UP INPUT FILES
      OPEN(UNIT=61,FILE=QQIN,STATUS='UNKNOWN')
C--Read (non compulsory) headers here if need be. Look for extra
C--weights.
      do
         READ(61,'(a)')STRING
         if (index(string,'<initrwgt>').ne.0) then
c Found extra weight labels:  
            do
               READ(61,'(a)')STRING
c     exit when end of weight information is found
               IF ( INDEX(STRING,'</initrwgt>').ne.0 .and.
     &              STRING(1:1).ne.'#' ) exit
c     skip comments and weightgroup lines. We just need to copy this
c     information
               if ( index(string,"<weightgroup").ne.0 .or.
     &              index(string,"</weightgroup>").ne.0 .or.
     &              string(1:1).eq.'#') cycle
               if (INDEX(STRING,"<weight id").ne.0)then
                  nwgt=nwgt+1
                  read(string(index(string,"'>")+2: index(string
     $                 ,"</weight>")-1),'(a)') weights_info(nwgt)
                  if (nwgt.gt.max_weight_shower) then
                     write (*,*) 'Too many weights in event file. '/
     $                    /'Increase max_weight_shower'
                     stop
                  endif
               endif
            enddo
         ELSEIF ( INDEX(STRING,'</header>').ne.0 .and.
     &        STRING(1:1).ne.'#' ) then
            EXIT
         ENDIF
      ENDDO
c--Find the start of the <init> block
      DO WHILE(.TRUE.)
         READ(61,'(a)')STRING
         IF ( INDEX(STRING,'<init>').ne.0 .and.
     &        STRING(1:1).ne.'#' ) exit
      ENDDO
C--Read up to </init> in the event file
      read(61,*,err=998)IDBMUP(1),IDBMUP(2),EBMUP(1),EBMUP(2),
     #            PDFGUP(1),PDFGUP(2),PDFSUP(1),PDFSUP(2),
     #            IDWTUP,NPRUP
      do i=1,NPRUP
         read(61,*,err=998)XSECUP(i),XERRUP(i),XMAXUP(i),LPRUP(i)
      enddo
 111  format(a4,f3.1,x,a4,f3.1)
 112  format(a4,i8,a3)
 113  format(a15)
      return
 998  write(*,*)'FATAL ERROR #2 IN UPINIT'
      stop
 999  END


C----------------------------------------------------------------------
      SUBROUTINE HWURSC(NP,PP)
C  RESCALES A SET OF NP (<21) 3-MOMENTA PP(1-3,*) IN
C  THEIR CMF TO PUT PP ON MASS-SHELL AT MASSES PP(5,*) 
C----------------------------------------------------------------------
      IMPLICIT NONE
      INTEGER NP,IP,IT,NT
      DOUBLE PRECISION PP(5,*),P(5,20),P2(20),M2(20),SP(5),
     & TINY,FAC,ECM,DCM,EP,STEP,FRT,HWUSQR
      DATA TINY,NT/1D-9,20/
      IF (NP.GT.20)THEN
        write(*,*)'FATAL ERROR #1 IN HWURSC'
        stop
      ENDIF
C--COMPUTE CM MOMENTUM
      CALL HWVZRO(4,SP)
      DO IP=1,NP
         CALL HWVSUM(4,PP(1,IP),SP,SP)
      ENDDO
      CALL HWUMAS(SP)
C--BOOST TO CMF
      DO IP=1,NP
         CALL HWULOF(SP,PP(1,IP),P(1,IP))
         P2(IP)=P(1,IP)**2+P(2,IP)**2+P(3,IP)**2
         M2(IP)=P(5,IP)**2
      ENDDO
C--ITERATE RESCALING OF 3-MOMENTA
      FAC=1D0
      DO IT=1,NT
         ECM=0D0
         DCM=0D0
         DO IP=1,NP
            EP=HWUSQR(M2(IP)+FAC*P2(IP))
            IF (EP.GT.0D0) THEN
               ECM=ECM+EP
               DCM=DCM+P2(IP)/EP
            ENDIF
         ENDDO
         IF (DCM.EQ.0D0)THEN
            write(*,*)'FATAL ERROR #2 IN HWURSC'
            stop
         ENDIF
         STEP=2D0*(ECM-SP(5))/DCM
         FAC=FAC-STEP
         IF (ABS(STEP).LT.TINY) GOTO 100
      ENDDO
C--FAILED TO CONVERGE
        write(*,*)'WARNING #1 IN HWURSC'
C--CONVERGED: RESCALE 3-MOMENTA AND BOOST BACK 
 100  IF (FAC.LT.0D0)THEN
         write(*,*)'FATAL ERROR #3 IN HWURSC'
         stop
      ENDIF
      FRT=SQRT(FAC)
      DO IP=1,NP
         CALL HWVSCA(3,FRT,P(1,IP),P(1,IP))
         P(4,IP)=SQRT(M2(IP)+FAC*P2(IP))
         CALL HWULOB(SP,P(1,IP),PP(1,IP))
      ENDDO
      END


      subroutine read_rwgt_line_wgt(unit,wgt)
c read a line in the <rwgt> tag. The syntax should be
c  <wgt id='1001'> 0.1234567e+01 </wgt>
c The id should be exactly 4 digits long.
      implicit none
      integer unit,wgt_start
      double precision wgt
      character*100 buff
      read (unit,'(a)') buff
c Use char() to make sure that the non-standard characters are compiler
c independent (char(62)=">", char(61)="=", char(39)="'")
      wgt_start=index(buff,CHAR(39)//CHAR(62))+2
      read (buff(wgt_start:100),*) wgt
      return
      end
