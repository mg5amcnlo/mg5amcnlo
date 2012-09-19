      subroutine rclos()
      end

C----------------------------------------------------------------------
      SUBROUTINE HWABEG
C     USER'S ROUTINE FOR INITIALIZATION
C----------------------------------------------------------------------
      real * 4 pi
      parameter (pi=3.14160E0)
c
      call inihist 
      call mbook(1,'Thrust',0.05e0,-0.05e0,1.05e0)
      call mbook(2,'Thrust Major',0.05e0,-0.05e0,1.05e0)
      call mbook(3,'Thrust Minor',0.05e0,-0.05e0,1.05e0)
      call mbook(4,'C parameter',0.05e0,-0.05e0,1.05e0)
      call mbook(5,'D parameter',0.05e0,-0.05e0,1.05e0)
c
      call mbook(11,'Thrust',0.01e0,0.0e0,1.0e0)
      call mbook(12,'Thrust Major',0.01e0,0.0e0,1.0e0)
      call mbook(13,'Thrust Minor',0.01e0,0.0e0,1.0e0)
      call mbook(14,'C parameter',0.01e0,0.0e0,1.0e0)
      call mbook(15,'D parameter',0.01e0,0.0e0,1.0e0)
 999  END


C----------------------------------------------------------------------
      SUBROUTINE HWAEND
C     USER'S ROUTINE FOR TERMINAL CALCULATIONS, HISTOGRAM OUTPUT, ETC
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      REAL*8 XNORM
      INTEGER I
C
      OPEN(UNIT=99,NAME='HEREE.TOP',STATUS='UNKNOWN')
C XNORM IS SUCH THAT THE CROSS SECTION PER BIN IS IN PB, SINCE THE HERWIG 
C WEIGHT IS IN NB, AND CORRESPONDS TO THE AVERAGE CROSS SECTION
      XNORM=1.D3/DFLOAT(NEVHEP)
      DO I=1,100              
 	CALL MFINAL3(I)             
        CALL MCOPY(I,I+100)
        CALL MOPERA(I+100,'F',I+100,I+100,SNGL(XNORM),0.E0)
 	CALL MFINAL3(I+100)             
      ENDDO                          
C
      call multitop(101,99,2,2,'T',' ','LOG')
      call multitop(102,99,2,2,'T max',' ','LOG')
      call multitop(103,99,2,2,'T min',' ','LOG')
      call multitop(104,99,2,2,'C',' ','LOG')
      call multitop(105,99,2,2,'D',' ','LOG')
c
      call multitop(111,99,2,2,'T',' ','LOG')
      call multitop(112,99,2,2,'T max',' ','LOG')
      call multitop(113,99,2,2,'T min',' ','LOG')
      call multitop(114,99,2,2,'C',' ','LOG')
      call multitop(115,99,2,2,'D',' ','LOG')
c
      CLOSE(99)
      END

C----------------------------------------------------------------------
      SUBROUTINE HWANAL
C     USER'S ROUTINE TO ANALYSE DATA FROM EVENT
C----------------------------------------------------------------------
      INCLUDE 'HERWIG65.INC'
      DOUBLE PRECISION HWVDOT,PSUM(4)
      INTEGER IHEP,JPR,ICHSUM,ICHINI,NMAX,NN,I,J
      LOGICAL DIDSOF
      PARAMETER (NMAX=1000)
      DOUBLE PRECISION PP(4,NMAX)
      REAL*8 WWW0
      REAL*8 PI
      PARAMETER (PI=3.14159265358979312D0)
c Variables for Dissertori's routines
      real*4 xp(4,nmax),thrust(4,3),cpar,dpar,var
      logical miss
      parameter (miss=.false.)
c
      IF (IERROR.NE.0) RETURN
c
c      IF(IPROC.GT.0)CALL HWWARN('HWANAL',500)
c      JPR=MOD(ABS(IPROC),10000)/100
c      IF(JPR.NE.1)CALL HWWARN('HWANAL',502)
      WWW0=EVWGT
      CALL HWVSUM(4,PHEP(1,1),PHEP(1,2),PSUM)
      CALL HWVSCA(4,-1D0,PSUM,PSUM)
      ICHSUM=0
      ICHINI=ICHRG(IDHW(1))+ICHRG(IDHW(2))
      DIDSOF=.FALSE.
c NN is the number of final-state hadrons to be cluster into jets;
c their four momenta are PP(1-4,NN)
      NN=0
      DO 100 IHEP=1,NHEP
        IF (IDHW(IHEP).EQ.16) DIDSOF=.TRUE.
        IF (ISTHEP(IHEP).EQ.1) THEN
          CALL HWVSUM(4,PHEP(1,IHEP),PSUM,PSUM)
          ICHSUM=ICHSUM+ICHRG(IDHW(IHEP))
        ENDIF
C---FIND FINAL STATE HADRONS
        IF (ISTHEP(IHEP).EQ.1 .AND. ABS(IDHEP(IHEP)).GT.100) THEN
          NN=NN+1
          IF (NN.GT.NMAX) STOP 'Too many particles!'
          DO I=1,4
             PP(I,NN)=PHEP(I,IHEP)
          ENDDO
        ENDIF
  100 CONTINUE
C CHECK MOMENTUM AND CHARGE CONSERVATION
      IF (HWVDOT(3,PSUM,PSUM).GT.1.E-4*PHEP(4,1)**2) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',112)
         GOTO 999
      ENDIF
      IF (ICHSUM.NE.ICHINI) THEN
         CALL HWUEPR
         CALL HWWARN('HWANAL',113)
         GOTO 999
      ENDIF
C
      do i=1,nn
        do j=1,4
          xp(j,i)=sngl(pp(j,i))
        enddo
      enddo
c Thrust: thrust(4,#) is thrust, thrust major, and thrust minor
c for #1,2,3 respectively
      call ThrustDG(xp,nn,thrust,miss)
c C parameter, D parameter  
      call Cparameter(xp,nn,cpar,dpar)
c
      call oflufl(thrust(4,1),var)
      call mfill(1,var,sngl(WWW0))
      call mfill(11,var,sngl(WWW0))
      call oflufl(thrust(4,2),var)
      call mfill(2,var,sngl(WWW0))
      call mfill(12,var,sngl(WWW0))
      call oflufl(thrust(4,3),var)
      call mfill(3,var,sngl(WWW0))
      call mfill(13,var,sngl(WWW0))
      call oflufl(cpar,var)
      call mfill(4,var,sngl(WWW0))
      call mfill(14,var,sngl(WWW0))
      call oflufl(dpar,var)
      call mfill(5,var,sngl(WWW0))
      call mfill(15,var,sngl(WWW0))
c
 999  END


      subroutine oflufl(varin,varout)
c Numerical inaccuracies in shape variables routines may lead to
c results outside physical range (0,1); the proper range is forced
c in this routine if the shape variable is close enough to the
c boundaries of the physical range.
      implicit none
      real*4 varin,varout,tiny,vtiny
      parameter (tiny=1.e-4)
      parameter (vtiny=1.e-5)
c
      if(varin.ge.1.d0.and.varin.le.(1.d0+tiny))then
        varout=1-vtiny
      elseif(varin.le.0.d0.and.varin.ge.-tiny)then
        varout=vtiny
      else
        varout=varin
      endif
      return
      end


      subroutine SORTZV()
      write(*,*)'This is missing'
      stop
      end

