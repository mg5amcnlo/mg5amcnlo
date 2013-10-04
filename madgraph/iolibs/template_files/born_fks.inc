      SUBROUTINE SBORN(P1,ANS)
C  
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C  
C  
C BORN AMPLITUDE IS 
%(process_lines)s
C
      IMPLICIT NONE
C  
C CONSTANTS
C  
      Include 'genps.inc'
      include "nexternal.inc"
c      Include 'born_maxamps.inc'
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB=  %(ncomb)d, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS*%(nconfs)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(NCROSS*2)
C  
C LOCAL VARIABLES 
C  
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY(%(nconfs)d)
      COMPLEX*16 T,T1
      REAL*8 BORN
      REAL*8 ZERO
      PARAMETER(ZERO=0d0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS,%(nconfs)d)
      INTEGER NGOOD(%(nconfs)d),igood(ncomb,%(nconfs)d),jhel
      data ngood /%(nconfs)d*0/
      save igood,jhel
      REAL*8 hwgt
      REAL*8 xtot, xtry, xrej, xr, yfrac(0:ncomb)
      INTEGER j, jj
      LOGICAL warned
      REAL     xran1
      EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2

      include "born_nhel.inc"
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom

      character*79         hel_buff(2)
      common/to_helicity/  hel_buff

      REAL*8 POL(2)
      common/to_polarization/ POL

      integer          isum_hel
      logical                    multi_channel
      common/to_matrix/isum_hel, multi_channel
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      common/to_mconfigs/mapconfig, iconfig
      DATA NTRY /%(nconfs)d*0/
      DATA xtry, xrej /0,0/
      SAVE yfrac
      DATA jamp2(0) /   %(ncolor)d/          
      DATA GOODHEL/THEL*.FALSE./
  %(helicity_lines)s
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer glu_ij
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
%(den_factor_lines)s
%(ij_lines)s
C ----------
C BEGIN CODE
C ----------
      iden(1)=iden_values(nfksprocess)
      glu_ij = ij_values(nfksprocess)
      NTRY(nFKSprocess)=NTRY(nFKSprocess)+1
      if (NTRY(nFKSprocess).lt.2) then
         skip(nFKSprocess)=1			 
         do while(nhel(glu_ij ,skip(nFKSprocess)).ne.1)
         skip(nFKSprocess)=skip(nFKSprocess)+1
         enddo
         skip(nFKSprocess)=skip(nFKSprocess)-1
      endif
      DO IPROC=1,NCROSS
      DO IHEL=1,NEXTERNAL-1
         JC(IHEL) = +1
      ENDDO
       
      DO IHEL=1,NGRAPHS
          amp2(ihel)=0d0
      ENDDO
      DO IHEL=1,int(jamp2(0))
          jamp2(ihel)=0d0
      ENDDO
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p1(0,j) .or. savemom(j,2).ne.p1(3,j)) then
               calculatedBorn=.false.
               write (*,*) "momenta not the same in Born"
	       stop
            endif
         enddo
      endif
      if (.not.calculatedBorn) then
         do j=1,nexternal-1
            savemom(j,1)=p1(0,j)
            savemom(j,2)=p1(3,j)
         enddo
         do j=1,max_bhel
            do jj=1,ngraphs
               saveamp(jj,j)=(0d0,0d0)
            enddo
         enddo
      endif
      ANS(IPROC) = 0D0
      ANS(IPROC+1) = 0D0
      write(hel_buff(1),'(16i5)') (0,i=1,nexternal-1)
      IF (ISUM_HEL .EQ. 0 .OR. NTRY(nFKSprocess) .LT. 2) THEN
             hel_fac=1d0
          DO IHEL=1,NCOMB
             IF ((GOODHEL(IHEL,IPROC,nFKSprocess) .OR. NTRY(nFKSprocess) .LT. 2).and.nhel(glu_ij ,ihel).eq.-1) THEN
                 T=BORN(P1,NHEL(1,IHEL),IHEL,JC(1),T1)            
               DO JJ=1,nincoming
                 IF(POL(JJ).NE.1d0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                   T=T*ABS(POL(JJ))
                   T1=T1*ABS(POL(JJ))
                 ELSE IF(POL(JJ).NE.1d0)THEN
                   T=T*(2d0-ABS(POL(JJ)))
                   T1=T1*(2d0-ABS(POL(JJ)))
                 ENDIF
               ENDDO
               ANS(IPROC)=ANS(IPROC)+T
               ANS(IPROC+1)=ANS(IPROC+1)+T1
               IF ( (T .NE. 0D0 .OR. T1 .NE. 0d0) .AND. .NOT. GOODHEL(IHEL,IPROC,nFKSprocess)) THEN
                   GOODHEL(IHEL,IPROC,nFKSprocess)=.TRUE.
                   NGOOD(nFKSprocess) = NGOOD(nFKSprocess) +1
                   IGOOD(NGOOD(nFKSprocess),nFKSprocess) = IHEL
               ENDIF
             ENDIF
          ENDDO
          JHEL = 1
          ISUM_HEL=MIN(ISUM_HEL,NGOOD(nFKSprocess))
      ELSE              !RANDOM HELICITY
          DO J=1,ISUM_HEL
              HWGT = REAL(NGOOD(nFKSprocess))/REAL(ISUM_HEL)
              hel_fac=HWGT
              if (get_hel.eq.0) then
              JHEL=JHEL+1
              IF (JHEL .GT. NGOOD(nFKSprocess)) JHEL=1
                 IHEL = IGOOD(JHEL,nFKSprocess)
                 get_hel=ihel
              else
                 ihel=get_hel
              endif
              IF(GOODHEL(IHEL,IPROC,nFKSprocess)) THEN
              T=BORN(P1,NHEL(1,IHEL),IHEL,JC(1),T1)            
              DO JJ=1,nincoming
                IF(POL(JJ).NE.1d0.AND. NHEL(JJ,IHEL).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                  T=T*ABS(POL(JJ))
                   T1=T1*ABS(POL(JJ))
                ELSE IF(POL(JJ).NE.1d0)THEN
                  T=T*(2d0-ABS(POL(JJ)))
                   T1=T1*(2d0-ABS(POL(JJ)))
                ENDIF
              ENDDO
              ANS(IPROC)=ANS(IPROC)+T*HWGT
              ANS(IPROC+1)=ANS(IPROC+1)+T1*HWGT
          ENDIF
          ENDDO
          IF (ISUM_HEL .EQ. 1) THEN
              WRITE(HEL_BUFF(1),'(16i5)')(NHEL(i,IHEL),i=1,nexternal-1)
              WRITE(HEL_BUFF(2),'(16i5)')(NHEL(i,IHEL+SKIP(nFKSprocess)),i=1,nexternal-1)
          ENDIF
      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ANS(IPROC+1)=ANS(IPROC+1)/DBLE(IDEN(IPROC))
      ENDDO
      calculatedBorn=.true.
      END
       
       
      REAL*8 FUNCTION BORN(P,NHEL,HELL,IC,BORNTILDE)
C  
%(info_lines)s
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C FOR THE POINT WITH EXTERNAL LINES W(0:6,NEXTERNAL-1)

%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS,    NEIGEN 
      PARAMETER (NGRAPHS=   %(ngraphs)d,NEIGEN=  1) 
      include 'genps.inc'
      include "nexternal.inc"
c      INCLUDE 'born_maxamps.inc'
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1), IC(NEXTERNAL-1), HELL
      complex *16 borntilde
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(%(wavefunctionsize)d,NWAVEFUNCS)
      complex*16 imag1
      integer ihel, back_hel
      parameter (imag1 = (0d0,1d0))
      complex *16 jamph(-1:1, ncolor)
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2
      include "born_nhel.inc"
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      include "coupl.inc"
      integer glu_ij
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
%(ij_lines)s
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      glu_ij = ij_values(nfksprocess)
      BORN = 0d0
      BORNTILDE = (0d0,0d0)
      back_hel = nhel(glu_ij)
      do ihel=-1,1,2
      nhel(glu_ij) = ihel
      if (.not. calculatedBorn) then
  %(helas_calls)s
      do i=1,ngraphs
         if(ihel.eq.-1)then
            saveamp(i,hell)=amp(i)
         elseif(ihel.eq.1)then
            saveamp(i,hell+skip(nFKSprocess))=amp(i)
         else
            write(*,*) "ERROR #1 in born.f"
            stop
         endif
      enddo
      elseif (calculatedBorn) then
      do i=1,ngraphs
         if(ihel.eq.-1)then
            amp(i)=saveamp(i,hell)
         elseif(ihel.eq.1)then
            amp(i)=saveamp(i,hell+skip(nFKSprocess))
         else
            write(*,*) "ERROR #1 in born.f"
            stop
         endif
      enddo
      endif
  %(jamp_lines)s
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          BORN =BORN+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
      ENDDO
      Do I = 1, NGRAPHS
          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
      Enddo
      Do I = 1, NCOLOR
          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
          Jamph(ihel,i)=Jamp(i)
      Enddo
       
      Enddo
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMPH(1,J)
          ENDDO
          BORNTILDE = BORNTILDE + ZTEMP*DCONJG(JAMPH(-1,I))/DENOM(I)
      ENDDO
      nhel(glu_ij) = back_hel
      END
       
       
