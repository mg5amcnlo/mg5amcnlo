      SUBROUTINE SBORN_HEL(P1,ANS)
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
      real*8 ANS
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY(%(nconfs)d)
      COMPLEX*16 T,T1
      REAL*8 BORN_HEL
      REAL*8 ZERO
      PARAMETER(ZERO=0d0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS,%(nconfs)d)
      INTEGER NGOOD(%(nconfs)d),igood(ncomb,%(nconfs)d),jhel
      data ngood /%(nconfs)d*0/
      save igood,jhel
      INTEGER j, jj
C  
C GLOBAL VARIABLES
C  

      include "born_nhel.inc"
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom
      DATA NTRY /%(nconfs)d*0/
      DATA GOODHEL/THEL*.FALSE./
  %(helicity_lines)s
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
%(den_factor_lines)s
      double precision wgt_hel(max_bhel)
      common/c_born_hel/wgt_hel

C ----------
C BEGIN CODE
C ----------
      iden(1)=iden_values(nfksprocess)
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
         write(*,*) 'Error in sb_sf: color_linked borns should be called only with calculatedborn = true'
	 stop
      endif
      ANS = 0D0
      DO IHEL=1,NCOMB
         wgt_hel(ihel)=0d0
         IF (GOODHEL(IHEL,1,nFKSprocess) .OR. NTRY(nFKSprocess) .LT. 2) THEN
           wgt_hel(ihel)=BORN_HEL(P1,NHEL(1,IHEL),IHEL,JC(1))/DBLE(IDEN(1))
           ANS=ANS+wgt_hel(ihel)
           IF ( T.NE.0D0 .AND. .NOT.GOODHEL(IHEL,1,nFKSprocess)) THEN
               GOODHEL(IHEL,1,nFKSprocess)=.TRUE.
               NGOOD(nFKSprocess) = NGOOD(nFKSprocess) +1
               IGOOD(NGOOD(nFKSprocess),nFKSprocess) = IHEL
           ENDIF
         ENDIF
      ENDDO
      END
       
       
      REAL*8 FUNCTION born_hel(P,NHEL,HELL,IC)
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
      include "nexternal.inc"
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1), IC(NEXTERNAL-1), HELL
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J
      COMPLEX*16 ZTEMP
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
      COMPLEX*16 W(18,NWAVEFUNCS)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
C  
C GLOBAL VARIABLES
C  
      include "born_nhel.inc"
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      include "coupl.inc"
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------

      if (.not. calculatedBorn) then
        write(*,*) 'Error in b_sf: color_linked borns should be called only with calculatedborn = true'
        stop
      elseif (calculatedBorn) then
      do i=1,ngraphs
         amp(i)=saveamp(i,hell)
      enddo
      endif
  %(jamp_lines)s
      born_hel = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(j,i)*JAMP(J)
          ENDDO
          born_hel =born_hel+ZTEMP*DCONJG(JAMP(I))/DENOM(I)
      ENDDO
      END
