      SUBROUTINE SB_SF_%(iborn)3.3d(P1,ANS)
C  
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P(0:3,NEXTERNAL-1)
C  
C  
C BORN AMPLITUDE IS 
%(process_lines)s
C
      IMPLICIT NONE
C  
C CONSTANTS
C  
      Include "nexternal.inc"
      INTEGER                 NCOMB,     NCROSS         
      PARAMETER (             NCOMB=  %(ncomb)d, NCROSS=  1)
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*NCROSS*%(nconfs)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL-1),ANS(NCROSS)
C  
C LOCAL VARIABLES 
C  
      REAL*8 P(0:3,NEXTERNAL-1)
      INTEGER NHEL(NEXTERNAL-1,NCOMB),NTRY(%(nconfs)d)
      REAL*8 T
      REAL*8 B_SF_%(iborn)3.3d
      REAL*8 ZERO
      PARAMETER(ZERO=0d0)
      INTEGER IHEL,IDEN(NCROSS),IC(NEXTERNAL-1,NCROSS)
      INTEGER IPROC,JC(NEXTERNAL-1), I,L,K
      LOGICAL GOODHEL(NCOMB,NCROSS,%(nconfs)d)
      DATA NTRY /%(nconfs)d*0/
      INTEGER NGOOD(%(nconfs)d),igood(ncomb,%(nconfs)d),jhel
      data ngood /%(nconfs)d*0/
      save igood,jhel
      REAL*8 hwgt
      integer j,jj
      include "born_nhel.inc"
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom

      character*79         hel_buff(2)
      common/to_helicity/  hel_buff

      DATA GOODHEL/THEL*.FALSE./
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
%(den_factor_lines)s
C ----------
C BEGIN CODE
C ----------
      iden(1)=iden_values(nfksprocess)
      NTRY(nfksprocess)=NTRY(nfksprocess)+1
      DO IPROC=1,NCROSS
      DO IHEL=1,NEXTERNAL-1
         JC(IHEL) = +1
      ENDDO
      if (calculatedBorn) then
         do j=1,nexternal-1
            if (savemom(j,1).ne.p1(0,j) .or. savemom(j,2).ne.p1(3,j)) then
               calculatedBorn=.false.
               write(*,*) 'Error in sb_sf: momenta not the same in the born'
               stop
            endif
         enddo
      endif
      if (.not.calculatedBorn) then
         write(*,*) 'Error in sb_sf: color_linked borns should be called only with calculatedborn = true'
         stop
      endif
      ANS(IPROC) = 0D0
      IF (GET_HEL .EQ. 0 .OR. NTRY(nFKSprocess) .LT. 2) THEN
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL,IPROC,nFKSprocess) .OR. NTRY(nFKSprocess) .LT. 2) THEN
                 T=B_SF_%(iborn)3.3d(P1,NHEL(1,IHEL),IHEL,JC(1))              
               ANS(IPROC)=ANS(IPROC)+T
               IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IHEL,IPROC,nFKSprocess)) THEN
                   GOODHEL(IHEL,IPROC,nFKSprocess)=.TRUE.
                   NGOOD(nFKSprocess) = NGOOD(nFKSprocess) +1
                   IGOOD(NGOOD(nFKSprocess),nFKSprocess) = IHEL
               ENDIF
             ENDIF
          ENDDO
      ELSE              !RANDOM HELICITY
              HWGT = REAL(NGOOD(nFKSprocess))
              ihel=get_hel
              T=B_SF_%(iborn)3.3d(P1,NHEL(1,IHEL),IHEL,JC(1))              
              ANS(IPROC)=ANS(IPROC)+T*HWGT
      ENDIF
      ANS(IPROC)=ANS(IPROC)/DBLE(IDEN(IPROC))
      ENDDO
      END
       
       
      REAL*8 FUNCTION B_SF_%(iborn)3.3d(P,NHEL,HELL,IC)
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
      INTEGER NCOLOR1, NCOLOR2
      PARAMETER (NCOLOR1=%(ncolor1)d, NCOLOR2=%(ncolor2)d) 
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
      REAL*8 DENOM(NCOLOR1), CF(NCOLOR2,NCOLOR1)
      COMPLEX*16 AMP(NGRAPHS), JAMP1(NCOLOR1), JAMP2(NCOLOR2)
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
  %(jamp1_lines)s
  %(jamp2_lines)s
      B_SF_%(iborn)3.3d = 0.D0 
      DO I = 1, NCOLOR1
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR2
              ZTEMP = ZTEMP + CF(j,i)*JAMP2(J)
          ENDDO
          B_SF_%(iborn)3.3d =B_SF_%(iborn)3.3d+ZTEMP*DCONJG(JAMP1(I))/DENOM(I)   
      ENDDO
      END
       
       
