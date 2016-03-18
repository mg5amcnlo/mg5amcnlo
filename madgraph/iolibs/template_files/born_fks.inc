      SUBROUTINE SBORN(P1,ANS)
C  
%(info_lines)s
C
C RETURNS AMPLITUDE SQUARED SUMMED/AVG OVER COLORS
C AND HELICITIES
C FOR THE POINT IN PHASE SPACE P1(0:3,NEXTERNAL-1)
C  
%(process_lines)s
C
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include "nexternal.inc"
      include "born_nhel.inc"
      include "genps.inc"
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      INTEGER NGRAPHS
      PARAMETER (NGRAPHS=   %(ngraphs)d)
C  
C ARGUMENTS 
C  
      REAL*8 P1(0:3,NEXTERNAL-1)
      COMPLEX*16 ANS(2)
C  
C LOCAL VARIABLES 
C  
      INTEGER IHEL,IDEN,i,j,jj,glu_ij
      REAL*8 BORN,borns(2)
      COMPLEX*16 BORNTILDE
      INTEGER NTRY(%(nconfs)d)
      DATA NTRY /%(nconfs)d*0/
      INTEGER NHEL(NEXTERNAL-1,NCOMB)
%(helicity_lines)s
%(den_factor_lines)s
%(ij_lines)s
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2
      DATA jamp2(0) /   %(ncolor)d/          
      LOGICAL GOODHEL(NCOMB,%(nconfs)d)
      common /c_goodhel/goodhel
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision savemom(nexternal-1,2)
      common/to_savemom/savemom
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      double precision       wgt_ME_born,wgt_ME_real
      common /c_wgt_ME_tree/ wgt_ME_born,wgt_ME_real
C ----------
C BEGIN CODE
C ----------
      iden=iden_values(nfksprocess)
      glu_ij = ij_values(nfksprocess)
      NTRY(nFKSprocess)=NTRY(nFKSprocess)+1
      if (NTRY(nFKSprocess).lt.2) then
         skip(nFKSprocess)=1			 
         do while(nhel(glu_ij ,skip(nFKSprocess)).ne.-NHEL(GLU_IJ ,1))
            skip(nFKSprocess)=skip(nFKSprocess)+1
         enddo
         skip(nFKSprocess)=skip(nFKSprocess)-1
      endif
      DO JJ=1,NGRAPHS
          amp2(jj)=0d0
      ENDDO
      DO JJ=1,int(jamp2(0))
          jamp2(jj)=0d0
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
      ANS(1) = 0D0
      ANS(2) = 0D0
      hel_fac=1d0
      DO IHEL=1,NCOMB
        if (nhel(glu_ij,ihel).EQ.NHEL(GLU_IJ,1)) then
          IF ((GOODHEL(IHEL,nFKSprocess) .OR. GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess) .OR. NTRY(nFKSprocess) .LT. 2) ) THEN
            ANS(1)=ANS(1)+BORN(P1,NHEL(1,IHEL),IHEL,BORNTILDE,borns)
            ANS(2)=ANS(2)+BORNTILDE
            if ( borns(1).ne.0d0 .AND. .NOT. GOODHEL(IHEL,nFKSprocess) ) then
              GOODHEL(IHEL,nFKSprocess)=.TRUE.
            endif
            if ( borns(2).ne.0d0 .AND. .NOT. GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess) ) then
              GOODHEL(IHEL+SKIP(nFKSprocess),nFKSprocess)=.TRUE.
            endif
	  ENDIF
        ENDIF
      ENDDO
      ANS(1)=ANS(1)/DBLE(IDEN)
      ANS(2)=ANS(2)/DBLE(IDEN)
      wgt_me_born=dble(ans(1))
      calculatedBorn=.true.
      END
       
       
      REAL*8 FUNCTION BORN(P,NHEL,HELL,BORNTILDE,borns)
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
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      complex*16 imag1
      parameter (imag1 = (0d0,1d0))
      include "nexternal.inc"
      include "born_nhel.inc"
      include "coupl.inc"
      include "genps.inc"
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL-1),borns(2)
      INTEGER NHEL(NEXTERNAL-1), HELL
      complex*16 borntilde
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,ihel,back_hel,glu_ij
      INTEGER IC(NEXTERNAL-1),nmo
      parameter (nmo=nexternal-1)
      data ic /nmo*1/
      REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
      COMPLEX*16 ZTEMP, AMP(NGRAPHS), JAMP(NCOLOR), W(%(wavefunctionsize)d,NWAVEFUNCS), jamph(2, ncolor)
C  
C GLOBAL VARIABLES
C  
      Double Precision amp2(maxamps), jamp2(0:maxamps)
      common/to_amps/  amp2,       jamp2
      double complex saveamp(ngraphs,max_bhel)
      common/to_saveamp/saveamp
      double precision hel_fac
      integer get_hel,skip(%(nconfs)d)
      common/cBorn/hel_fac,get_hel,skip
      logical calculatedBorn
      common/ccalculatedBorn/calculatedBorn
      integer nfksprocess
      common/c_nfksprocess/nfksprocess
      integer step_hel
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
      borns(1) = 0d0
      borns(2) = 0d0
      if (back_hel.ne.0) then
        step_hel=-2*back_hel
      else
        step_hel=1
      endif
      DO IHEL=back_hel,-back_hel,step_hel
         IF (IHEL.EQ.back_hel.OR.NHEL(GLU_IJ).NE.0) THEN
        if (nhel(glu_ij).ne.0) nhel(glu_ij) = ihel
        if (.not. calculatedBorn) then
%(helas_calls)s
        do i=1,ngraphs
          if(ihel.eq.BACK_HEL)then
            saveamp(i,hell)=amp(i)
          elseif(ihel.eq.-BACK_HEL)then
            saveamp(i,hell+skip(nFKSprocess))=amp(i)
          else
            write(*,*) "ERROR #1 in born.f"
            stop
          endif
        enddo
        elseif (calculatedBorn) then
        do i=1,ngraphs
          if(ihel.eq.BACK_HEL)then
            amp(i)=saveamp(i,hell)
          elseif(ihel.eq.-BACK_HEL)then
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
          BORNS(2-(1+back_hel*ihel)/2)=BORNS(2-(1+back_hel*ihel)/2)+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
        ENDDO
        Do I = 1, NGRAPHS
          amp2(i)=amp2(i)+amp(i)*dconjg(amp(i))
        Enddo
        Do I = 1, NCOLOR
          Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
          Jamph(2-(1+back_hel*ihel)/2,i)=Jamp(i)
        Enddo
      endif
      Enddo
      BORN=BORNS(1)+BORNS(2)
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
          ZTEMP = ZTEMP + CF(J,I)*JAMPH(2,J)
        ENDDO
        BORNTILDE = BORNTILDE + ZTEMP*DCONJG(JAMPH(1,I))/DENOM(I)
      ENDDO
      nhel(glu_ij) = back_hel
      END
       

      BLOCK DATA GOODHELS
      INTEGER     NCOMB
      PARAMETER ( NCOMB=  %(ncomb)d )
      INTEGER    THEL
      PARAMETER (THEL=NCOMB*%(nconfs)d)
      LOGICAL GOODHEL(NCOMB,%(nconfs)d)
      common /c_goodhel/goodhel
      DATA GOODHEL/THEL*.FALSE./
      end
