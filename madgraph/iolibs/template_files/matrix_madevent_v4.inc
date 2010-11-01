SUBROUTINE SMATRIX%(proc_id)s(P,ANS)
C  
%(info_lines)s
C 
C MadGraph for Madevent Version
C 
C Returns amplitude squared summed/avg over colors
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C  
%(process_lines)s
C  
    IMPLICIT NONE
C  
C CONSTANTS
C  
    Include 'genps.inc'
    Include 'nexternal.inc'
    Include 'maxamps.inc'
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)d)
    INTEGER    NGRAPHS,    NDIAGS
    PARAMETER (NGRAPHS=%(ngraphs)d,NDIAGS=%(ndiags)d) 
    INTEGER    THEL
    PARAMETER (THEL=NCOMB)
C  
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL),ANS
C  
C LOCAL VARIABLES 
C  
    INTEGER NHEL(NEXTERNAL,NCOMB),NTRY
    REAL*8 T,MATRIX
    INTEGER IHEL,IDEN
    INTEGER IPROC,JC(NEXTERNAL), I
    LOGICAL GOODHEL(NCOMB)
    REAL*8 hwgt, xtot, xtry, xrej, xr, yfrac(0:ncomb)
    INTEGER idum, ngood, igood(ncomb), jhel, j, jj
    LOGICAL warned
    REAL     xran1
    EXTERNAL xran1
C  
C GLOBAL VARIABLES
C  
    Double Precision amp2(maxamps), jamp2(0:maxflow)
    common/to_amps/  amp2,       jamp2
    
    character*79         hel_buff
    common/to_helicity/  hel_buff
    
    REAL*8 POL(2)
    common/to_polarization/ POL
    
    integer          isum_hel
    logical                    multi_channel
    common/to_matrix/isum_hel, multi_channel
    INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
    common/to_mconfigs/mapconfig, iconfig
    DATA NTRY,IDUM /0,-1/
    DATA xtry, xrej, ngood /0,0,0/
    DATA warned, isum_hel/.false.,0/
    DATA multi_channel/.true./
    SAVE yfrac, igood, jhel
    DATA jamp2(0) /%(ncolor)d/          
    DATA GOODHEL/THEL*.FALSE./
%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
    NTRY=NTRY+1
    DO IHEL=1,NEXTERNAL
       JC(IHEL) = +1
    ENDDO
     
    IF (multi_channel) THEN
        DO IHEL=1,NDIAGS
            amp2(ihel)=0d0
        ENDDO
        DO IHEL=1,int(jamp2(0))
            jamp2(ihel)=0d0
        ENDDO
    ENDIF
    ANS = 0D0
    write(hel_buff,'(16i5)') (0,i=1,nexternal)
    IF (ISUM_HEL .EQ. 0 .OR. NTRY .LT. 10) THEN
        DO IHEL=1,NCOMB
           IF (GOODHEL(IHEL) .OR. NTRY .LT. 2) THEN
               T=MATRIX(P ,NHEL(1,IHEL),JC(1))            
             DO JJ=1,nincoming
               IF(POL(JJ).NE.1d0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                 T=T*ABS(POL(JJ))
               ELSE IF(POL(JJ).NE.1d0)THEN
                 T=T*(2d0-ABS(POL(JJ)))
               ENDIF
             ENDDO
             ANS=ANS+T
             IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
                 GOODHEL(IHEL)=.TRUE.
                 NGOOD = NGOOD +1
                 IGOOD(NGOOD) = IHEL
             ENDIF
           ENDIF
        ENDDO
        JHEL = 1
        ISUM_HEL=MIN(ISUM_HEL,NGOOD)
    ELSE              !RANDOM HELICITY
        DO J=1,ISUM_HEL
            JHEL=JHEL+1
            IF (JHEL .GT. NGOOD) JHEL=1
            HWGT = REAL(NGOOD)/REAL(ISUM_HEL)
            IHEL = IGOOD(JHEL)
            T=MATRIX(P ,NHEL(1,IHEL),JC(1))            
            DO JJ=1,nincoming
              IF(POL(JJ).NE.1d0.AND.NHEL(JJ,IHEL).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                T=T*ABS(POL(JJ))
              ELSE IF(POL(JJ).NE.1d0)THEN
                T=T*(2d0-ABS(POL(JJ)))
              ENDIF
            ENDDO
            ANS=ANS+T*HWGT
        ENDDO
        IF (ISUM_HEL .EQ. 1) THEN
            WRITE(HEL_BUFF,'(16i5)')(NHEL(i,IHEL),i=1,nexternal)
        ENDIF
    ENDIF
    IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO IHEL=1,MAPCONFIG(0)
            XTOT=XTOT+AMP2(MAPCONFIG(IHEL))
        ENDDO
        IF (XTOT.NE.0D0) THEN
            ANS=ANS*AMP2(MAPCONFIG(ICONFIG))/XTOT
        ELSE
            ANS=0D0
        ENDIF
    ENDIF
    ANS=ANS/DBLE(IDEN)
    END
 
 
REAL*8 FUNCTION MATRIX%(proc_id)s(P,NHEL,IC)
C  
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
    IMPLICIT NONE
C  
C CONSTANTS
C  
    INTEGER    NGRAPHS,    NDIAGS
    PARAMETER (NGRAPHS=%(ngraphs)d,NDIAGS=%(ndiags)d) 
    include 'genps.inc'
    include 'nexternal.inc'
    include 'maxamps.inc'
    INTEGER    NWAVEFUNCS,     NCOLOR
    PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
    REAL*8     ZERO
    PARAMETER (ZERO=0D0)
C  
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL)
    INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
    INTEGER I,J
    COMPLEX*16 ZTEMP
    REAL*8 DENOM(NCOLOR), CF(NCOLOR,NCOLOR)
    COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR)
    COMPLEX*16 W(18,NWAVEFUNCS), AMP2TMP
C  
C GLOBAL VARIABLES
C  
    Double Precision amp2(maxamps), jamp2(0:maxflow)
    common/to_amps/  amp2,       jamp2
    include 'coupl.inc'
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
%(helas_calls)s
%(jamp_lines)s
    MATRIX = 0.D0 
    DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = 1, NCOLOR
            ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
        ENDDO
        MATRIX=MATRIX+ZTEMP*DCONJG(JAMP(I))/DENOM(I)   
    ENDDO
%(amp2_lines)s
    Do I = 1, NCOLOR
        Jamp2(i)=Jamp2(i)+Jamp(i)*dconjg(Jamp(i))
    Enddo

    END