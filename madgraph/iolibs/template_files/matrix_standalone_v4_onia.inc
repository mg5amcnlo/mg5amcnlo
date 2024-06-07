C MadOnia
      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL,ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NCOMB         
      PARAMETER (NCOMB=%(ncomb)d)
CF2PY INTENT(OUT) :: ANS                                                                                                                                                                                     
CF2PY INTENT(IN) :: HEL                                                                                                                                                                                    
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)  

C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL %(proc_prefix)sSMATRIX(P,ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIX(P,ANS)
C  
%(info_lines)s
C 
C MadGraph5_aMC@NLO StandAlone Version
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
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NINITIAL 
      PARAMETER (NINITIAL=%(nincoming)d)
      INTEGER    NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER    NCOMB         
      PARAMETER (NCOMB=%(ncomb)d)
      INTEGER    HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER NHEL(NEXTERNAL,NCOMB), NTRY
C  put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL 
      REAL*8 T
      REAL*8 %(proc_prefix)sMATRIX
      INTEGER IHEL, IDEN, I, J
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/%(beamone_helavgfactor)d,%(beamtwo_helavgfactor)d/
      INTEGER JC(NEXTERNAL)
      LOGICAL GOODHEL(NCOMB)
      DATA NTRY/0/
      DATA GOODHEL/NCOMB*.FALSE./

C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
      DATA USERHEL/-1/    
      LOGICAL HELRESET
      COMMON/%(proc_prefix)sHELRESET/HELRESET
      DATA HELRESET/.TRUE./
      
      %(helicity_lines)s
      %(den_factor_line)s

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/
C
C MADONIA VARIABLES
C
      INCLUDE 'onia.inc'

      LOGICAL IS_PROJ_COMPUTED(NONIA,-1:1)
      COMMON/%(proc_prefix)sIS_PROJ_COMPUTED/IS_PROJ_COMPUTED

      LOGICAL IS_JAMP_COMPUTED(4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/%(proc_prefix)sIS_JAMP_COMPUTED/IS_JAMP_COMPUTED

C
C     FUNCTIONS
C
      LOGICAL %(proc_prefix)sIS_BORN_HEL_SELECTED

c----------
c   Check if helreset mode is on
c---------
      IF (HELRESET) THEN
        NTRY = 0
        DO I=1,NCOMB
          GOODHEL(I) = .FALSE.
        ENDDO
        HELRESET = .FALSE.
      ENDIF

C ----------
C BEGIN CODE
C ----------
      
      IS_PROJ_COMPUTED=.FALSE.
      IS_JAMP_COMPUTED=.FALSE.

      IF(USERHEL.EQ.-1) NTRY=NTRY+1
      DO IHEL=1,NEXTERNAL
        JC(IHEL) = +1
      ENDDO
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filterin when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        DO IHEL=1,NCOMB
          GOODHEL(IHEL)=.TRUE.
        ENDDO
      ENDIF
      ANS = 0D0
        DO IHEL=1,NCOMB
          IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN       
            IF (GOODHEL(IHEL) .OR. NTRY .LT. 20.OR.USERHEL.NE.-1) THEN
              IF(NTRY.GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL))) THEN
                CYCLE
              ENDIF
              T=%(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1))
              IF(POLARIZATIONS(0,0).eq.-1.or.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL)) THEN
                ANS=ANS+T
              ENDIF
              IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL)) THEN
                GOODHEL(IHEL)=.TRUE.
              ENDIF
            ENDIF
          ENDIF
        ENDDO
      ANS=ANS/DBLE(IDEN)
      IF(USERHEL.NE.-1) THEN
        ANS=ANS*HELAVGFACTOR    
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).ne.-1) THEN
            ANS=ANS*BEAMS_HELAVGFACTOR(J)
            ANS=ANS/POLARIZATIONS(J,0)
          ENDIF
        ENDDO
      ENDIF
      END
       
       
      REAL*8 FUNCTION %(proc_prefix)sMATRIX(P,NHEL,IC)
C  
%(info_lines)s
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c for the point with external lines W(0:6,NEXTERNAL)
C  
%(process_lines)s
C  
      IMPLICIT NONE
C  
C CONSTANTS
C  
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d) 
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NWAVEFUNCS, NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d) 
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
C  
C LOCAL VARIABLES 
C  
      INTEGER I, J, K, QQ, Q1, Q2
      REAL*8 FRAC
      COMPLEX*16 ZTEMP
      REAL*8 CF(NCOLOR,NCOLOR)
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 W(20,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C MADONIA VAIABLES
C
      INCLUDE 'onia.inc'
      INCLUDE 'ldme.inc'
      REAL*8 NORM, LDME
      COMPLEX*16 PROJ(4**NONIA)
      REAL*8 P_ONIA(0:3,NEXTERNAL+NONIA)
      INTEGER NHEL_ONIA(NEXTERNAL+NONIA), IC_ONIA(NEXTERNAL+NONIA)
      INTEGER HELQ1(2), HELQ2(2)
      DATA HELQ1/-1, 1/
      DATA HELQ2/-1, 1/

      LOGICAL IS_PROJ_COMPUTED(NONIA,-1:1)
      COMPLEX*16 PROJ_SAVED(NONIA,-1:1,-1:1,-1:1)
      COMMON/%(proc_prefix)sIS_PROJ_COMPUTED/IS_PROJ_COMPUTED
      COMMON/%(proc_prefix)sPROJ_SAVED/PROJ_SAVED

      LOGICAL IS_JAMP_COMPUTED(4**NONIA,3**(NEXTERNAL-NONIA))
      COMPLEX*16 JAMP_SAVED(NCOLOR,4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/%(proc_prefix)sIS_JAMP_COMPUTED/IS_JAMP_COMPUTED
      COMMON/%(proc_prefix)sJAMP_SAVED/JAMP_SAVED
C  
C GLOBAL VARIABLES
C
      REAL*8 PMASS(NEXTERNAL)
      REAL*8 PMASS_ONIA(NEXTERNAL+NONIA)
      INCLUDE 'coupl.inc'
      INCLUDE 'pmass.inc'
      INCLUDE 'pmassonia.inc'
      %(global_variable)s
C  
C COLOR DATA
C  
      %(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      DO I = 1, NEXTERNAL+NONIA
        IF (PAIRS(I).NE.0) THEN
          FRAC = PMASS_ONIA(I)/PMASS(MAPPING(I))
        ELSE
          FRAC = 1.D0
        ENDIF
        P_ONIA(:,I) = FRAC*P(:,MAPPING(I))
        NHEL_ONIA(I) = NHEL(MAPPING(I))
        IC_ONIA(I) = IC(MAPPING(I))
      ENDDO

      NORM = 1D0
      LDME = 1D0
      DO I = 1, NONIA
        QQ = ONIA(I)
        Q1 = FINDLOC(PAIRS, I, DIM=1)
        Q2 = FINDLOC(PAIRS,-I, DIM=1)
C       Computing the projectors
        IF (.NOT.IS_PROJ_COMPUTED(I,NHEL(QQ))) THEN
          DO J = 1, SIZE(HELQ1)
            DO K = 1, SIZE(HELQ2)
              NHEL_ONIA(Q1) = HELQ1(J)
              NHEL_ONIA(Q2) = HELQ2(K) 
              CALL ONIA_PROJ(P_ONIA(:,Q2), P_ONIA(:,Q1),PMASS_ONIA(Q2),PMASS_ONIA(Q1),NHEL_ONIA(Q2),NHEL_ONIA(Q1),S_ONIA(I),NHEL(QQ),PROJ(1))   
              PROJ_SAVED(I,NHEL(QQ),NHEL_ONIA(Q1),NHEL_ONIA(Q2)) = PROJ(1)
            ENDDO
          ENDDO
          IS_PROJ_COMPUTED(I,NHEL(QQ)) = .TRUE.
        ENDIF
C       Computing normalisation factor from color projector
        IF (C_ONIA(I).EQ.1) THEN
          NORM = NORM/3D0
        ELSE IF (C_ONIA(I).EQ.8) THEN
          NORM = NORM*2D0
        ENDIF 
C       Computing normalisation factor for LDME convention
        LDME = LDME*PMASS(QQ)/(2D0*PMASS_ONIA(Q1)*PMASS_ONIA(Q2))
      ENDDO

C     Computing product of LDMEs
      LDME = LDME*%(ldme_product)s

C     Computing the amplitudes
      K = 1
      J = 0
      DO I = 1, NEXTERNAL+NONIA
        IF (PAIRS(I).EQ.0) THEN
          K = K + 3**J*(NHEL_ONIA(I)+1)
          J = J+1
        ENDIF
      ENDDO
      DO J = 1, 4**NONIA
        PROJ(J) = SQRT(NORM*LDME)
        DO I =1, NONIA
          QQ = ONIA(I)  
          Q1 = FINDLOC(PAIRS, I, DIM=1)
          Q2 = FINDLOC(PAIRS,-I, DIM=1)
          NHEL_ONIA(Q1) = HELQ1(IBITS(J-1,2*I-2,1)+1)
          NHEL_ONIA(Q2) = HELQ2(IBITS(J-1,2*I-1,1)+1)
          PROJ(J) = PROJ(J) * PROJ_SAVED(I,NHEL(QQ),NHEL_ONIA(Q1),NHEL_ONIA(Q2))
        ENDDO
        IF (PROJ(J).NE.(0D0,0D0)) THEN
          IF (.NOT.IS_JAMP_COMPUTED(J,K)) THEN
            %(helas_calls)s

            %(jamp_lines)s
C           Store amplitudes for reuse
            JAMP_SAVED(:,J,K) = JAMP
            IS_JAMP_COMPUTED(J,K) = .TRUE.
          ENDIF
        ENDIF
      ENDDO

      JAMP = (0D0,0D0)
      DO I = 1, 4**NONIA
        JAMP = JAMP + JAMP_SAVED(:,I,K)*PROJ(I)
      ENDDO  

      %(proc_prefix)sMATRIX = 0.D0 
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
              ZTEMP = ZTEMP + CF(J,I)*JAMP(J)
          ENDDO
          %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX+ZTEMP*DCONJG(JAMP(I))
      ENDDO
      %(amp2_lines)s
      END



      SUBROUTINE %(proc_prefix)sGET_VALUE(P, ALPHAS, NHEL ,ANS)
      IMPLICIT NONE   
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
C  
C ARGUMENTS 
C  
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER NHEL
      DOUBLE PRECISION ALPHAS 
      real*8 pi     
CF2PY INTENT(OUT) :: ANS  
CF2PY INTENT(IN) :: NHEL   
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL) 
CF2PY INTENT(IN) :: ALPHAS
C     ROUTINE FOR F2PY to read the benchmark point.    
C     the include file with the values of the parameters and masses 
      include "coupl.inc"
      
      pi = 3.141592653589793d0
      G = 2*DSQRT(ALPHAS*pi)
      call UPDATE_AS_PARAM()
      if (NHEL.ne.0) then
        CALL %(proc_prefix)sSMATRIXHEL(P, NHEL, ANS)
      else
        CALL %(proc_prefix)sSMATRIX(P, ANS)
      endif
      return 
      end

      SUBROUTINE %(proc_prefix)sINITIALISEMODEL(PATH)
C     ROUTINE FOR F2PY to read the benchmark point.    
      IMPLICIT NONE   
      CHARACTER*512 PATH
CF2PY INTENT(IN) :: PATH 
      call setpara(PATH)  !first call to setup the paramaters    
      return 
      end      

      LOGICAL FUNCTION %(proc_prefix)sIS_BORN_HEL_SELECTED(HELID)
      IMPLICIT NONE
C     
C     CONSTANTS
C     
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)
C
C     ARGUMENTS
C
      INTEGER HELID
C
C     LOCALS
C
      INTEGER I,J
      LOGICAL FOUNDIT
C
C     GLOBALS
C
      INTEGER HELC(NEXTERNAL,NCOMB)
      COMMON/%(proc_prefix)sPROCESS_NHEL/HELC

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
C     ----------
C     BEGIN CODE
C     ----------
      
      %(proc_prefix)sIS_BORN_HEL_SELECTED = .True.
      if (POLARIZATIONS(0,0).eq.-1) THEN
        RETURN
      ENDIF

      DO I=1,NEXTERNAL
        IF (POLARIZATIONS(I,0).eq.-1) THEN
          CYCLE
        ENDIF
        FOUNDIT = .FALSE.
        DO J=1,POLARIZATIONS(I,0)
          IF (HELC(I,HELID).eq.POLARIZATIONS(I,J)) THEN
            FOUNDIT = .True.
            EXIT
          ENDIF
        ENDDO
        IF(.not.FOUNDIT) THEN
          %(proc_prefix)sIS_BORN_HEL_SELECTED = .False.
          RETURN
        ENDIF
      ENDDO

      RETURN
      END
