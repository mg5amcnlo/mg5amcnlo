C MadSONS P-state version
      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,P_RESH,HEL,ANS)
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
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL),P_RESH(0:3,NEXTERNAL)

C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),P_RESH(0:3,NEXTERNAL),ANS
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
      CALL %(proc_prefix)sSMATRIX(P,P_RESH,ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIX(P,P_RESH,ANS)
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
      REAL*8 P(0:3,NEXTERNAL),P_RESH(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL),P_RESH(0:3,NEXTERNAL)
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
C ONIA VARIABLES
C
      INCLUDE 'onia.inc'
      LOGICAL IS_PROJ_COMPUTED(NONIA,-1:1,-1:1)
      COMMON/%(proc_prefix)sIS_PROJ_COMPUTED/IS_PROJ_COMPUTED

      LOGICAL IS_JAMP_COMPUTED(4**NONIA,3**(NEXTERNAL-NONIA),-(3**(NPWAVE)-1)/2:(3**(NPWAVE)-1)/2)
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
C     lorentz invariant with external spin-2 particles (only the helicity sum is).
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
              T=%(proc_prefix)sMATRIX(P,P_RESH,NHEL(1,IHEL),JC(1))
              IF(POLARIZATIONS(0,0).eq.-1.or.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL)) THEN
                ANS=ANS+T
              ENDIF
              IF (T .NE. 0D0 .AND. .NOT. GOODHEL(IHEL)) THEN
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


      REAL*8 FUNCTION %(proc_prefix)sMATRIX(P,P_RESH,NHEL,IC)
      USE DUAL_VARIABLES
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
      INTEGER I, J, K, H, HRUN
      INTEGER QQ, Q1, Q2
      REAL*8 FRAC
      COMPLEX*16 ZTEMP, JAMP_CMPLX(NCOLOR)
      INTEGER CF_INDEX
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM
      TYPE(DUAL) AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      TYPE(DUAL) W(20,NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C MADSONS (DUAL)VARIABLES
C
      INCLUDE 'onia.inc'
      INCLUDE 'ldme.inc'
      REAL*8 NORM, LDME
      TYPE(DUAL) P_ONIA(0:3,NEXTERNAL+NONIA)
      REAL*8 P_RESH(0:3,NEXTERNAL),PMASS_RESH(NEXTERNAL)
      TYPE(DUAL) EPS_L(8), PQQ(0:3)
      TYPE(DUAL) FQ(8), FQB(8)
      TYPE(DUAL) PROJ(4**NONIA,-(3**(NPWAVE)-1)/2:(3**(NPWAVE)-1)/2)
      REAL*8 CG_COEFF
      INTEGER NHEL_ONIA(NEXTERNAL+NONIA), IC_ONIA(NEXTERNAL+NONIA)
      INTEGER HELQ1(2), HELQ2(2)
      INTEGER LZ, SZ
      INTEGER LZ_LIST(NPWAVE)
      DATA HELQ1/-1, 1/
      DATA HELQ2/-1, 1/


      LOGICAL IS_PROJ_COMPUTED(NONIA,-1:1,-1:1)
      TYPE(DUAL),SAVE::PROJ_SAVED(NONIA,-1:1,-1:1,-1:1,-1:1) 
      REAL*8 CG_SAVED(NONIA,-1:1,-1:1,-2:2)
      COMMON/%(proc_prefix)sIS_PROJ_COMPUTED/IS_PROJ_COMPUTED

      LOGICAL IS_JAMP_COMPUTED(4**NONIA,3**(NEXTERNAL-NONIA),-(3**(NPWAVE)-1)/2:(3**(NPWAVE)-1)/2)
      TYPE(DUAL),SAVE::JAMP_SAVED(NCOLOR,4**NONIA,3**(NEXTERNAL-NONIA),-(3**(NPWAVE)-1)/2:(3**(NPWAVE)-1)/2)

      COMMON/%(proc_prefix)sIS_JAMP_COMPUTED/IS_JAMP_COMPUTED

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
C     Assigning momenta to the elementary particles
      PMASS_RESH = 0.D0
      DO I = 1, NEXTERNAL+NONIA
        PMASS_RESH(MAPPING(I)) = PMASS_RESH(MAPPING(I))+PMASS_ONIA(I)
      ENDDO
      CALL %(proc_prefix)sRESHUFFLE_MOMENTA(P,PMASS_RESH,P_RESH)

      DO I = 1, NEXTERNAL+NONIA
        IF (PAIRS(I).NE.0) THEN
          FRAC = PMASS_ONIA(I)/PMASS_RESH(MAPPING(I))
        ELSE
          FRAC = 1.D0
        ENDIF
        P_ONIA(:,I)%%COMP(0) = FRAC*P_RESH(:,MAPPING(I))
        NHEL_ONIA(I) = NHEL(MAPPING(I))
        IC_ONIA(I) = IC(MAPPING(I))
      ENDDO

      NORM = 1D0
      LDME = 1D0
      DO I = 1, NONIA
        QQ = ONIA(I)
        Q1 = FINDLOC(PAIRS, I, DIM=1)
        Q2 = FINDLOC(PAIRS,-I, DIM=1)
        PQQ%%COMP(0) = P_RESH(:,QQ)
        
C       Computing the projectors
        DO LZ = -L_ONIA(I), L_ONIA(I)
C         SZ = JZ - LZ with the physical condition |SZ| < S 
          SZ = NHEL(QQ)-LZ
          IF (ABS(SZ).GT.S_ONIA(I)) THEN
            CYCLE
          ENDIF

C         Computing the spin projector
          IF (.NOT.IS_PROJ_COMPUTED(I,LZ,SZ)) THEN
C           Inserting the dual component of momenta (only for L>0)
            IF (L_ONIA(I).EQ.1) THEN
              CALL VXXXXX(PQQ,PMASS_RESH(QQ),LZ,+1,EPS_L)
              P_ONIA(:,Q1)%%COMP(2**(SUM(L_ONIA(1:I))-1)) = +EPS_L(5:8)%%COMP(0)
              P_ONIA(:,Q2)%%COMP(2**(SUM(L_ONIA(1:I))-1)) = -EPS_L(5:8)%%COMP(0)
            ENDIF
            DO J = 1, SIZE(HELQ1)
              DO K = 1, SIZE(HELQ2)
                NHEL_ONIA(Q1) = HELQ1(J)
                NHEL_ONIA(Q2) = HELQ2(K)
                CALL IXXXXX(P_ONIA(:,Q1),PMASS_ONIA(Q1),NHEL_ONIA(Q1),+1,FQ )
                CALL OXXXXX(P_ONIA(:,Q2),PMASS_ONIA(Q2),NHEL_ONIA(Q2),-1,FQB)
                CALL ONIA_PROJ_DUAL(FQ,FQB,(/PMASS_ONIA(Q1),PMASS_ONIA(Q2)/),PQQ,PMASS_RESH(QQ),SZ,S_ONIA(I),PROJ(1,0))
c               LM::I think that the normalisation of the proj can also be moved here, this would avoid passing the masses to the projector routine.
                PROJ_SAVED(I,LZ,SZ,NHEL_ONIA(Q1),NHEL_ONIA(Q2)) = PROJ(1,0)
              ENDDO
            ENDDO        
            IS_PROJ_COMPUTED(I,LZ,SZ) = .TRUE.
          ENDIF

C         Computing the Jz projector (Clebsch-Gordan coefficient table)
          CALL CLEBSCH_GORDAN(L_ONIA(I),LZ,S_ONIA(I),SZ,J_ONIA(I),NHEL(QQ),CG_COEFF)
          CG_SAVED(I,LZ,SZ,NHEL(QQ)) = CG_COEFF
        ENDDO

C       Computing normalisation factor from color projector
        IF (C_ONIA(I).EQ.1) THEN
          NORM = NORM/3D0
        ELSE IF (C_ONIA(I).EQ.8) THEN
          NORM = NORM*2D0
        ENDIF

C       Computing normalisation factor for LDME convention
        LDME = LDME*PMASS_RESH(QQ)/(2D0*PMASS_ONIA(Q1)*PMASS_ONIA(Q2))
        IF (C_ONIA(I).EQ.1) THEN
          LDME = LDME/6D0/(2D0*J_ONIA(I)+1D0)
        ELSE IF (C_ONIA(I).EQ.8) THEN
          LDME = LDME/8D0/(2D0*J_ONIA(I)+1D0)
        ENDIF
      ENDDO

C     Computing perturbative LDMEs
      %(ldme_perturbative)s
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
      
      JAMP_CMPLX = (0D0,0D0)
      LZ_LOOP: DO H = -(3**(NPWAVE)-1)/2, (3**(NPWAVE)-1)/2
        HRUN = H

        DO I = 1, NONIA
          IF (L_ONIA(I).EQ.1) THEN
            QQ = ONIA(I)
C           For each L>0 state, retrieving LZ encoded in H
            IF (MODULO(HRUN,3).EQ.0) THEN
              LZ_LIST(SUM(L_ONIA(1:I))) = 0
              LZ = 0
              HRUN = HRUN/3
            ELSEIF (MODULO(HRUN,3).EQ.1) THEN
              LZ_LIST(SUM(L_ONIA(1:I))) = 1
              LZ = 1
              HRUN = (HRUN - 1)/3
            ELSEIF (MODULO(HRUN,3).EQ.2) THEN
              LZ_LIST(SUM(L_ONIA(1:I))) = -1
              LZ = -1
              HRUN = (HRUN + 1)/3
            ENDIF
C           Verifying if the current H value is accepted
            SZ = NHEL(QQ) - LZ
            IF (ABS(SZ).GT.S_ONIA(I)) CYCLE LZ_LOOP
          ENDIF
        ENDDO
        
        DO J = 1, 4**NONIA
          PROJ(J,H)%%COMP(0) = SQRT(NORM)
          DO I = 1, NONIA
            QQ = ONIA(I)
            Q1 = FINDLOC(PAIRS,+I, DIM=1)
            Q2 = FINDLOC(PAIRS,-I, DIM=1)
            NHEL_ONIA(Q1) = HELQ1(IBITS(J-1,2*I-2,1)+1)
            NHEL_ONIA(Q2) = HELQ2(IBITS(J-1,2*I-1,1)+1)
            IF (L_ONIA(I).EQ.1) THEN
              LZ = LZ_LIST(SUM(L_ONIA(1:I)))
C             Cross-check SZ physical condition 
              SZ = NHEL(QQ) - LZ
              IF (ABS(SZ).GT.S_ONIA(I)) CYCLE LZ_LOOP
C             Cross-check passed!
C             Inserting dual components
              PQQ%%COMP(0) = P_RESH(:,QQ)
              CALL VXXXXX(PQQ,PMASS_RESH(QQ),LZ,+1,EPS_L)
              P_ONIA(:,Q1)%%COMP(2**(SUM(L_ONIA(1:I))-1)) = +EPS_L(5:8)%%COMP(0)
              P_ONIA(:,Q2)%%COMP(2**(SUM(L_ONIA(1:I))-1)) = -EPS_L(5:8)%%COMP(0)
            ELSE
              LZ = 0
              SZ = NHEL(QQ)
            ENDIF
            PROJ(J,H) = PROJ(J,H) * PROJ_SAVED(I,LZ,SZ,NHEL_ONIA(Q1),NHEL_ONIA(Q2)) * CG_SAVED(I,LZ,SZ,NHEL(QQ))
          ENDDO
          IF (ANY(PROJ(J,H)%%COMP(:).NE.(0D0,0D0))) THEN
            IF (.NOT.IS_JAMP_COMPUTED(J,K,H)) THEN
          
              %(helas_calls)s
  
              %(jamp_lines)s
C             Store amplitudes for reuse
              JAMP_SAVED(:,J,K,H) = JAMP
              IS_JAMP_COMPUTED(J,K,H) = .TRUE.
            ENDIF
          ENDIF
C         Adding the current helicity configuration to the total JAMP
          DO I = 1, NCOLOR
            JAMP(I) = JAMP_SAVED(I,J,K,H)*PROJ(J,H)
            JAMP_CMPLX(I) = JAMP_CMPLX(I) + JAMP(I)%%COMP(2**NPWAVE-1)
          ENDDO
        ENDDO
      ENDDO LZ_LOOP

      %(proc_prefix)sMATRIX = 0.D0
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX +1
          ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP_CMPLX(J)
        ENDDO
        %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX+REAL(ZTEMP*CONJG(JAMP_CMPLX(I))*LDME)
      ENDDO
      %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX / DENOM
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
      REAL*8 P(0:3,NEXTERNAL),P_RESH(0:3,NEXTERNAL),ANS
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
        CALL %(proc_prefix)sSMATRIXHEL(P, P_RESH, NHEL, ANS)
      else
        CALL %(proc_prefix)sSMATRIX(P, P_RESH, ANS)
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

      SUBROUTINE %(proc_prefix)sRESHUFFLE_MOMENTA(P,PMASS,P_RESH)
      IMPLICIT NONE
C
C     CONSTANT
C
      INCLUDE 'nexternal.inc'
      INCLUDE 'onia.inc'
C
C     ARGUMENTS
C
      REAL*8    P(0:3,NEXTERNAL), P_RESH(0:3,NEXTERNAL)
      REAL*8    PMASS(NEXTERNAL)
C
C     LOCALS
C
      LOGICAL RESHUFFLE
      INTEGER MOM_RESH_TYPE
      COMMON/TO_MOM_RESH_TYPE/MOM_RESH_TYPE
      INTEGER I, J, CNT_ONIA, CNT_ELEM, NELEM, MAP(NEXTERNAL)
      PARAMETER (NELEM = NEXTERNAL-NONIA-2)

      REAL*8 TINY
      PARAMETER (TINY=1D-99)

      ! HADRONIC SYSTEM
      REAL*8 HAD_P(0:3,NONIA), HAD_M(NONIA), HAD_K(0:3,NELEM)
      REAL*8 HAD_PIN(0:3,NONIA), HAD_MIN(NONIA), HAD_KX(0:3)
      REAL*8 HAD_SQRTS, MREC

      ! PARTONIC SYSTEM
      REAL*8 PAR_P(0:3,NONIA), PAR_M(NONIA), PAR_K(0:3,NELEM)
      REAL*8 PAR_PIN(0:3,NONIA), PAR_MIN(NONIA), PAR_KX(0:3)
      REAL*8 PAR_SQRTS, PAR_E1, PAR_E2

      ! AUXILIARY
      REAL*8 PCOM(0:3,NEXTERNAL), PAUX(0:3), UNIT_VECTOR(1:3)

C     ----------
C     BEGIN CODE
C     ----------

      RESHUFFLE = .FALSE.
      DO I = 3, NEXTERNAL
        IF (ABS(PMASS(I)-MASS(P(:,I)))/PMASS(I).GT.1D-11) THEN 
          RESHUFFLE = .TRUE.
        ENDIF
      ENDDO
      IF (.NOT.RESHUFFLE) THEN
        P_RESH = P
        RETURN
      ENDIF

      IF (NINCOMING.NE.2) THEN
        PRINT*, 'Momentum reshuffling only possible for 2->N processes.'
        STOP
      ENDIF

      IF (MOM_RESH_TYPE > 0) THEN
C     FINAL-STATE MOMENTUM RESHUFFLING
        DO I = 1, NEXTERNAL
          PAUX = P(0:3,1)+P(0:3,2)
          CALL BOOST(PAUX,+1D0,P(0:3,I),PCOM(0:3,I))
        ENDDO

        CNT_ONIA = 0
        CNT_ELEM = 0
        HAD_KX(0:3) = 0D0
        DO I = NINCOMING+1, NEXTERNAL
          IF (ANY(I==ONIA)) THEN
            CNT_ONIA = CNT_ONIA + 1
            MAP(I) = CNT_ONIA
            HAD_P(0:3,CNT_ONIA) = PCOM(0:3,I)
            HAD_M(CNT_ONIA) = MASS(HAD_P(0:3,CNT_ONIA))
            PAR_M(CNT_ONIA) = PMASS(I)
          ELSE
            CNT_ELEM = CNT_ELEM + 1
            MAP(I) = -CNT_ELEM
            HAD_K(0:3,CNT_ELEM) = PCOM(0:3,I)
            HAD_KX(0:3) = HAD_KX(0:3) + HAD_K(0:3,CNT_ELEM)
          ENDIF
        ENDDO
        MREC = MASS(HAD_KX)

        HAD_PIN(0:3,NONIA) = HAD_P(0:3,NONIA)
        HAD_MIN(NONIA) = HAD_M(NONIA)
        PAR_MIN(NONIA) = PAR_M(NONIA)
        DO I = NONIA-1, 1, -1
          HAD_PIN(0:3,I) = HAD_PIN(0:3,I+1) + HAD_P(0:3,I)
          HAD_MIN(I) = MASS(HAD_PIN(0:3,I))
          PAR_MIN(I) = PHI(HAD_MIN(I),PAR_M(I),PAR_MIN(I+1),HAD_M(I),HAD_MIN(I+1),MOM_RESH_TYPE)
        ENDDO

        HAD_SQRTS = PCOM(0,1)+PCOM(0,2)
        PAR_SQRTS = PHI(HAD_SQRTS,PAR_MIN(1),MREC,HAD_MIN(1),MREC,MOM_RESH_TYPE)

        PAR_PIN(0,1) = EPSILON(PAR_SQRTS,PAR_MIN(1),MREC)
        IF (SQRT(PABS2(HAD_PIN(0:3,1))).GT.TINY) THEN
          PAR_PIN(1:3,1) = PI(PAR_SQRTS,PAR_MIN(1),MREC)*HAD_PIN(1:3,1)/SQRT(PABS2(HAD_PIN(0:3,1)))
        ELSE
          PAR_PIN(1:3,1) = 0D0
        ENDIF

        PAR_KX(0) = EPSILON(PAR_SQRTS,MREC,PAR_MIN(1))
        PAR_KX(1:3) = PI(PAR_SQRTS,PAR_MIN(1),MREC)*HAD_KX(1:3)/SQRT(PABS2(HAD_KX))

        IF (NELEM.GT.1) THEN
          DO I = 1, NELEM
            CALL BOOST(HAD_KX,+1D0,HAD_K(0:3,I),PAUX)
            CALL BOOST(PAR_KX,-1D0,PAUX,PAR_K(0:3,I))
          ENDDO
        ELSE IF (NELEM.EQ.1) THEN
          PAR_K(0:3,1) = PAR_KX(0:3)
        ENDIF

        DO I = 1, NONIA-1
          CALL BOOST(HAD_PIN(0:3,I),+1D0,HAD_P(0:3,I),PAUX)
          UNIT_VECTOR = PAUX(1:3)/PI(HAD_MIN(I),HAD_M(I),HAD_MIN(I+1))
          PAUX(0) = EPSILON(PAR_MIN(I),PAR_M(I),PAR_MIN(I+1))
          PAUX(1:3) = PI(PAR_MIN(I),PAR_M(I),PAR_MIN(I+1))*UNIT_VECTOR
          CALL BOOST(PAR_PIN(0:3,I),-1D0,PAUX,PAR_P(0:3,I))
          PAUX(0) = EPSILON(PAR_MIN(I),PAR_MIN(I+1),PAR_M(I))
          PAUX(1:3) = -PAUX(1:3)
          CALL BOOST(PAR_PIN(0:3,I),-1D0,PAUX,PAR_PIN(0:3,I+1))
        ENDDO
        PAR_P(0:3,NONIA) = PAR_PIN(0:3,NONIA)

        DO I = NINCOMING+1, NEXTERNAL
          J = MAP(I)
          IF (J.GT.0) THEN
            P_RESH(0:3,I) = PAR_P(0:3,J)
          ELSE
            P_RESH(0:3,I) = PAR_K(0:3,-J)
          ENDIF
        ENDDO
        P_RESH(0:3,1) = 0.5D0*PAR_SQRTS*(/1D0,0D0,0D0,1D0/)
        P_RESH(0:3,2) = 0.5D0*PAR_SQRTS*(/1D0,0D0,0D0,-1D0/)

        PCOM = P_RESH
        DO I = 1, NEXTERNAL
          PAUX = P(0:3,1)+P(0:3,2)
          CALL BOOST(PAUX,-1D0,PCOM(0:3,I),P_RESH(0:3,I))
        ENDDO
      ELSE
C     INITIAL-STATE RESHUFFLING
        DO I = NINCOMING+1, NEXTERNAL
          IF (MASS(P(0:3,I)).NE.PMASS(I)) THEN
            P_RESH(1:3,I) = P(1:3,I)
            P_RESH(0,I) = SQRT(PMASS(I)**2+PABS2(P(0:3,I)))
          ELSE
            P_RESH(0:3,I) = P(0:3,I)
          ENDIF
        ENDDO
        PAR_SQRTS = SUM(P_RESH(0,NINCOMING+1:))
        PAR_E1 = 0.5D0*(PAR_SQRTS+P(0,1)-P(0,2))
        PAR_E2 = 0.5D0*(PAR_SQRTS-P(0,1)+P(0,2))
        P_RESH(0:3,1) = PAR_E1*(/ 1D0, 0D0, 0D0, 1D0/)
        P_RESH(0:3,2) = PAR_E2*(/ 1D0, 0D0, 0D0,-1D0/)
      ENDIF

      CONTAINS

        DOUBLE PRECISION FUNCTION MASS(P)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8    P(0:3)

C         ----------
C         BEGIN CODE
C         ----------
          MASS = P(0)*P(0)-P(1)*P(1)-P(2)*P(2)-P(3)*P(3)
          IF (ABS(MASS).LT.1D-07) THEN
            MASS = 0D0
          ELSE
            MASS = SQRT(MASS)
          ENDIF
          RETURN
        END

        DOUBLE PRECISION FUNCTION PABS2(P)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8    P(0:3)

C         ----------
C         BEGIN CODE
C         ----------
          PABS2 = P(1)*P(1)+P(2)*P(2)+P(3)*P(3)
          IF (ABS(PABS2).LT.1D-07) THEN
            PABS2 = 0D0
          ENDIF
          RETURN
        END

        DOUBLE PRECISION FUNCTION PI(M,M1,M2)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8 M,M1,M2

C         ----------
C         BEGIN CODE
C         ----------
          PI = 0.5D0*M*SQRT(1D0-(M1+M2)**2/M**2)*SQRT(1D0-(M1-M2)**2/M**2)
          RETURN
        END

        DOUBLE PRECISION FUNCTION EPSILON(M,M1,M2)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8 M,M1,M2

C         ----------
C         BEGIN CODE
C         ----------
          EPSILON = 0.5D0*M*(1D0+(M1**2-M2**2)/M**2)
          RETURN
        END

        DOUBLE PRECISION FUNCTION PHI(M,MP1,MP2,M1,M2,MOM_RESH_TYPE)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          INTEGER    MOM_RESH_TYPE
          REAL*8     M,MP1,MP2,M1,M2
C
C         LOCALS
C
          REAL*8     N
          PARAMETER (N=100D0)

C         ----------
C         BEGIN CODE
C         ----------
          IF (MOM_RESH_TYPE == 1) THEN
            PHI = 0.5D0*(M+MP1+MP2+SQRT((M-MP1-MP2)**2+(M1+M2-MP1-MP2)**2/N))
          ELSE IF (MOM_RESH_TYPE == 2) THEN
            IF (M.GE.(MP1+MP2)) THEN
              PHI = M
            ELSE
              PHI = (MP1+MP2)
            ENDIF
          ELSE IF (MOM_RESH_TYPE == 3) THEN
            IF (M.GE.(MP1+MP2)) THEN
              PHI = M
            ELSE
              PHI = MAX((MP1+MP2)/(M1+M2)*M, MP1+MP2)
            ENDIF
          ELSE
            STOP 1
          END IF
        END

        SUBROUTINE BOOST(PREF, DIRECTION, P, PBOOST)
          IMPLICIT NONE
C
C         ARGUMENTS
C
          REAL*8 PREF(0:3), DIRECTION, P(0:3), PBOOST(0:3)
C
C         LOCALS
C
          REAL*8 BETA(1:3), BETA2, GAMMA, LAMBDA(0:3,0:3)
          REAL*8 TINY
          PARAMETER (TINY=1D-99)

C         ----------
C         BEGIN CODE
C         ----------
          BETA(1:3) = DIRECTION*PREF(1:3)/PREF(0)

          BETA2 = BETA(1)*BETA(1)+BETA(2)*BETA(2)+BETA(3)*BETA(3)
          GAMMA = 1D0/SQRT(1D0-BETA2)
          BETA2 = MAX(BETA2,TINY)

C         CONSTRUCT LORENTZ TRANSFORMATION MATRIX
          LAMBDA(0,0) = GAMMA
          LAMBDA(0,1) = -GAMMA*BETA(1)
          LAMBDA(0,2) = -GAMMA*BETA(2)
          LAMBDA(0,3) = -GAMMA*BETA(3)

          LAMBDA(1,0) = LAMBDA(0,1)
          LAMBDA(1,1) = 1D0+(GAMMA-1D0)*BETA(1)*BETA(1)/BETA2
          LAMBDA(1,2) = (GAMMA-1D0)*BETA(1)*BETA(2)/BETA2
          LAMBDA(1,3) = (GAMMA-1D0)*BETA(1)*BETA(3)/BETA2

          LAMBDA(2,0) = LAMBDA(0,2)
          LAMBDA(2,1) = LAMBDA(1,2)
          LAMBDA(2,2) = 1D0+(GAMMA-1D0)*BETA(2)*BETA(2)/BETA2
          LAMBDA(2,3) = (GAMMA-1D0)*BETA(2)*BETA(3)/BETA2

          LAMBDA(3,0) = LAMBDA(0,3)
          LAMBDA(3,1) = LAMBDA(1,3)
          LAMBDA(3,2) = LAMBDA(2,3)
          LAMBDA(3,3) = 1D0+(GAMMA-1D0)*BETA(3)*BETA(3)/BETA2

C         APPLY TRANSFORMATION
          PBOOST = MATMUL(LAMBDA,P)
        END
      END
