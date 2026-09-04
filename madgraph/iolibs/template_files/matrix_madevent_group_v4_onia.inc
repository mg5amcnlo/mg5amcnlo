C MadSONS
      SUBROUTINE SMATRIX%(proc_id)s(P, RHEL, RCOL, channel, IVEC, ANS, IHEL, ICOL)
C
%(info_lines)s
C
C MadGraph5_aMC@NLO for Madevent Version
C
C Returns amplitude squared -- no average over initial state/symmetry factor
c and helicities
c for the point in phase space P(0:3,NEXTERNAL)
C
%(process_lines)s
C
      use DiscreteSampler
      IMPLICIT NONE
C
C CONSTANTS
C
      INCLUDE 'genps.inc'
      INCLUDE 'maxconfigs.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INCLUDE '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
      INCLUDE 'run.inc'
      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)s)
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NDIAGS
      PARAMETER (NDIAGS=%(ndiags)d)
      INTEGER    THEL
      PARAMETER (THEL=2*NCOMB)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      REAL*8 RHEL ! random number for selecting helicity
      REAL*8 RCOL ! random number for selecting helicity
      INTEGER CHANNEL ! channel to keep for the multi-channel
      INTEGER IVEC ! for using the correct coupling
c
c   output argument
c
    integer ihel, icol
c
c global (due to reading writting)
c
      LOGICAL GOODHEL(NCOMB, MAXSPROC)
      INTEGER NTRY(MAXSPROC)
      COMMON/BLOCK_GOODHEL/NTRY,GOODHEL

C
C LOCAL VARIABLES
C
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      INCLUDE 'config_subproc_map.inc'
      INTEGER NHEL(NEXTERNAL,NCOMB)
      INTEGER ISHEL
      REAL*8 T,MATRIX%(proc_id)s
      REAL*8 R,SUMHEL,TS(NCOMB)
      INTEGER I,IDEN
      INTEGER JC(NEXTERNAL),II
      REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
      INTEGER NGOOD
      INTEGER J, JJ
      INTEGER NB_FAIL
      SAVE NB_FAIL
      DATA NB_FAIL /0/
      REAL*8 GET_CHANNEL_CUT
      EXTERNAL GET_CHANNEL_CUT

C
C MadSONS VARIABLES
C
      INCLUDE 'onia.inc'

      LOGICAL IS_PROJ_COMPUTED%(proc_id)s(NONIA,-1:1)
      COMMON/IS_PROJ_COMPUTED_%(proc_id)s/IS_PROJ_COMPUTED%(proc_id)s

      LOGICAL IS_AMP_COMPUTED%(proc_id)s(4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/IS_AMP_COMPUTED_%(proc_id)s/IS_AMP_COMPUTED%(proc_id)s

c
C This is just to temporarily store the reference grid for helicity of the DiscreteSampler so as to obtain its number of entries with ref_helicity_grid%%n_tot_entries
      TYPE(SAMPLEDDIMENSION) REF_HELICITY_GRID
C
C GLOBAL VARIABLES
C
      LOGICAL INIT_MODE
      COMMON /TO_DETERMINE_ZERO_HEL/INIT_MODE
      REAL*8 AMP2(MAXAMPS), JAMP2(0:MAXFLOW)

      INTEGER NB_SPIN_STATE_IN(2)
      COMMON /NB_HEL_STATE/ NB_SPIN_STATE_IN

      INTEGER IMIRROR, IPROC
      COMMON/TO_MIRROR/ IMIRROR,IPROC

c       included vi run.inc
c       REAL*8 TMIN_FOR_CHANNEL
c       INTEGER SDE_STRAT ! 1 means standard single diagram enhancement strategy
c                         2 means approximation by the denominator of the propagator
c       COMMON/TO_CHANNEL_STRAT/TMIN_FOR_CHANNEL, SDE_STRAT

      REAL*8 POL(2)
      COMMON/TO_POLARIZATION/ POL

      INTEGER ISUM_HEL
      LOGICAL MULTI_CHANNEL
      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
      %(define_iconfigs_lines)s
      DATA XTRY, XREJ /0,0/
      DATA NGOOD /0/
      DATA ISHEL/0/
      SAVE YFRAC
      %(helicity_lines)s
      %(den_factor_line)s

C To be able to control when the matrix<i> subroutine can add entries to the grid for the MC over helicity configuration
      LOGICAL ALLOW_HELICITY_GRID_ENTRIES
      COMMON/TO_ALLOW_HELICITY_GRID_ENTRIES/ALLOW_HELICITY_GRID_ENTRIES

C ----------
C BEGIN CODE
C ----------

      IS_PROJ_COMPUTED%(proc_id)s=.FALSE.
      IS_AMP_COMPUTED%(proc_id)s=.FALSE.

      NTRY(%(proc_id)s)=NTRY(%(proc_id)s)+1
      DO I=1,NEXTERNAL
        JC(I) = +1
      ENDDO

      IF (multi_channel) THEN
        DO I=1,NDIAGS
          AMP2(I)=0D0
        ENDDO
      ENDIF
      JAMP2(0)=%(ncolor)d
      DO I=1,INT(JAMP2(0))
        JAMP2(I)=0D0
      ENDDO
      ANS = 0D0
      DO I=1,NCOMB
        TS(I)=0d0
      ENDDO

!     If the helicity grid status is 0, this means that it is not yet initialized.
!     If HEL_PICKED==-1, this means that calls to other matrix<i> where in initialization mode as well for the helicity.
      IF ((ISHEL.EQ.0.AND.ISUM_HEL.EQ.0).OR.(DS_GET_DIM_STATUS('Helicity').EQ.0).OR.(HEL_PICKED.EQ.-1)) THEN
        DO I=1,NCOMB
          IF (GOODHEL(I,%(proc_id)s) .OR. NTRY(%(proc_id)s).LE.MAXTRIES.OR.(ISUM_HEL.NE.0)) THEN
            T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1),I,AMP2, JAMP2, IVEC)
            %(beam_polarization)s
            IF (ISUM_HEL.NE.0.AND.DS_GET_DIM_STATUS('Helicity').EQ.0.AND.ALLOW_HELICITY_GRID_ENTRIES) THEN
              CALL DS_ADD_ENTRY('Helicity',I,T)
            ENDIF
            ANS=ANS+DABS(T)
            TS(I)=T
          ENDIF
        ENDDO
        IF(NTRY(%(proc_id)s).EQ.(MAXTRIES+1).AND.DS_GET_DIM_STATUS('Helicity').NE.-1) THEN
          CALL RESET_CUMULATIVE_VARIABLE() ! avoid biais of the initialization
        ENDIF
        IF (ISUM_HEL.NE.0) then
!         We set HEL_PICKED to -1 here so that later on, the call to DS_add_point in dsample.f does not add anything to the grid since it was already done here.
          HEL_PICKED = -1
!         For safety, hardset the helicity sampling jacobian to 0.0d0 to make sure it is not .
          hel_jacobian   = 1.0d0
!         We don't want to re-update the helicity grid if it was already updated by another matrix<i>, so we make sure that the reference grid is empty.
          ref_helicity_grid = DS_get_dimension(ref_grid,'Helicity')
          IF((DS_get_dim_status('Helicity').eq.1).and.(ref_helicity_grid%%n_tot_entries.eq.0)) then
!           If we finished the initialization we can update the grid so as to start sampling over it.
!           However the grid will now be filled by dsample with different kind of weights (including pdf, flux, etc...) so by setting the grid_mode of the reference grid to 'initialization' we make sure it will be overwritten (as opposed to 'combined') by the running grid at the next update.
            CALL DS_UPDATE_GRID('Helicity')
            CALL DS_SET_GRID_MODE('Helicity','init')
          ENDIF
        ELSE
          IF(NTRY(%(proc_id)s).LE.MAXTRIES)THEN
            DO I=1,NCOMB
              IF(INIT_MODE) THEN
                IF (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB) THEN
                  PRINT *, 'Matrix Element/Good Helicity: %(proc_id)s ', i, 'IMIRROR', IMIRROR
                ENDIF
              ELSE IF (.NOT.GOODHEL(I,%(proc_id)s) .AND. (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB)) THEN
                 GOODHEL(I,%(proc_id)s)=.TRUE.
                 NGOOD = NGOOD +1
                 PRINT *,'Added good helicity ',I, 'for process %(proc_id)s',TS(I)*NCOMB/ANS,' in event ',NTRY(%(proc_id)s)
              ENDIF
            ENDDO
          ENDIF
          IF(NTRY(%(proc_id)s).EQ.MAXTRIES)THEN
            ISHEL=MIN(ISUM_HEL,NGOOD)
          ENDIF
        ENDIF
      ELSE IF (.NOT.INIT_MODE) THEN            ! random helicity
C       The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
        I = HEL_PICKED

        T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1),I,AMP2, JAMP2, IVEC)

        %(beam_polarization)s
c       Always one helicity at a time
        ANS = T
c       Include the Jacobian from helicity sampling
        ANS = ANS * hel_jacobian
        IHEL = HEL_PICKED
      ELSE
        ANS = 1D0
        RETURN
      ENDIF
      IF (ANS.ne.0d0.and.(ISUM_HEL .NE. 1.or.HEL_PICKED.eq.-1)) THEN
c        CALL RANMAR(R) ! rhel passed as input
        SUMHEL=0d0
        DO I=1,NCOMB
          SUMHEL=SUMHEL+DABS(TS(I))/ANS
          IF(RHEL.LT.SUMHEL)THEN
            IHEL = I
c           Set right sign for ANS, based on sign of chosen helicity
            ANS=DSIGN(ANS,TS(I))
            GOTO 10
          ENDIF
        ENDDO
10 CONTINUE
      ENDIF
      IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,LMAXCONFIGS
          J = CONFSUB(%(proc_id)s, I)
          IF (J.NE.0) THEN
            IF(SDE_STRAT.EQ.1) THEN
              AMP2(J) = AMP2(J) * GET_CHANNEL_CUT(P, I)
              XTOT=XTOT+AMP2(J)
            ELSE
              AMP2(J) = GET_CHANNEL_CUT(P, I)
              XTOT=XTOT+AMP2(J)
            ENDIF
          ENDIF
        ENDDO
        IF (XTOT.NE.0D0) THEN
          ANS=ANS*AMP2(CHANNEL)/XTOT
        ELSE IF(ANS.NE.0D0) THEN
          IF(NB_FAIL.GE.10)THEN
            WRITE(*,*) "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"

            STOP 1
          ELSE
            NB_FAIL = NB_FAIL +1
          ENDIF
        ENDIF
      ENDIF
      ANS=ANS/DBLE(IDEN)

      CALL SELECT_COLOR(RCOL, JAMP2, ICONFIG,%(proc_id)s,  ICOL)

      END


      REAL*8 FUNCTION MATRIX%(proc_id)s(P,NHEL,IC, IHEL,AMP2, JAMP2, IVEC)
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
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)s)
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INTEGER    NWAVEFUNCS,     NCOLOR
      PARAMETER (NWAVEFUNCS=%(nwavefuncs)d, NCOLOR=%(ncolor)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
      SAVE CHOSEN_SO_CONFIGS
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER IHEL
      INTEGER IVEC
C
C LOCAL VARIABLES
C
      INTEGER I,J,K,M,N,QQ,Q1,Q2
      REAL*8 FRAC
      COMPLEX*16 ZTEMP, TMP_JAMP(%(nb_temp_jamp)i)
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM, CF_INDEX
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
      COMPLEX*16 W(%(wavefunctionsize)d,NWAVEFUNCS)
C   Needed for v4 models
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/

      %(fake_width_declaration)s
      LOGICAL first
      DATA FIRST /.TRUE./
      SAVE FIRST
C
C FUNCTION
C
      INTEGER SQSOINDEX%(proc_id)s
C
C MadSONS VAIABLES
C
      INCLUDE 'onia.inc'
      INCLUDE 'ldme.inc'
      REAL*8 NORM, LDME
      REAL*8 P_ONIA(0:3,NEXTERNAL+NONIA)
      REAL*8 P_RESH(0:3,NEXTERNAL+NONIA), PMASS_RESH(NEXTERNAL)
      COMPLEX*16 PROJ(4**NONIA)
      INTEGER NHEL_ONIA(NEXTERNAL+NONIA), IC_ONIA(NEXTERNAL+NONIA)
      INTEGER HELQ1(2), HELQ2(2)
      DATA HELQ1/-1, 1/
      DATA HELQ2/-1, 1/

      LOGICAL IS_PROJ_COMPUTED%(proc_id)s(NONIA,-1:1)
      COMPLEX*16 PROJ_SAVED%(proc_id)s(NONIA,-1:1,-1:1,-1:1)
      COMMON/IS_PROJ_COMPUTED_%(proc_id)s/IS_PROJ_COMPUTED%(proc_id)s
      COMMON/PROJ_SAVED_%(proc_id)s/PROJ_SAVED%(proc_id)s

      LOGICAL IS_AMP_COMPUTED%(proc_id)s(4**NONIA,3**(NEXTERNAL-NONIA))
      COMPLEX*16 AMP_SAVED%(proc_id)s(NGRAPHS,4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/IS_AMP_COMPUTED_%(proc_id)s/IS_AMP_COMPUTED%(proc_id)s
      COMMON/AMP_SAVED_%(proc_id)s/AMP_SAVED%(proc_id)s
C
C GLOBAL VARIABLES
C
      INCLUDE '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
      REAL*8 AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
      INCLUDE 'coupl.inc' ! needs VECSIZE_MEMMAX (defined in vector.inc)

      REAL*8 SMALL_WIDTH_TREATMENT
      COMMON/NARROW_WIDTH/SMALL_WIDTH_TREATMENT

      LOGICAL INIT_MODE
      COMMON/TO_DETERMINE_ZERO_HEL/INIT_MODE

      LOGICAL ZEROAMP_%(proc_id)s(NCOMB,NGRAPHS)
      COMMON/TO_ZEROAMP_%(proc_id)s/zeroamp_%(proc_id)s

      REAL*8 TMIN_FOR_CHANNEL
      INTEGER SDE_STRAT ! 1 means standard single diagram enhancement strategy,
c                          2 means approximation by the denominator of the propagator
      COMMON/TO_CHANNEL_STRAT/TMIN_FOR_CHANNEL,SDE_STRAT

      REAL*8 PMASS(NEXTERNAL)
      REAL*8 PMASS_ONIA(NEXTERNAL+NONIA)
      INCLUDE 'pmass.inc'
      INCLUDE 'pmassonia.inc'
C
C COLOR DATA
C
      %(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
      IF (FIRST) THEN
        FIRST=.FALSE.
        %(fake_width_definitions)s

        IF(INIT_MODE) THEN
          ZEROAMP_%(proc_id)s(:,:) = .TRUE.
        ENDIF
      ENDIF

      PMASS_RESH = 0.D0
      DO I = 1, NEXTERNAL+NONIA
        PMASS_RESH(MAPPING(I)) = PMASS_RESH(MAPPING(I))+PMASS_ONIA(I)
      ENDDO
      CALL RESHUFFLE_MOMENTA%(proc_id)s(P,PMASS_RESH,P_RESH)

      DO I = 1, NEXTERNAL+NONIA
        IF (PAIRS(I).NE.0) THEN
          FRAC = PMASS_ONIA(I)/PMASS_RESH(MAPPING(I))
        ELSE
          FRAC = 1.D0
        ENDIF
        P_ONIA(:,I) = FRAC*P_RESH(:,MAPPING(I))
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
        IF (.NOT.IS_PROJ_COMPUTED%(proc_id)s(I,NHEL(QQ))) THEN
          DO J = 1, SIZE(HELQ1)
            DO K = 1, SIZE(HELQ2)
              NHEL_ONIA(Q1) = HELQ1(J)
              NHEL_ONIA(Q2) = HELQ2(K)
              CALL ONIA_PROJ(P_ONIA(:,Q1),PMASS_ONIA(Q1),NHEL_ONIA(Q1),P_ONIA(:,Q2),PMASS_ONIA(Q2),NHEL_ONIA(Q2),P_RESH(:,QQ),PMASS_RESH(QQ),NHEL(QQ),S_ONIA(I),PROJ(1))
              PROJ_SAVED%(proc_id)s(I,NHEL(QQ),NHEL_ONIA(Q1),NHEL_ONIA(Q2)) = PROJ(1)
            ENDDO
          ENDDO
          IS_PROJ_COMPUTED%(proc_id)s(I,NHEL(QQ)) = .TRUE.
        ENDIF
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
      DO J = 1, 4**NONIA
        PROJ(J) = SQRT(NORM)
        DO I =1, NONIA
          QQ = ONIA(I)
          Q1 = FINDLOC(PAIRS, I, DIM=1)
          Q2 = FINDLOC(PAIRS,-I, DIM=1)
          NHEL_ONIA(Q1) = HELQ1(IBITS(J-1,2*I-2,1)+1)
          NHEL_ONIA(Q2) = HELQ2(IBITS(J-1,2*I-1,1)+1)
          PROJ(J) = PROJ(J) * PROJ_SAVED%(proc_id)s(I,NHEL(QQ),NHEL_ONIA(Q1),NHEL_ONIA(Q2))
        ENDDO
        IF (PROJ(J).NE.(0D0,0D0)) THEN
          IF (.NOT.IS_AMP_COMPUTED%(proc_id)s(J,K)) THEN
            %(helas_calls)s

C           Store amplitudes for reuse
            AMP_SAVED%(proc_id)s(:,J,K) = AMP
            IS_AMP_COMPUTED%(proc_id)s(J,K) = .TRUE.
          ENDIF
        ENDIF
      ENDDO

      AMP = (0D0,0D0)
      DO I = 1, 4**NONIA
        AMP = AMP + AMP_SAVED%(proc_id)s(:,I,K)*PROJ(I)
      ENDDO

      JAMP = (0D0,0D0)
%(jamp_lines)s

      IF(INIT_MODE)THEN
        DO I=1, NGRAPHS
          IF (AMP(I).NE.0) THEN
            ZEROAMP_%(proc_id)s(IHEL,I) = .FALSE.
         ENDIF
        ENDDO
      ENDIF

      MATRIX%(proc_id)s = 0.D0
      DO M = 1, NAMPSO
        CF_INDEX = 0
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX + 1
            ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
          ENDDO
          DO N = 1, NAMPSO
            %(select_configs_if)s
            MATRIX%(proc_id)s = MATRIX%(proc_id)s + ZTEMP*DCONJG(JAMP(I,N))*LDME
            %(select_configs_endif)s
          ENDDO
        ENDDO
      ENDDO
      MATRIX%(proc_id)s = MATRIX%(proc_id)s/DENOM

      IF(SDE_STRAT.EQ.1)THEN
        %(amp2_lines)s
      ENDIF

      DO I = 1, NCOLOR
        DO M = 1, NAMPSO
          DO N = 1, NAMPSO
            %(select_configs_if)s
            JAMP2(I)=JAMP2(I)+DABS(DBLE(JAMP(I,M)*DCONJG(JAMP(I,N))))
            %(select_configs_endif)s
          ENDDO
        ENDDO
      ENDDO

      END

      SUBROUTINE PRINT_ZERO_AMP_%(proc_id)s()

      IMPLICIT NONE
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)

      INTEGER    NCOMB
      PARAMETER (NCOMB=%(ncomb)d)

      LOGICAL ZEROAMP_%(proc_id)s(NCOMB, NGRAPHS)
      COMMON/TO_ZEROAMP_%(proc_id)s/ZEROAMP_%(proc_id)s

      INTEGER I,J
      LOGICAL ALL_FALSE

      DO I=1, NGRAPHS
        ALL_FALSE = .TRUE.
        DO J=1,NCOMB
          IF (.NOT.ZEROAMP_%(proc_id)s(J, I)) THEN
            ALL_FALSE = .FALSE.
            EXIT
          ENDIF
        ENDDO
        IF (ALL_FALSE) THEN
          WRITE(*,*) 'AMPLITUDE/ZEROAMP:', %(proc_id)s, I
        ELSE
          DO J=1,NCOMB
            IF (ZEROAMP_%(proc_id)s(J, I)) THEN
              WRITE(*,*) 'HEL/ZEROAMP:', %(proc_id)s, J, I
            ENDIF
          ENDDO
        ENDIF
      ENDDO

      RETURN
      END

      SUBROUTINE RESHUFFLE_MOMENTA%(proc_id)s(P,PMASS,P_RESH)
      IMPLICIT NONE
C
C     CONSTANT
C
      INCLUDE 'nexternal.inc'
      INCLUDE 'onia.inc'
c      INCLUDE 'run.inc'
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
