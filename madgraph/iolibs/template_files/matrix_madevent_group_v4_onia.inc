C MadOnia
      SUBROUTINE SMATRIX%(proc_id)s(P,ANS)
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
c
c global (due to reading writting) 
c
      LOGICAL GOODHEL(NCOMB,2)
      INTEGER NTRY(2)
      COMMON/BLOCK_GOODHEL/NTRY,GOODHEL

C  
C LOCAL VARIABLES 
C
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      INCLUDE 'config_subproc_map.inc'
      INTEGER NHEL(NEXTERNAL,NCOMB)
      INTEGER ISHEL(2)
      REAL*8 T,MATRIX%(proc_id)s
      REAL*8 R,SUMHEL,TS(NCOMB)
      INTEGER I,IDEN
      INTEGER JC(NEXTERNAL),II
      REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
      INTEGER NGOOD(2), IGOOD(NCOMB,2)
      INTEGER JHEL(2), J, JJ
      INTEGER THIS_NTRY(2)
      SAVE THIS_NTRY
      INTEGER NB_FAIL
      SAVE NB_FAIL
      DATA THIS_NTRY /0,0/
      DATA NB_FAIL /0/
      REAL*8 GET_CHANNEL_CUT
      EXTERNAL GET_CHANNEL_CUT

C
C MADONIA VARIABLES
C
      INCLUDE 'onia.inc'

      LOGICAL IS_PROJ_COMPUTED(NONIA,-1:1)
      COMMON/%(proc_prefix)sIS_PROJ_COMPUTED/IS_PROJ_COMPUTED

      LOGICAL IS_JAMP_COMPUTED(4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/%(proc_prefix)sIS_JAMP_COMPUTED/IS_JAMP_COMPUTED
        
c
C This is just to temporarily store the reference grid for helicity of the DiscreteSampler so as to obtain its number of entries with ref_helicity_grid%%n_tot_entries
      TYPE(SAMPLEDDIMENSION) REF_HELICITY_GRID
C  
C GLOBAL VARIABLES
C  
      LOGICAL INIT_MODE
      COMMON /TO_DETERMINE_ZERO_HEL/INIT_MODE
      REAL*8 AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
      COMMON/TO_AMPS/AMP2, JAMP2
     
      CHARACTER*101 HEL_BUFF
      COMMON/TO_HELICITY/ HEL_BUFF
  
      INTEGER NB_SPIN_STATE_IN(2)
      COMMON /NB_HEL_STATE/ NB_SPIN_STATE_IN
      
      INTEGER IMIRROR,IPROC
      COMMON/TO_MIRROR/ IMIRROR,IPROC

      REAL*8 TMIN_FOR_CHANNEL        
      INTEGER SDE_STRAT ! 1 means standard single diagram enhancement strategy,
c                         2 means approximation by the denominator of the propagator
      COMMON/TO_CHANNEL_STRAT/TMIN_FOR_CHANNEL, SDE_STRAT

      REAL*8 POL(2)
      COMMON/TO_POLARIZATION/ POL

      REAL*8 SMALL_WIDTH_TREATMENT
      COMMON/NARROW_WIDTH/SMALL_WIDTH_TREATMENT
    
      INTEGER ISUM_HEL
      LOGICAL MULTI_CHANNEL
      COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
      %(define_iconfigs_lines)s
      DATA XTRY, XREJ /0,0/
      DATA NGOOD /0,0/
      DATA ISHEL/0,0/
      SAVE YFRAC, IGOOD, JHEL
      %(helicity_lines)s
      %(den_factor_line)s

C To be able to control when the matrix<i> subroutine can add entries to the grid for the MC over helicity configuration
      LOGICAL ALLOW_HELICITY_GRID_ENTRIES
      COMMON/TO_ALLOW_HELICITY_GRID_ENTRIES/ALLOW_HELICITY_GRID_ENTRIES

C ----------
C BEGIN CODE
C ----------

      IS_PROJ_COMPUTED=.FALSE.
      IS_JAMP_COMPUTED=.FALSE.

      NTRY(IMIRROR)=NTRY(IMIRROR)+1
      THIS_NTRY(IMIRROR) = THIS_NTRY(IMIRROR)+1 
      DO I=1,NEXTERNAL
         JC(I) = +1
      ENDDO
       
      IF (multi_channel) THEN
          DO I=1,NDIAGS
              AMP2(I)=0D0
          ENDDO
          JAMP2(0)=%(ncolor)d
          DO I=1,INT(JAMP2(0))
              JAMP2(I)=0D0
          ENDDO
      ENDIF
      ANS = 0D0
      WRITE(HEL_BUFF,'(20I5)') (0,I=1,NEXTERNAL)
      DO I=1,NCOMB
         TS(I)=0d0
      ENDDO

!     If the helicity grid status is 0, this means that it is not yet initialized.
!     If HEL_PICKED==-1, this means that calls to other matrix<i> where in initialization mode as well for the helicity.
      IF ((ISHEL(IMIRROR).EQ.0.and.ISUM_HEL.eq.0).or.(DS_get_dim_status('Helicity').eq.0).or.(HEL_PICKED.eq.-1)) THEN
        DO I=1,NCOMB
          IF (GOODHEL(I,IMIRROR) .OR. NTRY(IMIRROR).LE.MAXTRIES.or.(ISUM_HEL.NE.0).or.THIS_NTRY(IMIRROR).le.10) THEN
            T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1),I)
            %(beam_polarization)s
            IF (ISUM_HEL.NE.0.and.DS_get_dim_status('Helicity').eq.0.and.ALLOW_HELICITY_GRID_ENTRIES) then
              call DS_add_entry('Helicity',I,T)
            ENDIF
            ANS=ANS+DABS(T)
            TS(I)=T
          ENDIF
        ENDDO
        IF(NTRY(IMIRROR).EQ.(MAXTRIES+1)) THEN
          call reset_cumulative_variable() ! avoid biais of the initialization
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
          JHEL(IMIRROR) = 1
          IF(NTRY(IMIRROR).LE.MAXTRIES.or.THIS_NTRY(IMIRROR).le.10)THEN
            DO I=1,NCOMB
              IF(init_mode) THEN
                IF (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB) THEN
                  PRINT *, 'Matrix Element/Good Helicity: %(proc_id)s ', i, 'IMIRROR', IMIRROR
                ENDIF
              ELSE IF (.NOT.GOODHEL(I,IMIRROR) .AND. (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB)) THEN
                GOODHEL(I,IMIRROR)=.TRUE.
                NGOOD(IMIRROR) = NGOOD(IMIRROR) +1
                IGOOD(NGOOD(IMIRROR),IMIRROR) = I
                PRINT *,'Added good helicity ',I,TS(I)*NCOMB/ANS,' in event ',NTRY(IMIRROR), 'local:',THIS_NTRY(IMIRROR)
              ENDIF
            ENDDO
          ENDIF
          IF(NTRY(IMIRROR).EQ.MAXTRIES)THEN
            ISHEL(IMIRROR)=MIN(ISUM_HEL,NGOOD(IMIRROR))
          ENDIF
        ENDIF
      ELSE IF (.not.init_mode) then            ! random helicity 
C       The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
        I = HEL_PICKED

        T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1),I)

        %(beam_polarization)s
c       Always one helicity at a time
        ANS = T
c       Include the Jacobian from helicity sampling
        ANS = ANS * hel_jacobian

        WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)
      ELSE
        ANS = 1D0
        RETURN
      ENDIF
      IF (ANS.ne.0d0.and.(ISUM_HEL .NE. 1.or.HEL_PICKED.eq.-1)) THEN
        CALL RANMAR(R)
        SUMHEL=0d0
        DO I=1,NCOMB
          SUMHEL=SUMHEL+DABS(TS(I))/ANS
          IF(R.LT.SUMHEL)THEN
            WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)
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
%(set_amp2_line)s
        ELSE IF(ANS.ne.0d0) THEN
          IF(NB_FAIL.ge.10)then
            WRITE(*,*) "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"
                        
            STOP 1
          ELSE
            NB_FAIL = NB_FAIL +1
          ENDIF
        ENDIF
      ENDIF
      ANS=ANS/DBLE(IDEN)
      END
 
 
      REAL*8 FUNCTION MATRIX%(proc_id)s(P,NHEL,IC, IHEL)
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
C  
C LOCAL VARIABLES 
C  
      INTEGER I,J,K,M,N,QQ,Q1,Q2
      REAL*8 FRAC
      COMPLEX*16 ZTEMP, TMP_JAMP(%(nb_temp_jamp)i)
      REAL*8 CF(NCOLOR,NCOLOR)
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
C MADONIA VAIABLES
C
      INCLUDE 'onia.inc'
      INCLUDE 'ldme.inc'
      REAL*8 NORM, LDME
      REAL*8 P_ONIA(0:3,NEXTERNAL+NONIA)
      COMPLEX*16 PROJ(4**NONIA)
      INTEGER NHEL_ONIA(NEXTERNAL+NONIA), IC_ONIA(NEXTERNAL+NONIA)
      INTEGER HELQ1(2), HELQ2(2)
      DATA HELQ1/-1, 1/
      DATA HELQ2/-1, 1/

      LOGICAL IS_PROJ_COMPUTED(NONIA,-1:1)
      COMPLEX*16 PROJ_SAVED(NONIA,-1:1,-1:1,-1:1)
      COMMON/IS_PROJ_COMPUTED%(proc_id)s/IS_PROJ_COMPUTED
      COMMON/PROJ_SAVED%(proc_id)s/PROJ_SAVED

      LOGICAL IS_JAMP_COMPUTED(4**NONIA,3**(NEXTERNAL-NONIA))
      COMPLEX*16 JAMP_SAVED(NCOLOR,NAMPSO,4**NONIA,3**(NEXTERNAL-NONIA))
      COMMON/IS_JAMP_COMPUTED%(proc_id)s/IS_JAMP_COMPUTED
      COMMON/JAMP_SAVED%(proc_id)s/JAMP_SAVED
C  
C GLOBAL VARIABLES
C  
      REAL*8 AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
      COMMON/TO_AMPS/AMP2,JAMP2
      INCLUDE 'coupl.inc'

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
      ENDDO
      LDME = %(ldme_product)s

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

            JAMP = (0D0,0D0)
%(jamp_lines)s
C           Store amplitudes for reuse
            JAMP_SAVED(:,:,J,K) = JAMP
            IS_JAMP_COMPUTED(J,K) = .TRUE.
          ENDIF
        ENDIF  
      ENDDO

      JAMP = (0D0,0D0)
      DO I = 1, 4**NONIA
        JAMP = JAMP + JAMP_SAVED(:,:,I,K)*PROJ(I)
      ENDDO    

      IF(INIT_MODE)THEN
        DO I=1, NGRAPHS   
          IF (AMP(I).NE.0) THEN
            ZEROAMP_%(proc_id)s(IHEL,I) = .FALSE.
         ENDIF
        ENDDO
      ENDIF       

      MATRIX%(proc_id)s = 0.D0 
      DO M = 1, NAMPSO
        DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = 1, NCOLOR
            ZTEMP = ZTEMP + CF(J,I)*JAMP(J,M)
          ENDDO
          DO N = 1, NAMPSO
            %(select_configs_if)s
            MATRIX%(proc_id)s = MATRIX%(proc_id)s + ZTEMP*DCONJG(JAMP(I,N))
            %(select_configs_endif)s
          ENDDO
        ENDDO
      ENDDO

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