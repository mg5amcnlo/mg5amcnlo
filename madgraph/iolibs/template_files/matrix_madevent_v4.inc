SUBROUTINE SMATRIX%(proc_id)s(P, RHEL, RCOL, DUM, IVEC, ANS, SELECTED_HEL, SELECTED_COL)
C  
%(info_lines)s
C 
C MadGraph5_aMC@NLO for Madevent Version
C 
C Returns amplitude squared summed/avg over colors
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
    Include '../../Source/vector.inc' 
    Include 'genps.inc'
    Include 'maxconfigs.inc'
    Include 'nexternal.inc'
    Include 'maxamps.inc'
    include 'run.inc'
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)d)
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d) 
    INTEGER    NDIAGS
    PARAMETER (NDIAGS=%(ndiags)d) 
    INTEGER    THEL
    PARAMETER (THEL=NCOMB)
C 
C ARGUMENTS 
C  
    REAL*8 P(0:3,NEXTERNAL),ANS
    REAL*8 RHEL, RCOL
    INTEGER DUM, SELECTED_HEL, SELECTED_COL, IVEC
c
c global (due to reading writting)
c 
    LOGICAL GOODHEL(NCOMB)
    INTEGER NTRY
    common/BLOCK_GOODHEL/NTRY,GOODHEL
    integer nb_spin_state(2)
    data  nb_spin_state /%(nb_spin_state1)i,%(nb_spin_state2)i/
    common /nb_hel_state/ nb_spin_state
C  
C LOCAL VARIABLES 
C  
    INTEGER NHEL(NEXTERNAL,NCOMB)
    REAL*8 T,MATRIX%(proc_id)s
    REAL*8 R,SUMHEL,TS(NCOMB)
    INTEGER I,IDEN
    INTEGER IPROC,JC(NEXTERNAL),II
    REAL*8 HWGT, XTOT, XTRY, XREJ, XR, YFRAC(0:NCOMB)
    INTEGER IDUM, NGOOD, J, JJ
    REAL     XRAN1
    EXTERNAL XRAN1

C  
C GLOBAL VARIABLES
C  
    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
    COMMON/TO_AMPS/  AMP2,       JAMP2
    
c    CHARACTER*101        HEL_BUFF
c    COMMON/TO_HELICITY/  HEL_BUFF
    
	REAL*8 POL(2)
    COMMON/TO_POLARIZATION/ POL
    
    INTEGER          ISUM_HEL
    LOGICAL                    MULTI_CHANNEL
    COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
%(define_iconfigs_lines)s
    DATA IDUM /-1/
    DATA XTRY, XREJ, NGOOD /0,0,0/
    SAVE YFRAC

%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
    NTRY=NTRY+1
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
c    WRITE(HEL_BUFF,'(20I5)') (0,I=1,NEXTERNAL)
    DO I=1,NCOMB
       TS(I)=0d0
    ENDDO

!   If the helicity grid status is 0, this means that it is not yet initialized.
    IF (ISUM_HEL.EQ.0.or.(DS_get_dim_status('Helicity').eq.0)) THEN
        DO I=1,NCOMB
           IF (GOODHEL(I) .OR. NTRY .LE. MAXTRIES.OR.(ISUM_HEL.NE.0)) THEN
               T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1), IVEC)            
            %(beam_polarization)s
			 IF (ISUM_HEL.NE.0) then
			   call DS_add_entry('Helicity',I,T)
			 endif
             ANS=ANS+DABS(T)
             TS(I)=T
           ENDIF
        ENDDO
        IF(NTRY.EQ.(MAXTRIES+1)) THEN
           call reset_cumulative_variable() ! avoid biais of the initialization
        ENDIF
	IF (ISUM_HEL.NE.0) then
!         We set HEL_PICKED to -HEL_PICKED here so that later on, the call to DS_add_point in dsample.f does not add anything to the grid since it was already done here.
		  HEL_PICKED = -iabs(HEL_PICKED)
!         For safety, hardset the helicity sampling jacobian to 0.0d0 to make sure it is not .
		  hel_jacobian   = 1.0d0
		  IF(DS_get_dim_status('Helicity').eq.1) then 
!           If we finished the initialization we can update the grid so as to start sampling over it.
!           However the grid will now be filled by dsample with different kind of weights (including pdf, flux, etc...) so by setting the grid_mode of the reference grid to 'initialization' we make sure it will be overwritten (as opposed to 'combined') by the running grid at the next update.
            CALL DS_UPDATE_GRID('Helicity')
			CALL DS_SET_GRID_MODE('Helicity','init')
          endif
	    ELSE
          IF(NTRY.LE.MAXTRIES)THEN
           DO I=1,NCOMB
              IF (.NOT.GOODHEL(I) .AND. (DABS(TS(I)).GT.ANS*LIMHEL/NCOMB)) THEN
                 GOODHEL(I)=.TRUE.
                 NGOOD = NGOOD +1
                 print *,'Adding good helicity ',I,TS(I)/ANS
              ENDIF
           ENDDO
          ENDIF
          IF(NTRY.EQ.MAXTRIES)THEN
             ISUM_HEL=MIN(ISUM_HEL,NGOOD)
          ENDIF
		endif
    ELSE              !RANDOM HELICITY
       
C           The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
            I = HEL_PICKED
			
			T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1), IVEC)            
            %(beam_polarization)s
c           Always one helicity at a time
            ANS = T
c           Include the Jacobian from helicity sampling
            ANS = ANS * hel_jacobian
            SELECTED_HEL = I
c            WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)  
    ENDIF
    IF (ISUM_HEL .NE. 1.or.(HEL_PICKED.le.0)) THEN
    R=XRAN1(IDUM)*ANS
    SUMHEL=0d0
    DO I=1,NCOMB
       SUMHEL=SUMHEL+DABS(TS(I))
       IF(R.LT.SUMHEL)THEN
           SELECTED_HEL = I
c          WRITE(HEL_BUFF,'(20i5)')(NHEL(II,I),II=1,NEXTERNAL)
          ANS=DSIGN(ANS,TS(I))		  
          GOTO 10
       ENDIF
    ENDDO
 10 CONTINUE   
    ENDIF
    IF (MULTI_CHANNEL) THEN
        XTOT=0D0
        DO I=1,NDIAGS
            XTOT=XTOT+AMP2(I)
        ENDDO
        IF (XTOT.NE.0D0) THEN
%(set_amp2_line)s
        ELSE IF(ANS.ne.0d0) THEN
			write(*,*) "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"
			stop 1
        ENDIF
    ENDIF
    ANS=ANS/DBLE(IDEN)

    CALL SELECT_COLOR(RCOL, JAMP2, ICONFIG, 1,  SELECTED_COL)

    END
 
 
REAL*8 FUNCTION MATRIX%(proc_id)s(P,NHEL,IC, IVEC)
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
    include '../../Source/vector.inc'
    include 'genps.inc'
    include 'nexternal.inc'
    include 'maxamps.inc'
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
    INTEGER IVEC
C  
C LOCAL VARIABLES 
C  
    INTEGER I,J,M,N
    COMPLEX*16 ZTEMP, TMP_JAMP(%(nb_temp_jamp)i)
    INTEGER CF(NCOLOR*(NCOLOR+1))
    INTEGER CF_INDEX,DENOM
    COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
    COMPLEX*16 W(18,NWAVEFUNCS)
C   Needed for v4 models
    COMPLEX*16 DUM0,DUM1
    DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/

	%(fake_width_declaration)s
	logical first
	data first /.true./
	save first
C
C FUNCTION
C
      INTEGER SQSOINDEX%(proc_id)s
C  
C GLOBAL VARIABLES
C  
    Double Precision amp2(maxamps), jamp2(0:maxflow)
    common/to_amps/  amp2,       jamp2
    include 'coupl.inc'

	double precision small_width_treatment
	common/narrow_width/small_width_treatment
C  
C COLOR DATA
C  
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
if (first) then
	first=.false.
	%(fake_width_definitions)s
endif

%(helas_calls)s
%(jamp_lines)s

    MATRIX%(proc_id)s = 0.D0 
	DO M = 1, NAMPSO
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX +1
          ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J,M)
        ENDDO
		DO N = 1, NAMPSO
          IF (CHOSEN_SO_CONFIGS(SQSOINDEX%(proc_id)s(M,N))) THEN
		     MATRIX%(proc_id)s = MATRIX%(proc_id)s + ZTEMP*DCONJG(JAMP(I,N))
		  ENDIF
		ENDDO
      ENDDO
	ENDDO
     MATRIX%(proc_id)s = MATRIX%(proc_id)s / DENOM

%(amp2_lines)s
    Do I = 1, NCOLOR
	  DO M = 1, NAMPSO	
		DO N = 1, NAMPSO
          IF (CHOSEN_SO_CONFIGS(SQSOINDEX%(proc_id)s(M,N))) THEN		
            Jamp2(i)=Jamp2(i)+DABS(DBLE(Jamp(i,m)*dconjg(Jamp(i,n))))
		  ENDIF
		enddo
	  enddo
    Enddo

    END
