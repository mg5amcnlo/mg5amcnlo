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
    Include 'genps.inc'
    Include 'maxconfigs.inc'
    Include 'nexternal.inc'
    Include 'maxamps.inc'
    INTEGER                 NCOMB         
    PARAMETER (             NCOMB=%(ncomb)s)
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
        common/BLOCK_GOODHEL/NTRY,GOODHEL

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
        double precision get_channel_cut
        external get_channel_cut
	
c
C       This is just to temporarily store the reference grid for helicity of the DiscreteSampler so as to obtain its number of entries with ref_helicity_grid%%n_tot_entries
	    type(SampledDimension) ref_helicity_grid
C  
C GLOBAL VARIABLES
C  
    logical init_mode
    common /to_determine_zero_hel/init_mode
    DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
    COMMON/TO_AMPS/  AMP2,       JAMP2
   
    CHARACTER*101         HEL_BUFF
    COMMON/TO_HELICITY/  HEL_BUFF

    INTEGER NB_SPIN_STATE_in(2)
    common /nb_hel_state/ nb_spin_state_in
    
    INTEGER IMIRROR,IPROC
    COMMON/TO_MIRROR/ IMIRROR,IPROC

       double precision tmin_for_channel	
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat

    REAL*8 POL(2)
    COMMON/TO_POLARIZATION/ POL

	double precision small_width_treatment
	common/narrow_width/small_width_treatment
    
    INTEGER          ISUM_HEL
    LOGICAL                    MULTI_CHANNEL
    COMMON/TO_MATRIX/ISUM_HEL, MULTI_CHANNEL
%(define_iconfigs_lines)s
    DATA XTRY, XREJ /0,0/
    DATA NGOOD /0,0/
    DATA ISHEL/0,0/
    SAVE YFRAC, IGOOD, JHEL
%(helicity_lines)s
%(den_factor_line)s

C  To be able to control when the matrix<i> subroutine can add entries to the grid for the MC over helicity configuration
   LOGICAL ALLOW_HELICITY_GRID_ENTRIES
   COMMON/TO_ALLOW_HELICITY_GRID_ENTRIES/ALLOW_HELICITY_GRID_ENTRIES

C ----------
C BEGIN CODE
C ----------

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

!   If the helicity grid status is 0, this means that it is not yet initialized.
!   If HEL_PICKED==-1, this means that calls to other matrix<i> where in initialization mode as well for the helicity.
    IF ((ISHEL(IMIRROR).EQ.0.and.ISUM_HEL.eq.0).or.(DS_get_dim_status('Helicity').eq.0).or.(HEL_PICKED.eq.-1)) THEN
      DO I=1,NCOMB
         IF (GOODHEL(I,IMIRROR) .OR. NTRY(IMIRROR).LE.MAXTRIES.or.(ISUM_HEL.NE.0).or.THIS_NTRY(IMIRROR).le.10) THEN
            T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1),I)
            %(beam_polarization)s
			IF (ISUM_HEL.NE.0.and.DS_get_dim_status('Helicity').eq.0.and.ALLOW_HELICITY_GRID_ENTRIES) then
			  call DS_add_entry('Helicity',I,T)
			endif
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
          endif
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
		endif
        IF(NTRY(IMIRROR).EQ.MAXTRIES)THEN
           ISHEL(IMIRROR)=MIN(ISUM_HEL,NGOOD(IMIRROR))
        ENDIF
      ENDIF
    ELSE IF (.not.init_mode) then            ! random helicity 
C           The helicity configuration was chosen already by genps and put in a common block defined in genps.inc.
            I = HEL_PICKED

            T=MATRIX%(proc_id)s(P ,NHEL(1,I),JC(1),I)

            %(beam_polarization)s
c           Always one helicity at a time
            ANS = T
c           Include the Jacobian from helicity sampling
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
c         Set right sign for ANS, based on sign of chosen helicity
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
	    if (J.ne.0) then
	    if(sde_strat.eq.1) then
	         AMP2(J) = AMP2(J) * GET_CHANNEL_CUT(P, I)
                 XTOT=XTOT+AMP2(J)
	    else
	    	 AMP2(J) = GET_CHANNEL_CUT(P, I)
                 XTOT=XTOT+AMP2(J)
	    endif	
            endif	
        ENDDO
        IF (XTOT.NE.0D0) THEN
%(set_amp2_line)s
        ELSE IF(ANS.ne.0d0) THEN
	     IF(NB_FAIL.ge.10)then
			write(*,*) "Problem in the multi-channeling. All amp2 are zero but not the total matrix-element"
			
	        stop 1
            else
	    nb_fail = nb_fail +1
	    endif
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
    INTEGER IHEL
C  
C LOCAL VARIABLES 
C  
    INTEGER I,J,M,N
    COMPLEX*16 ZTEMP, TMP_JAMP(%(nb_temp_jamp)i)
    REAL*8 CF(NCOLOR,NCOLOR)
    COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR,NAMPSO)
    COMPLEX*16 W(%(wavefunctionsize)d,NWAVEFUNCS)
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

	logical init_mode
	common/to_determine_zero_hel/init_mode

	logical zeroamp_%(proc_id)s(NCOMB,NGRAPHS)
	common/to_zeroamp_%(proc_id)s/zeroamp_%(proc_id)s

       double precision tmin_for_channel
       integer sde_strat ! 1 means standard single diagram enhancement strategy,
c      	      	      	   2 means approximation by the	denominator of the propagator
       common/TO_CHANNEL_STRAT/tmin_for_channel,	sde_strat
       
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

	if(init_mode) then
	   zeroamp_%(proc_id)s(:,:) = .true.
	endif	
endif


%(helas_calls)s

JAMP(:,:) = (0d0,0d0) 
%(jamp_lines)s

    if(init_mode)then
      DO I=1, NGRAPHS	
        if (AMP(I).ne.0) then
       	  zeroamp_%(proc_id)s(ihel,i) = .false.
       endif
      ENDDO
    endif	

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

    if(sde_strat.eq.1)then
%(amp2_lines)s
    endif 

    Do I = 1, NCOLOR
	  DO M = 1, NAMPSO	
		DO N = 1, NAMPSO
		  %(select_configs_if)s
            Jamp2(i)=Jamp2(i)+DABS(DBLE(Jamp(i,m)*dconjg(Jamp(i,n))))
		  %(select_configs_endif)s
		enddo
	  enddo
    Enddo

    END

    SUBROUTINE PRINT_ZERO_AMP_%(proc_id)s()

    IMPLICIT NONE
    INTEGER    NGRAPHS
    PARAMETER (NGRAPHS=%(ngraphs)d)

    INTEGER    NCOMB
    PARAMETER (NCOMB=%(ncomb)d)

    logical zeroamp_%(proc_id)s(NCOMB, NGRAPHS)
    common/to_zeroamp_%(proc_id)s/zeroamp_%(proc_id)s

    integer i,j
    logical all_false

    do i=1, NGRAPHS
       all_false = .true.
       do j=1,NCOMB
       	  if (.not.zeroamp_%(proc_id)s(j, i)) then
	     all_false = .false.
	     exit
	  endif
       enddo	  
       if (all_false) then
       	  write(*,*) 'Amplitude/ZEROAMP:', %(proc_id)s, i
       else
          do j=1,NCOMB
               if (zeroamp_%(proc_id)s(j, i)) then
	         write(*,*) 'HEL/ZEROAMP:', %(proc_id)s, j  , i
	      endif	    
           enddo	
       endif
    enddo   

    return
    end