DOUBLE PRECISION FUNCTION DSIG(PP,WGT,IMODE)
C ****************************************************
C
%(info_lines)s
C
%(process_lines)s
C
C     RETURNS DIFFERENTIAL CROSS SECTION 
c     FOR MULTIPLE PROCESSES IN PROCESS GROUP
C     Input:
C             pp    4 momentum of external particles
C             wgt   weight from Monte Carlo
C             imode 0 run, 1 init, 2 reweight,
C                   3 finalize, 4 only PDFs
C     Output:
C             Amplitude squared and summed
C ****************************************************
      use DiscreteSampler
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'genps.inc'
      include 'maxconfigs.inc'
      include 'nexternal.inc'
      include 'maxamps.inc'
      REAL*8     PI
      PARAMETER (PI=3.1415926d0)
C  
C ARGUMENTS 
C  
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER IMODE
C   
C LOCAL VARIABLES 
C  
      integer Lmapped
      INTEGER I,J,K,LUN,ICONF,IMIRROR,NPROC
      SAVE NPROC
      INTEGER SYMCONF(0:LMAXCONFIGS)
      SAVE SYMCONF
      DOUBLE PRECISION SUMPROB,TOTWGT,R,XDUM
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
      INCLUDE 'config_subproc_map.inc'            
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'
      LOGICAL MIRRORPROCS(MAXSPROC)
      INCLUDE 'mirrorprocs.inc'
C   SELPROC is vector of selection weights for the subprocesses
C   SUMWGT is vector of total weight for the subprocesses
C   NUMEVTS is vector of event calls for the subprocesses
      DOUBLE PRECISION SELPROC(2, MAXSPROC,LMAXCONFIGS)
      DOUBLE PRECISION SUMWGT(2, MAXSPROC,LMAXCONFIGS)
      INTEGER NUMEVTS(2, MAXSPROC,LMAXCONFIGS)
      INTEGER LARGEDIM
      PARAMETER (LARGEDIM=2*MAXSPROC*LMAXCONFIGS)
      DATA SELPROC/LARGEDIM*0d0/
      DATA SUMWGT/LARGEDIM*0d0/
      DATA NUMEVTS/LARGEDIM*0/
      SAVE SELPROC,SUMWGT,NUMEVTS
	  real*8 MC_grouped_proc_jacobian
      integer grouped_MC_grid_status	  
C  
C EXTERNAL FUNCTIONS
C  
      INTEGER NEXTUNOPEN
      DOUBLE PRECISION DSIGPROC
      EXTERNAL NEXTUNOPEN,DSIGPROC
C  
C GLOBAL VARIABLES
C  
      include 'coupl.inc'
      include 'run.inc'
C   ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     IPROC has the present process number
      INTEGER IPROC
      COMMON/TO_MIRROR/IMIRROR, IPROC
C     CM_RAP has parton-parton system rapidity
      DOUBLE PRECISION CM_RAP
      LOGICAL SET_CM_RAP
      COMMON/TO_CM_RAP/SET_CM_RAP,CM_RAP
C     Keep track of whether cuts already calculated for this event
      LOGICAL CUTSDONE,CUTSPASSED
      COMMON/TO_CUTSDONE/CUTSDONE,CUTSPASSED
C     To be able to control when the matrix<i> subroutine can add entries to the grid for the MC over helicity configuration
      LOGICAL ALLOW_HELICITY_GRID_ENTRIES
	  DATA ALLOW_HELICITY_GRID_ENTRIES/.True./
	  COMMON/TO_ALLOW_HELICITY_GRID_ENTRIES/ALLOW_HELICITY_GRID_ENTRIES
c     To limit the number of calls to switchmom, use in DSIGPROC the cached variable last_iconfig. It is in this subroutine as well so that we can set it to -1 to ignore caching (to prevent undesired effect if this subroutine is called from elsewhere and to 0 to reset the cache.
	  INTEGER LAST_ICONF
	  DATA LAST_ICONF/-1/
	  COMMON/TO_LAST_ICONF/LAST_ICONF

C ----------
C BEGIN CODE
C ----------
      DSIG=0D0

C     Make sure cuts are evaluated for first subprocess
      CUTSDONE=.FALSE.
      CUTSPASSED=.FALSE.

      IF(IMODE.EQ.1)THEN
C       Set up process information from file symfact
        LUN=NEXTUNOPEN()
        IPROC=1
        SYMCONF(IPROC)=ICONFIG
        OPEN(UNIT=LUN,FILE='../symfact.dat',STATUS='OLD',ERR=20)
        DO WHILE(.TRUE.)
          READ(LUN,*,ERR=10,END=10) XDUM, ICONF
          IF(ICONF.EQ.-MAPCONFIG(ICONFIG))THEN
            IPROC=IPROC+1
            SYMCONF(IPROC)=INT(XDUM)
          ENDIF
        ENDDO
 10     SYMCONF(0)=IPROC
        CLOSE(LUN)
        RETURN
 20     SYMCONF(0)=IPROC
        WRITE(*,*)'Error opening symfact.dat. No permutations used.'
        RETURN
      ELSE IF(IMODE.EQ.2)THEN
C     Output weights and number of events
        SUMPROB=0D0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              SUMPROB=SUMPROB+SUMWGT(K,I,J)
            ENDDO
          ENDDO
        ENDDO
        WRITE(*,*)'Relative summed weights:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(nsprocs)dE12.4)')((SUMWGT(K,I,J)/SUMPROB,K=1,2),I=1,MAXSPROC)
        ENDDO
        SUMPROB=0D0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              SUMPROB=SUMPROB+NUMEVTS(K,I,J)
            ENDDO
          ENDDO
        ENDDO
        WRITE(*,*)'Relative number of events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(nsprocs)dE12.4)')((NUMEVTS(K,I,J)/SUMPROB,K=1,2),I=1,MAXSPROC)
        ENDDO
        WRITE(*,*)'Events:'
        DO J=1,SYMCONF(0)
          WRITE(*,'(%(nsprocs)dI12)')((NUMEVTS(K,I,J),K=1,2),I=1,MAXSPROC)
        ENDDO
C       Reset weights and number of events
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            DO K=1,2
              NUMEVTS(K,I,J)=0
              SUMWGT(K,I,J)=0d0
            ENDDO
          ENDDO
        ENDDO
        RETURN
      ELSE IF(IMODE.EQ.3)THEN
C     No finalize needed
        RETURN
      ENDIF

C   IMODE.EQ.0, regular run mode
	  IF(MC_grouped_subproc.and.DS_get_dim_status('grouped_processes').eq.-1) then
	    call DS_register_dimension('grouped_processes', 0)
		call DS_set_min_points(10,'grouped_processes')
        DO J=1,SYMCONF(0)
          DO IPROC=1,MAXSPROC
            IF(CONFSUB(IPROC,SYMCONF(J)).NE.0) THEN
              DO IMIRROR=1,2
                IF(IMIRROR.EQ.1.OR.MIRRORPROCS(IPROC))THEN
                  CALL MAP_3_TO_1(J,IPROC,IMIRROR,MAXSPROC,2,LMAPPED)
                  CALL DS_add_bin('grouped_processes',LMAPPED)
                ENDIF
              ENDDO
            ENDIF
          ENDDO
        ENDDO
	  ENDIF
	  IF(MC_grouped_subproc.and.DS_dim_index(run_grid, 'PDF_convolution',.True.).eq.-1) then
	    call DS_register_dimension('PDF_convolution', 0, all_grids=.False.)
	  ENDIF
	
C     Select among the subprocesses based on PDF weight
      SUMPROB=0D0
C     Turn caching on in dsigproc to avoid too many calls to switchmom
      LAST_ICONF=0
      DO J=1,SYMCONF(0)
         DO IPROC=1,MAXSPROC
            IF(CONFSUB(IPROC,SYMCONF(J)).NE.0) THEN
               DO IMIRROR=1,2
                  IF(IMIRROR.EQ.1.OR.MIRRORPROCS(IPROC))THEN
C                 Calculate PDF weight for all subprocesses
                     SELPROC(IMIRROR,IPROC,J)=DSIGPROC(PP,J,IPROC,IMIRROR,SYMCONF,CONFSUB,1d0,4)
                     IF(MC_grouped_subproc) then
					   call map_3_to_1(J,IPROC,IMIRROR,MAXSPROC,2,Lmapped)
					   call DS_add_entry('PDF_convolution',Lmapped,SELPROC(IMIRROR,IPROC,J),.True.)
					 endif
                     SUMPROB=SUMPROB+SELPROC(IMIRROR,IPROC,J)
                     IF(IMIRROR.EQ.2)THEN
C                 Need to flip back x values
                        XDUM=XBK(1)
                        XBK(1)=XBK(2)
                        XBK(2)=XDUM
                        CM_RAP=-CM_RAP
                     ENDIF
                  ENDIF
               ENDDO
            endif
         ENDDO
       ENDDO
C      Turn caching in dsigproc back off to avoid side effects.
       LAST_ICONF=-1

c     Cannot make a selection with all PDFs to zero, so we return now
	  IF(SUMPROB.eq.0.0d0) then
	    RETURN
	  endif

C     Perform the selection
      CALL RANMAR(R)

C     It is important to cache the status before adding any entries to this grid in this
C     routine since it might change it
	  grouped_MC_grid_status = DS_get_dim_status('grouped_processes')
	  
	  IF (MC_grouped_subproc.and.grouped_MC_grid_status.eq.0) then
c       We must initialize the grid and probe all channels
        SUMPROB=0.0d0
C       Turn caching on in dsigproc to avoid too many calls to switchmom
        LAST_ICONF=0
        DO J=1,SYMCONF(0)
          DO I=1,MAXSPROC
            IF(CONFSUB(I,SYMCONF(J)).NE.0) THEN		  
              DO K=1,2
                IF(K.EQ.1.OR.MIRRORPROCS(I))THEN			
                  IPROC=I
                  ICONF=J
                  IMIRROR=K
C                 The IMODE=5 computes the matrix_element only, without PDF convolution 
                  DSIG=DSIGPROC(PP,ICONF,IPROC,IMIRROR,SYMCONF,CONFSUB,WGT,5)			  
			      call map_3_to_1(J,I,K,MAXSPROC,2,Lmapped)
			      IF (SELPROC(K,I,J).ne.0.0d0) then
		            call DS_add_entry('grouped_processes',Lmapped,DSIG)
		          endif
                  IF(K.EQ.2)THEN
C                   Need to flip back x values
                    XDUM=XBK(1)
                    XBK(1)=XBK(2)
                    XBK(2)=XDUM
                    CM_RAP=-CM_RAP
                  ENDIF
                  SELPROC(K,I,J) = DABS(DSIG*SELPROC(K,I,J))
                  SUMPROB = SUMPROB + SELPROC(K,I,J) 
		        ENDIF
              ENDDO
			ENDIF
          ENDDO
        ENDDO
C       Turn caching in dsigproc back off to avoid side effects.
        LAST_ICONF=-1
C       If these additional entries were enough to initialize the gird, then update it
C       To do this check we must *not* used the cached varianble grouped_MC_grid_status
		IF(DS_get_dim_status('grouped_processes').ge.1) then
		  CALL DS_update_grid('grouped_processes')
		  call reset_cumulative_variable()
		endif
      ENDIF

c     If we are still initializing the grid or simply not using one at all, then we pick a point based on PDF only.
	  IF (.not.MC_grouped_subproc.or.grouped_MC_grid_status.eq.0) then
	  R=R*SUMPROB
      ICONF=0
      IPROC=0
      TOTWGT=0D0
      DO J=1,SYMCONF(0)
        DO I=1,MAXSPROC
          DO K=1,2
            TOTWGT=TOTWGT+SELPROC(K,I,J)
            IF(R.LT.TOTWGT)THEN
                IPROC=I
                ICONF=J
                IMIRROR=K
                GOTO 50
             ENDIF
          ENDDO
        ENDDO
      ENDDO
 50   CONTINUE

      IF(IPROC.EQ.0) RETURN

C     Update weigth w.r.t SELPROC normalized to selection probability

      WGT=WGT*(SUMPROB/SELPROC(IMIRROR,IPROC,ICONF))

      ELSE
C       We are using the grouped_processes grid and it is initialized.
        call DS_get_point('grouped_processes',R,Lmapped,MC_grouped_proc_jacobian,'norm',(/'PDF_convolution'/))
		WGT=WGT*MC_grouped_proc_jacobian
		call map_1_to_3(Lmapped,MAXSPROC,2,ICONF,IPROC,IMIRROR)
	  ENDIF

C     Redo clustering to ensure consistent with final IPROC
      CUTSDONE=.FALSE.

	  IF(grouped_MC_grid_status.eq.0) then
C       If we were in the initialization phase of the grid for MC over grouped processes, we must instruct the matrix<i> subroutine not to add again an entry in the grid for this PS point at the call DSIGPROC just below.
        ALLOW_HELICITY_GRID_ENTRIES = .False.
      endif
C     Call DSIGPROC to calculate sigma for process
      DSIG=DSIGPROC(PP,ICONF,IPROC,IMIRROR,SYMCONF,CONFSUB,WGT,IMODE)
c     Reset ALLOW_HELICITY_GRID_ENTRIES
	  ALLOW_HELICITY_GRID_ENTRIES = .True. 

c     Below is how one would go about adding each point to the grouped_processes grid
C     However, keeping only the initialization grid is better because in that case all grouped ME's
C     were computed with the same kinematics. For this reason, the code below remains commented.
c	  IF(grouped_MC_grid_status.ge.1) then
c	    call map_3_to_1(ICONF,IPROC,IMIRROR,MAXSPROC,2,Lmapped)
c		call DS_add_entry('grouped_processes',Lmapped,(DSIG/SELPROC(IMIRROR,IPROC,ICONF)))
c	  ENDIF

      IF(DSIG.GT.0d0)THEN
C     Update summed weight and number of events
        SUMWGT(IMIRROR,IPROC,ICONF)=SUMWGT(IMIRROR,IPROC,ICONF)+DABS(DSIG*WGT)
        NUMEVTS(IMIRROR,IPROC,ICONF)=NUMEVTS(IMIRROR,IPROC,ICONF)+1
      ENDIF

      RETURN
      END

      FUNCTION DSIGPROC(PP,ICONF,IPROC,IMIRROR,SYMCONF,CONFSUB,WGT,IMODE)
C     ****************************************************
C     RETURNS DIFFERENTIAL CROSS SECTION 
C     FOR A PROCESS
C     Input:
C     pp    4 momentum of external particles
C     wgt   weight from Monte Carlo
C     imode 0 run, 1 init, 2 reweight, 3 finalize
C     Output:
C     Amplitude squared and summed
C     ****************************************************

      IMPLICIT NONE
      
      INCLUDE 'genps.inc'
      include 'maxconfigs.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INCLUDE 'coupl.inc'
      INCLUDE 'run.inc'
C     
C     ARGUMENTS 
C     
      DOUBLE PRECISION DSIGPROC
      DOUBLE PRECISION PP(0:3,NEXTERNAL), WGT
      INTEGER ICONF,IPROC,IMIRROR,IMODE
      INTEGER SYMCONF(0:LMAXCONFIGS)
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)
C     
C     GLOBAL VARIABLES
C     
C     SUBDIAG is vector of diagram numbers for this config
C     IB gives which beam is which (for mirror processes)
      INTEGER SUBDIAG(MAXSPROC),IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG,IB
C     ICONFIG has this config number
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     CM_RAP has parton-parton system rapidity
      DOUBLE PRECISION CM_RAP
      LOGICAL SET_CM_RAP
      COMMON/TO_CM_RAP/SET_CM_RAP,CM_RAP
c     To limit the number of calls to switchmom, use in DSIGPROC the cached variable last_iconfig. When set to -1, it ignores caching (to prevent undesired effect if this subroutine is called from elsewhere) and when set to 0, it resets the cache.
	  INTEGER LAST_ICONF
	  DATA LAST_ICONF/-1/
	  COMMON/TO_LAST_ICONF/LAST_ICONF
C  
C EXTERNAL FUNCTIONS
C  
%(dsig_def_line)s
      LOGICAL PASSCUTS
C     
C     LOCAL VARIABLES 
C     
      DOUBLE PRECISION P1(0:3,NEXTERNAL),xdum
      INTEGER I,J,K,JC(NEXTERNAL)
      INTEGER PERMS(NEXTERNAL,LMAXCONFIGS)
      INCLUDE 'symperms.inc'
	  SAVE P1,JC

      if (LAST_ICONF.eq.-1.or.LAST_ICONF.ne.ICONF) then

        ICONFIG=SYMCONF(ICONF)
        DO I=1,MAXSPROC
          SUBDIAG(I) = CONFSUB(I,SYMCONF(ICONF))
        ENDDO

C       Set momenta according to this permutation
        CALL SWITCHMOM(PP,P1,PERMS(1,MAPCONFIG(ICONFIG)),JC,NEXTERNAL)

	    if (LAST_ICONF.ne.-1) then
		  LAST_ICONF = ICONF
		endif
	  endif

      IB(1)=1
      IB(2)=2

      IF(IMIRROR.EQ.2)THEN
C       Flip momenta (rotate around x axis)
        DO I=1,NEXTERNAL
          P1(2,I)=-P1(2,I)
          P1(3,I)=-P1(3,I)
        ENDDO
C       Flip beam identity
        IB(1)=2
        IB(2)=1
C       Flip x values (to get boost right)
        XDUM=XBK(1)
        XBK(1)=XBK(2)
        XBK(2)=XDUM
C       Flip CM_RAP (to get rapidity right)
        CM_RAP=-CM_RAP
      ENDIF

      DSIGPROC=0D0

      IF (PASSCUTS(P1)) THEN
%(call_dsig_proc_lines)s
      ENDIF

	  IF (LAST_ICONF.ne.-1.and.IMIRROR.EQ.2) then
C       Flip back local momenta P1 if cached
        DO I=1,NEXTERNAL
          P1(2,I)=-P1(2,I)
          P1(3,I)=-P1(3,I)
        ENDDO
	  endif

      RETURN

      END


C     -----------------------------------------
C     Subroutine to map three positive integers
C     I, J and K with upper bounds J_bound and
C     K_bound to a one_dimensional
C     index L
C     -----------------------------------------

      subroutine map_3_to_1(I,J,K,J_bound,K_bound,L)
	  implicit none
	  integer, intent(in)  :: I,J,K,J_bound,K_bound
	  integer, intent(out) :: L

	  L = I*(J_bound*(K_bound+1)+K_bound+1)+J*(K_bound+1)+K

	  end subroutine map_3_to_1

C     -----------------------------------------
C     Subroutine to map back the positive 
C     integer L to the three integers 
C     I, J and K with upper bounds
C     J_bound and K_bound.
C     -----------------------------------------

      subroutine map_1_to_3(L,J_bound,K_bound,I,J,K)
	  implicit none
	  integer, intent(out)  :: I,J,K
	  integer, intent(in)   :: L, J_bound, K_bound
	  integer               :: L_run

      L_RUN = L
      I = L_RUN/(J_BOUND*(K_BOUND+1)+K_BOUND+1)
      L_RUN = L_RUN - I*((J_BOUND*(K_BOUND+1)+K_BOUND+1))
      J = L_RUN/(K_BOUND+1)
      L_RUN = L_RUN - J*(K_BOUND+1)
      K  = L_RUN

	  end subroutine map_1_to_3


C
C     Functionality to handling grid
C
%(read_write_good_hel)s



