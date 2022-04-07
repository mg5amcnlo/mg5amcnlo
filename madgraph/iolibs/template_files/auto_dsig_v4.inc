DOUBLE PRECISION FUNCTION DSIG%(proc_id)s(PP,WGT,IMODE)
C ****************************************************
C
%(info_lines)s
C
%(process_lines)s
C
C     RETURNS DIFFERENTIAL CROSS SECTION
C     Input:
C             pp    4 momentum of external particles
C             wgt   weight from Monte Carlo
C             imode 0 run, 1 init, 2 reweight, 
C                   3 finalize, 4 only PDFs,
C                   5 squared amplitude only (never
C                     generate events)
C     Output:
C             Amplitude squared and summed
C ****************************************************
      IMPLICIT NONE
C  
C CONSTANTS
C  
      include 'genps.inc'
      include 'nexternal.inc'
      include 'maxconfigs.inc'
      include 'maxamps.inc'
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379.66*1000)  !CONV TO PICOBARNS
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
      INTEGER I,ITYPE,LP,IPROC
%(pdf_vars)s      
      DOUBLE PRECISION XPQ(-7:7),PD(0:MAXPROC)
      DOUBLE PRECISION DSIGUU,R,RCONF

      INTEGER LUN,ICONF,IFACT,NFACT
      DATA NFACT/1/
      SAVE NFACT
C
C     STUFF FOR DRESSED EE COLLISIONS
C     
      include '../../Source/PDF/eepdf.inc'
      double precision ee_comp_prod
%(ee_comp_vars)s
      integer i_ee
C  
C EXTERNAL FUNCTIONS
C  
      LOGICAL PASSCUTS
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF,CUSTOM_BIAS
      INTEGER NEXTUNOPEN
C  
C GLOBAL VARIABLES
C  
      INTEGER          IPSEL
      COMMON /SubProc/ IPSEL
C     MINCFIG has this config number
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     Keep track of whether cuts already calculated for this event
      LOGICAL cutsdone,cutspassed
      COMMON/TO_CUTSDONE/cutsdone,cutspassed
%(define_subdiag_lines)s
      include 'coupl.inc'
      include 'run.inc'
C     Common blocks
      CHARACTER*7         PDLABEL,EPA_LABEL
      INTEGER       LHAID
      COMMON/TO_PDF/LHAID,PDLABEL,EPA_LABEL
c
c     local
c
	   double precision P1(0:3, nexternal)

C  
C DATA
C  
%(pdf_data)s
C ----------
C BEGIN CODE
C ----------
      DSIG%(proc_id)s=0D0
      %(cutsdone)s
      IF(IMODE.EQ.1)THEN
C       Set up process information from file symfact
        LUN=NEXTUNOPEN()
        NFACT=1
        OPEN(UNIT=LUN,FILE='../symfact.dat',STATUS='OLD',ERR=20)
        DO WHILE(.TRUE.)
          READ(LUN,*,ERR=10,END=10) RCONF, IFACT
          ICONF=INT(RCONF)
          IF(ICONF.EQ.MAPCONFIG(MINCFIG))THEN
             NFACT=IFACT
          ENDIF
        ENDDO
	DSIG%(proc_id)s = NFACT
 10     CLOSE(LUN)
        RETURN
 20     WRITE(*,*)'Error opening symfact.dat. No symmetry factor used.'
        RETURN
      ENDIF
C     Continue only if IMODE is 0, 4 or 5
      IF(IMODE.NE.0.AND.IMODE.NE.4.and.IMODE.NE.5) RETURN

%(passcuts_begin)s
%(pdf_lines)s
         IF (IMODE.EQ.4)THEN
            DSIG%(proc_id)s = PD(0)
            RETURN
         ENDIF
	 if(frame_id.ne.6)then
           call boost_to_frame(PP, frame_id, P1)
         else
           P1 = PP
         endif
c        CALL SMATRIX%(proc_id)s(P1,DSIGUU,1)
         stop 1 ! why is this called?
		 IF (IMODE.EQ.5) THEN
            IF (DSIGUU.LT.1D199) THEN		 
		      DSIG%(proc_id)s = DSIGUU*CONV
			else
		      DSIG%(proc_id)s = 0.0d0
			endif
			RETURN
		 ENDIF
C     Select a flavor combination (need to do here for right sign)
         CALL RANMAR(R)
         IPSEL=0
         DO WHILE (R.GE.0D0 .AND. IPSEL.LT.IPROC)
            IPSEL=IPSEL+1
            R=R-DABS(PD(IPSEL))/PD(0)
         ENDDO
	 
         DSIGUU=DSIGUU*REWGT(PP)

C        Apply the bias weight specified in the run card (default is 1.0)
         DSIGUU=DSIGUU*CUSTOM_BIAS(PP,DSIGUU,%(numproc)d,1)

         DSIGUU=DSIGUU*NFACT

         IF (DSIGUU.LT.1D199) THEN
c           Set sign of dsig based on sign of PDF and matrix element
            DSIG%(proc_id)s=DSIGN(%(conv)spd(0)*DSIGUU,DSIGUU*PD(IPSEL))
         ELSE
             write(*,*) "Error in matrix element"
             DSIGUU=0D0
             DSIG%(proc_id)s=0D0
         ENDIF
C       Generate events only if IMODE is 0.
        IF(IMODE.EQ.0.AND.DABS(DSIG%(proc_id)s).GT.0D0)THEN
C       Call UNWGT to unweight and store events
           CALL UNWGT(PP,DSIG%(proc_id)s*WGT,%(numproc)d,1)
        ENDIF
%(passcuts_end)s
      END
## if(read_write_good_hel) {
C
C     Functionality to handling grid
C
%(read_write_good_hel)s

## }

DOUBLE PRECISION FUNCTION DSIG%(proc_id)s_VEC(ALL_PP, ALL_XBK, ALL_Q2FACT, ALL_CM_RAP, ALL_WGT,IMODE, ALL_OUT)
C ****************************************************
C
%(info_lines)s
C
%(process_lines)s
C
C     RETURNS DIFFERENTIAL CROSS SECTION
C     Input:
C             pp    4 momentum of external particles
C             wgt   weight from Monte Carlo
C             imode 0 run, 1 init, 2 reweight, 
C                   3 finalize, 4 only PDFs,
C                   5 squared amplitude only (never
C                     generate events)
C     Output:
C             Amplitude squared and summed
C ****************************************************
      IMPLICIT NONE
C  
C CONSTANTS
C
      include '../../Source/vector.inc'
      include 'genps.inc'
      include 'nexternal.inc'
      include 'maxconfigs.inc'
      include 'maxamps.inc'
      DOUBLE PRECISION       CONV
      PARAMETER (CONV=389379.66*1000)  !CONV TO PICOBARNS
      REAL*8     PI
      PARAMETER (PI=3.1415926d0)
C  
C ARGUMENTS 
C  
      DOUBLE PRECISION ALL_PP(0:3,NEXTERNAL,NB_PAGE)
      DOUBLE PRECISION ALL_WGT(NB_PAGE)
      DOUBLE PRECISION ALL_XBK(2,NB_PAGE)
      DOUBLE PRECISION ALL_Q2FACT(2,NB_PAGE)
      DOUBLE PRECISION ALL_CM_RAP(NB_PAGE)
      INTEGER IMODE
      DOUBLE PRECISION ALL_OUT(NB_PAGE)
C ----------
C BEGIN CODE
C ----------
C  
C LOCAL VARIABLES 
C  
      INTEGER I,ITYPE,LP,IPROC
%(pdf_vars_vec)s      
      DOUBLE PRECISION XPQ(-7:7),PD(0:MAXPROC)
      DOUBLE PRECISION ALL_PD(0:MAXPROC, NB_PAGE) 
      DOUBLE PRECISION DSIGUU,R,RCONF
      INTEGER LUN,ICONF,IFACT,NFACT
      DATA NFACT/1/
      SAVE NFACT
      double precision RHEL ! random number
      integer channel
C
C     STUFF FOR DRESSED EE COLLISIONS --even if not supported for now--
C     
      include '../../Source/PDF/eepdf.inc'
      double precision ee_comp_prod
%(ee_comp_vars)s
      integer i_ee      
C  
C EXTERNAL FUNCTIONS
C  
      LOGICAL PASSCUTS
      DOUBLE PRECISION ALPHAS2,REWGT,PDG2PDF,CUSTOM_BIAS
      INTEGER NEXTUNOPEN
      DOUBLE PRECISION DSIG%(proc_id)s
C  
C GLOBAL VARIABLES
C  
      INTEGER          IPSEL
      COMMON /SubProc/ IPSEL
C     MINCFIG has this config number
      integer           mincfig, maxcfig
      common/to_configs/mincfig, maxcfig
      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG
C     Keep track of whether cuts already calculated for this event
      LOGICAL cutsdone,cutspassed
      COMMON/TO_CUTSDONE/cutsdone,cutspassed
C     jamp2 information      
      Double Precision jamp2(0:maxflow, nb_page)
      common/to_jamps/       jamp2
%(define_subdiag_lines)s
      include 'coupl.inc'
      include 'run.inc'

      double precision p_multi(0:3, nexternal, nb_page)
      double precision hel_rand(nb_page)
      integer selected_hel(nb_page)
      
C     Common blocks
      CHARACTER*7         PDLABEL,EPA_LABEL
      INTEGER       LHAID
      COMMON/TO_PDF/LHAID,PDLABEL,EPA_LABEL     

c
c     local
c
	   double precision P1(0:3, nexternal)
	   integer ivec

C  
C DATA
C  
%(pdf_data_vec)s
C ----------
C BEGIN CODE
C ----------
      %(cutsdone)s
      IF(IMODE.EQ.1)THEN
	NFACT = DSIG%(proc_id)s(all_pp(0,1,1), all_wgt(1), IMODE)
	RETURN
      ENDIF	   

C     Continue only if IMODE is 0, 4 or 5
      IF(IMODE.NE.0.AND.IMODE.NE.4.and.IMODE.NE.5) RETURN

%(passcuts_begin)s
%(pdf_lines_vec)s

	 
         IF (IMODE.EQ.4)THEN
            ALL_OUT(:) = ALL_PD(0,:)
            RETURN
         ENDIF

	 DO IVEC=1,NB_PAGE
c        Do not need those three here do I?	 
	 XBK(:) = ALL_XBK(:,IVEC)
c	 CM_RAP = ALL_CM_RAP(IVEC)
	 Q2FACT(:) = ALL_Q2FACT(:, IVEC)

	 if(frame_id.ne.6)then
           call boost_to_frame(ALL_PP(0,1,IVEC), frame_id, p_multi(0,1,IVEC))
         else
           p_multi(:,:,IVEC) = ALL_PP(:,:,IVEC)
         endif
	 CALL RANMAR(hel_rand(IVEC))
c         CALL SMATRIX%(proc_id)s(P1, RHEL, channel, ALL_OUT(IVEC), JAMP2(0, IVEC), IVEC)
	 ENDDO
	 channel = %(get_channel)s

c	 do IVEC=1, NB_PAGE
c	 	 CALL SMATRIX%(proc_id)s(p_multi(0,1,IVEC), hel_rand(ivec), channel, ALL_OUT(IVEC), JAMP2(0, IVEC), IVEC)
c	 enddo 
         call SMATRIX%(proc_id)s_MULTI(p_multi, hel_rand,  channel, ALL_OUT , selected_hel, jamp2)


	 DO IVEC=1,NB_PAGE
	 DSIGUU = ALL_OUT(IVEC)
	 IF (IMODE.EQ.5) THEN
            IF (DSIGUU.LT.1D199) THEN		 
		      ALL_OUT(IVEC) = DSIGUU*CONV
			else
		      ALL_OUT(IVEC) = 0.0d0
			endif
			RETURN
		 ENDIF
C     Select a flavor combination (need to do here for right sign)
         CALL RANMAR(R)
         IPSEL=0
         DO WHILE (R.GE.0D0 .AND. IPSEL.LT.IPROC)
            IPSEL=IPSEL+1
            R=R-DABS(ALl_PD(IPSEL,IVEC))/ALL_PD(0,IVEC)
         ENDDO

	 XBK(:) = ALL_XBK(:,IVEC)
c	 CM_RAP = ALL_CM_RAP(IVEC)
	 Q2FACT(:) = ALL_Q2FACT(:, IVEC)

	 if(frame_id.ne.6)then
           call boost_to_frame(ALL_PP(0,1,IVEC), frame_id, P1)
         else
           P1 = ALL_PP(:,:,IVEC)
         endif
	 call restore_cl_val_to(ivec)
         DSIGUU=DSIGUU*REWGT(P1)

C        Apply the bias weight specified in the run card (default is 1.0)
         DSIGUU=DSIGUU*CUSTOM_BIAS(P1,DSIGUU,%(numproc)d, IVEC)

         DSIGUU=DSIGUU*NFACT

         IF (DSIGUU.LT.1D199) THEN
c           Set sign of dsig based on sign of PDF and matrix element
            ALL_OUT(IVEC)=DSIGN(%(conv)sall_pd(0,IVEC)*dsiguu,DSIGUU*ALL_PD(IPSEL,IVEC))
         ELSE
             write(*,*) "Error in matrix element"
             DSIGUU=0D0
             ALL_OUT(IVEC)=0D0
         ENDIF
C       Generate events only if IMODE is 0.
        IF(IMODE.EQ.0.AND.DABS(ALL_OUT(IVEC)).GT.0D0)THEN
C       Call UNWGT to unweight and store events
           CALL UNWGT(ALL_PP(0,1,IVEC), ALL_OUT(IVEC)*ALL_WGT(IVEC),%(numproc)d, IVEC)
        ENDIF
	ENDDO
%(passcuts_end)s
      END
## if(read_write_good_hel) {
C
C     Functionality to handling grid
C
%(read_write_good_hel)s

## }




     SUBROUTINE PRINT_ZERO_AMP%(proc_id)s()

     return
     end


     SUBROUTINE SMATRIX%(proc_id)s_MULTI(p_multi, hel_rand,  channel, out , selected_hel, jamp2_multi)

     USE OMP_LIB

     include 'nexternal.inc'
     include '../../Source/vector.inc'
     include 'maxamps.inc'
     double precision p_multi(0:3, nexternal, NB_PAGE)
     double precision hel_rand(nb_page)
     integer channel
     double precision out(nb_page)
     integer selected_hel(nb_page)
     double precision jamp2_multi(0:MAXFLOW, nb_page)

     integer ivec


!$OMP PARALLEL
!$OMP DO
      DO IVEC=1, NB_PAGE
      	 call SMATRIX%(proc_id)s(p_multi(0,1,IVEC),
     &	                         hel_rand(IVEC),
     &				 channel,
     &				 out(IVEC),
c     &				 selected_hel(IVEC),
     &				 jamp2_multi(0,IVEC),
     &				 IVEC
     &				 )
      ENDDO
!$OMP END DO
!$OMP END PARALLEL
     return
     end

	integer FUNCTION GET_NHEL%(proc_id)s(hel, ipart)
c         if hel>0 return the helicity of particule ipart for the selected helicity configuration
c         if hel=0 return the number of helicity state possible for that particle 
	  implicit none
	  integer hel,i, ipart
	  Include 'nexternal.inc'
	  integer one_nhel(nexternal)
	  INTEGER                 NCOMB
	  PARAMETER (             NCOMB=%(ncomb)d)
	  INTEGER NHEL(NEXTERNAL,0:NCOMB)
	  %(helicity_lines)s

	  get_nhel%(proc_id)s = nhel(ipart, iabs(hel))
	  return
	  end

