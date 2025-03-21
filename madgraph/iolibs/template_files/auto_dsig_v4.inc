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
      double precision Qscale
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
C     STUFF FOR UPC
C
      double precision photonpdfsquare
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
      include '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
      include 'run.inc'
      include '../../Source/PDF/pdf.inc'
C     Common blocks
      double precision RHEL, RCOL
      integer selected_hel(VECSIZE_MEMMAX)
      integer selected_col(VECSIZE_MEMMAX)
c
c     local
c
	   double precision P1(0:3, nexternal)
      integer channel
C  
C DATA
C  
%(pdf_data)s
C ----------
C BEGIN CODE
C ----------
      selected_hel(:) = 0 
      selected_col(:) = 0
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
## if( nogrouping) {
      ! for no grouping update the scale here (done in main autodsig for grouping  
      call update_scale_coupling(PP, WGT)
## }	 
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

     channel = %(get_channel)s
	 call ranmar(RHEL)
	 call ranmar(RCOL)
         CALL SMATRIX%(proc_id)s(P1,RHEL, RCOL,channel,1, DSIGUU, selected_hel(1), selected_col(1))


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
	 
         DSIGUU=DSIGUU*REWGT(PP,1)

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
           CALL UNWGT(PP,DSIG%(proc_id)s*WGT,%(numproc)d,selected_hel(1), selected_col(1), 1)
        ENDIF
%(passcuts_end)s
      END
## if(read_write_good_hel) {
C
C     Functionality to handling grid
C
%(read_write_good_hel)s

## }

DOUBLE PRECISION FUNCTION DSIG%(proc_id)s_VEC(ALL_PP, ALL_XBK, ALL_Q2FACT, ALL_CM_RAP, ALL_WGT, IMODE, ALL_OUT, SYMCONF, CONFSUB, ICONF_VEC, IMIRROR_VEC, VECSIZE_USED)
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
      include '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
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
      DOUBLE PRECISION ALL_PP(0:3,NEXTERNAL,VECSIZE_MEMMAX)
      DOUBLE PRECISION ALL_WGT(VECSIZE_MEMMAX)
      DOUBLE PRECISION ALL_XBK(2,VECSIZE_MEMMAX)
      DOUBLE PRECISION ALL_Q2FACT(2,VECSIZE_MEMMAX)
      DOUBLE PRECISION ALL_CM_RAP(VECSIZE_MEMMAX)
      INTEGER IMODE
      DOUBLE PRECISION ALL_OUT(VECSIZE_MEMMAX)
      INTEGER IMIRROR_VEC(*)
      INTEGER ICONF_VEC(*)
      INTEGER VECSIZE_USED
      INTEGER SYMCONF(0:LMAXCONFIGS)
      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)      
C ----------
C BEGIN CODE
C ----------
C  
C LOCAL VARIABLES 
C
      INTEGER I,ITYPE,LP,IPROC
      DOUBLE PRECISION QSCALE
%(pdf_vars_vec)s      
      DOUBLE PRECISION XPQ(-7:7),PD(0:MAXPROC)
      DOUBLE PRECISION ALL_PD(0:MAXPROC, VECSIZE_MEMMAX) 
      DOUBLE PRECISION DSIGUU,R,RCONF
      INTEGER LUN,ICONF,IFACT,NFACT
      DATA NFACT/1/
      SAVE NFACT
      double precision RHEL ! random number
      integer channel
C
C     STUFF FOR DRESSED EE COLLISIONS
C     
      include '../../Source/PDF/eepdf.inc'
      double precision ee_comp_prod
%(ee_comp_vars_vec)s
      integer i_ee
C
C     STUFF FOR UPC
C
      double precision photonpdfsquare    
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
%(define_subdiag_lines)s
      include 'run.inc'

      double precision p_multi(0:3, nexternal, VECSIZE_MEMMAX)
      double precision hel_rand(VECSIZE_MEMMAX)
      double precision col_rand(VECSIZE_MEMMAX)
      integer selected_hel(VECSIZE_MEMMAX)
      integer selected_col(VECSIZE_MEMMAX)
      double precision all_rwgt(VECSIZE_MEMMAX)
      
C     Common blocks
      include '../../Source/PDF/pdf.inc'
c      CHARACTER*7         PDLABEL,EPA_LABEL
c      INTEGER       LHAID
c      COMMON/TO_PDF/LHAID,PDLABEL,EPA_LABEL     

c
c     local
c
	   double precision P1(0:3, nexternal)
	   integer ivec, curr_warp, iwarp, NB_WARP_USED
	   integer channels(VECSIZE_MEMMAX)
C  
C DATA
C  
%(pdf_data_vec)s
C ----------
C BEGIN CODE
C ----------
      selected_hel(:) = 0
      selected_col(:) = 0 
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

	 DO CURR_WARP=1, NB_WARP_USED
	   if(IMIRROR_VEC(CURR_WARP).EQ.1)then
	     IB(1) = 1
	     IB(2) = 2
	   else
	     IB(1) = 2
	     IB(2) = 1	   
	   endif
	   DO IWARP=1, WARP_SIZE
	      IVEC = (CURR_WARP-1)*WARP_SIZE+IWARP
c        Do not need those three here. do I?	 
	      XBK(:) = ALL_XBK(:,IVEC)
c	      CM_RAP = ALL_CM_RAP(IVEC)
	      Q2FACT(:) = ALL_Q2FACT(:, IVEC)


C     Select a flavor combination (need to do here for right sign)
       CALL RANMAR(R)
       IPSEL=0
       DO WHILE (R.GE.0D0 .AND. IPSEL.LT.IPROC)
          IPSEL=IPSEL+1
          R=R-DABS(ALl_PD(IPSEL,IVEC))/ALL_PD(0,IVEC)
       ENDDO
       %(get_channel_vec)s


       ALL_RWGT(IVEC) = REWGT(all_PP(0,1,IVEC), ivec)

	 if(frame_id.ne.6)then
           call boost_to_frame(ALL_PP(0,1,IVEC), frame_id, p_multi(0,1,IVEC))
         else
           p_multi(:,:,IVEC) = ALL_PP(:,:,IVEC)
         endif
	 CALL RANMAR(hel_rand(IVEC))
	 CALL RANMAR(col_rand(IVEC))
	 ENDDO ! end loop on IWARP/IVEC	 
	 ENDDO ! end loop on the CURR_WARP
         call SMATRIX%(proc_id)s_MULTI(p_multi, hel_rand, col_rand, channels, ALL_OUT , selected_hel, selected_col, VECSIZE_USED)


	 DO IVEC=1,VECSIZE_USED
	 DSIGUU = ALL_OUT(IVEC)
	 IF (IMODE.EQ.5) THEN
            IF (DSIGUU.LT.1D199) THEN		 
		      ALL_OUT(IVEC) = DSIGUU*CONV
			else
		      ALL_OUT(IVEC) = 0.0d0
			endif
			RETURN
		 ENDIF

	 XBK(:) = ALL_XBK(:,IVEC)
c	 CM_RAP = ALL_CM_RAP(IVEC)
	 Q2FACT(:) = ALL_Q2FACT(:, IVEC)

	 if(frame_id.ne.6)then
           call boost_to_frame(ALL_PP(0,1,IVEC), frame_id, P1)
         else
           P1 = ALL_PP(:,:,IVEC)
         endif
C	 call restore_cl_val_to(ivec)
c         DSIGUU=DSIGUU*REWGT(P1,ivec)
	  DSIGUU=DSIGUU*ALL_RWGT(IVEC)

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
           CALL UNWGT(ALL_PP(0,1,IVEC), ALL_OUT(IVEC)*ALL_WGT(IVEC),%(numproc)d, selected_hel(IVEC), selected_col(IVEC), IVEC)
        ENDIF
	ENDDO
%(passcuts_end)s
      END




     SUBROUTINE PRINT_ZERO_AMP%(proc_id)s()

     return
     end


     SUBROUTINE SMATRIX%(proc_id)s_MULTI(p_multi, hel_rand, col_rand, channels, out, selected_hel, selected_col, VECSIZE_USED)
%(OMP_LIB)s
     implicit none

     include 'nexternal.inc'
     include '../../Source/vector.inc' ! defines VECSIZE_MEMMAX
     include 'maxamps.inc'
     INTEGER                 NCOMB
     PARAMETER (             NCOMB=%(ncomb)d)
     double precision p_multi(0:3, nexternal, VECSIZE_MEMMAX)
     double precision hel_rand(VECSIZE_MEMMAX)
     double precision col_rand(VECSIZE_MEMMAX)
     integer channels(VECSIZE_MEMMAX)
     double precision out(VECSIZE_MEMMAX)
     integer selected_hel(VECSIZE_MEMMAX)
     integer selected_col(VECSIZE_MEMMAX)
     INTEGER VECSIZE_USED

     integer ivec

%(additional_header)s


%(OMP_PREFIX)s
      DO IVEC=1, VECSIZE_USED
      	 call SMATRIX%(proc_id)s(p_multi(0,1,IVEC),
     &	                         hel_rand(IVEC),
     &                           col_rand(IVEC),
     &				 channels(IVEC),
     &                           IVEC,
     &				 out(IVEC),
     &				 selected_hel(IVEC),
     &				 selected_col(IVEC)
     &				 )
      ENDDO
%(OMP_POSTFIX)s

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


%(ADDITIONAL_FCT)s

