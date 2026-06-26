      SUBROUTINE %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS_SUMMED)
C
C     Simple routine wrapper to provide the same interface for
C     backward compatibility for usage without split orders.
C
C
C CONSTANTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL), ANS_SUMMED
      INTEGER FLAV_IDX
C
C VARIABLES
C
      INTEGER I
      REAL*8 ANS(0:NSQAMPSO)
C
C BEGIN CODE
C
      CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
      ANS_SUMMED=ANS(0)

      END

      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
      INTEGER HEL
      INTEGER FLAV_IDX
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIXHEL_SPLITORDERS(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
      INTEGER HEL
      INTEGER FLAV_IDX
C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
C ----------
C BEGIN CODE
C ----------
      USERHEL=HEL
      CALL %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIX_SPLITORDERS(P, FLAV_IDX, ANS)
C
%(info_lines)s
C
C MadGraph StandAlone Version
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
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER NSQAMPSO
      PARAMETER (NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      DATA CHOSEN_SO_CONFIGS/%(chosen_so_configs)s/
      COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS(0:NSQAMPSO)
C     FLAV_IDX (the dummy argument) is declared with NFLAV below.
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      REAL*8 T(NSQAMPSO), BUFF
      INTEGER IHEL,IDEN, I, J
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/%(beamone_helavgfactor)d,%(beamtwo_helavgfactor)d/
      INTEGER JC(NEXTERNAL)
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER NNTRY_FLAV, NGOODHEL_FLAV
      PARAMETER (NNTRY_FLAV=NFLAV)
      PARAMETER (NGOODHEL_FLAV=NCOMB*NFLAV)
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
      INTEGER NTRY(NFLAV)
      LOGICAL GOODHEL(NCOMB,NFLAV)
      DATA NTRY/NNTRY_FLAV*0/
      DATA GOODHEL/NGOODHEL_FLAV*.FALSE./
%(den_factor_line)s
C
C GLOBAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB)
%(helicity_lines)s
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL

      INTEGER USERHEL
      DATA USERHEL/-1/
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
      LOGICAL HELRESET
      common/%(proc_prefix)shelreset/HELRESET
      data HELRESET/.true./

      INTEGER POLARIZATIONS(0:NEXTERNAL,0:5)
      COMMON/%(proc_prefix)sBORN_BEAM_POL/POLARIZATIONS
      DATA ((POLARIZATIONS(I,J),I=0,NEXTERNAL),J=0,5)/NPOLENTRIES*-1/

C
C     FUNCTIONS
C
      LOGICAL %(proc_prefix)sIS_BORN_HEL_SELECTED
      INTEGER %(proc_prefix)sBROKEN_SYM

c----------
c   Check if helreset mode is on
c---------
 if (HELRESET) then
   do i=1,NFLAV
      NTRY(i) = 0
   enddo
   do i=1,NCOMB
      do j=1,NFLAV
         GOODHEL(I,j) = .false.
      enddo
   enddo
    HELRESET = .false.
 endif


C ----------
C BEGIN CODE
C ----------
C     FLAV_IDX=0 (or out of range): GET_FLAVOR_INDEX could not resolve the
C     requested flavor -- it is not an allowed combination, so its matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS(:) = 0d0
        RETURN
      ENDIF
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
      IF(USERHEL.EQ.-1) NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      DO IHEL=1,NEXTERNAL
         JC(IHEL) = +1
      ENDDO
      ANS(:) = 0d0
C     When spin-2 particles are involved, the Helicity filtering is dangerous for the 2->1 topology.
C     This is because depending on the MC setup the initial PS points have back-to-back initial states
C     for which some of the spin-2 helicity configurations are zero. But they are no longer zero
C     if the point is boosted on the z-axis. Remember that HELAS helicity amplitudes are no longer
C     lorentz invariant with expternal spin-2 particles (only the helicity sum is).
C     For this reason, we simply remove the filterin when there is only three external particles.
      IF (NEXTERNAL.LE.3) THEN
        GOODHEL(:,:) = .True.
      ENDIF
      DO IHEL=1,NCOMB
         IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
           IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 20 .OR.USERHEL.NE.-1) THEN
              IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL))) THEN
                CYCLE
              ENDIF
              CALL %(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX, T)
              BUFF=0D0
              DO I=1,NSQAMPSO
                IF(POLARIZATIONS(0,0).eq.-1.or.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL)) THEN
                  ANS(I)=ANS(I)+T(I)
                ENDIF
                BUFF=BUFF+T(I)
              ENDDO
              IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
                GOODHEL(IHEL,FLAV_IDX)=.TRUE.
              ENDIF
            ENDIF
          ENDIF
       ENDDO
      ANS(0)=0.0d0
      DO I=1,NSQAMPSO
        ANS(I)=ANS(I)/DBLE(IDEN)*%(proc_prefix)sBROKEN_SYM(FLAVOR)
        IF (CHOSEN_SO_CONFIGS(I)) THEN
          ANS(0)=ANS(0)+ANS(I)
        ENDIF
      ENDDO
      IF(USERHEL.NE.-1) THEN
        DO I=0,NSQAMPSO
          ANS(I)=ANS(I)*HELAVGFACTOR
        ENDDO
      ELSE
        DO J=1,NINITIAL
          IF (POLARIZATIONS(J,0).ne.-1) THEN
            DO I=0,NSQAMPSO
              ANS(I)=ANS(I)*BEAMS_HELAVGFACTOR(J)
              ANS(I)=ANS(I)/POLARIZATIONS(J,0)
            ENDDO
          ENDIF
        ENDDO
      ENDIF
      END


      SUBROUTINE %(proc_prefix)sMATRIX(P,NHEL,IC,FLAV_IDX,RES)
      use model_object
C
%(info_lines)s
C
C Returns amplitude squared summed/avg over colors
c for the point with external lines W(0:6,NEXTERNAL)
C
%(process_lines)s
C
      use aloha_object
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
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      INTEGER %(proc_prefix)sDENOM
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      COMMON /%(proc_prefix)sCOLOR_MATRIX/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
      REAL*8 RES(NSQAMPSO)
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N
      COMPLEX*16 ZTEMP
      COMPLEX*16 AMP(NGRAPHS)
      INTEGER CF_INDEX
      COMPLEX*16 JAMP(NCOLOR,NAMPSO)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C FUNCTION
C
      INTEGER %(proc_prefix)sSQSOINDEX
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C
C COLOR DATA
C
%(color_data_lines)s
C ----------
C BEGIN CODE
C ----------
       call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
c      WRITE (*,*) '  -> AMP = ', AMP
       call %(proc_prefix)sGET_JAMP(AMP,JAMP)
c      WRITE (*,*) '  -> JAMP = ', JAMP
       call %(proc_prefix)sGET_MATRIX(JAMP, RES)
c      write (*,*) "  -> col.ave. |M|^2 for HEL=[", NHEL ,"] = ", %(proc_prefix)sMATRIX

 %(amp2_lines)s

       END


       SUBROUTINE %(proc_prefix)sGET_NHEL(IDEN_STAR,NHEL_STAR)
C CONSTANTS
C
CF2PY INTENT(OUT) :: NHEL_STAR
CF2PY INTENT(OUT) :: IDEN_STAR
       INTEGER    NEXTERNAL
       PARAMETER (NEXTERNAL=%(nexternal)d)
       INTEGER                 NCOMB
       PARAMETER (             NCOMB=%(ncomb)d)

       INTEGER NHEL(NEXTERNAL,NCOMB),NHEL_STAR(NEXTERNAL,NCOMB)
       INTEGER IDEN,IDEN_STAR

 %(helicity_lines)s
 %(den_factor_line)s
       IDEN_STAR = IDEN
       NHEL_STAR = NHEL
       END


SUBROUTINE %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
      use model_object
C
%(process_lines)s
C
CF2PY INTENT(OUT) :: AMP
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: IC
CF2PY INTENT(IN) :: FLAV_IDX

      use aloha_object
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
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
      INTEGER FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      COMPLEX*16 AMP(NGRAPHS)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
      double precision bwcutoff
%(flavor_mask_decl)s
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C
C
      bwcutoff=15 ! use if $ syntax is defined in the process
%(flavor_mask_setup)s
%(helas_calls)s
%(amp2_lines)s
END


    SUBROUTINE %(proc_prefix)sGET_JAMP(AMP,JAMP)
C
%(process_lines)s
C
CF2PY INTENT(OUT) :: JAMP
CF2PY INTENT(IN) :: AMP

      IMPLICIT NONE
C
C CONSTANTS
C
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER ( NCOLOR=%(ncolor)d)
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      COMPLEX*16 AMP(NGRAPHS), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 JAMP(NCOLOR,NAMPSO)

      JAMP(:,:) = 0d0
%(jamp_lines)s
      END

      SUBROUTINE %(proc_prefix)sGET_MATRIX(JAMP, RES)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
CF2PY INTENT(OUT) :: MATRIX
CF2PY INTENT(IN) :: JAMP(NCOLOR,NAMPSO)


      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      REAL*8     ZERO,RES(NSQAMPSO)
      PARAMETER (ZERO=0D0)
      INTEGER %(proc_prefix)sSQSOINDEX
C
C LOCAL VARIABLES
C
      INTEGER I,J,M,N
      COMPLEX*16 ZTEMP

      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 JAMP(NCOLOR,NAMPSO), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C COLOR DATA
C

     RES(:) = 0.D0
     DO M = 1, NAMPSO
       CF_INDEX= 0
       DO I = 1, NCOLOR
         ZTEMP = (0.D0,0.D0)
         DO J = I, NCOLOR
             CF_INDEX = CF_INDEX +1
             ZTEMP = ZTEMP + %(proc_prefix)sCF(CF_INDEX)*JAMP(J,M)
         ENDDO
         DO N = 1, NAMPSO
           RES(%(proc_prefix)sSQSOINDEX(M,N)) = RES(%(proc_prefix)sSQSOINDEX(M,N)) + REAL(ZTEMP*DCONJG(JAMP(I,N)))
         ENDDO
       ENDDO
     ENDDO
     RES(:) = RES(:)/%(proc_prefix)sDENOM
     END

      SUBROUTINE %(proc_prefix)sGET_INTER(JAMP_1,JAMP_2, INTER)
      IMPLICIT NONE
CF2PY INTENT(OUT) :: INTER(NSQAMPSO)
CF2PY INTENT(IN) :: JAMP_1(NCOLOR,NAMPSO)
CF2PY INTENT(IN) :: JAMP_2(NCOLOR,NAMPSO)

       INTEGER NAMPSO, NSQAMPSO
       PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
       INTEGER %(proc_prefix)sSQSOINDEX

       INTEGER I,J,M,N
       INTEGER NCOLOR
       PARAMETER (NCOLOR=%(ncolor)d)
       INTEGER CF_INDEX
       INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
       INTEGER %(proc_prefix)sDENOM
       common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
       COMPLEX*16 JAMP_1(NCOLOR,NAMPSO),JAMP_2(NCOLOR,NAMPSO),ZTEMP,INTER(NSQAMPSO)
C
C COLOR DATA
C

      INTER(:) = 0.D0
      DO M = 1, NAMPSO
        CF_INDEX= 0
        DO I = 1, NCOLOR
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX +1
              DO N = 1, NAMPSO
                 INTER(%(proc_prefix)sSQSOINDEX(M,N)) =
     $            INTER(%(proc_prefix)sSQSOINDEX(M,N)) +
     $            %(proc_prefix)sCF(CF_INDEX)*(DCONJG(JAMP_2(I,N))*
     $            JAMP_1(J,M)+DCONJG(JAMP_2(J,N))*JAMP_1(I,M))
              ENDDO
           ENDDO
        ENDDO
      ENDDO
      INTER(:) = INTER(:)/(2d0*%(proc_prefix)sDENOM)
      RETURN
      END

     SUBROUTINE  %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, INTER_SUM)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER((NCOMB*NCOMB+1)/2i,NSQAMPSO) : all interference term (not the symmetric one)
      use model_object
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(OUT) :: INTER_SUM(N_COMB*(N_COMB+1)/2)
c

      INTEGER NB_NHEL
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NB_NHEL=%(ncomb)d)
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER THISNHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER NINTER
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE COMPLEX, ALLOCATABLE :: TMP_INTER(:,:)
      DOUBLE PRECISION ALPHAS
      DOUBLE COMPLEX INTER_SUM(*)
c LOCAL
      INTEGER I,IHEL,IPART
      INTEGER J
      DOUBLE PRECISION PI
C
C     GLOBAL
C
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      COMMON/%(proc_prefix)sCHOSEN_BORN_SQSO/CHOSEN_SO_CONFIGS

      INTEGER NHEL(NEXTERNAL,NB_NHEL),NTRY
C     put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
C
C     include coupling definition to update the value of alphas
C
      include 'coupl.inc'

      ninter = N_COMB*(N_COMB+1)/2
      allocate(TMP_INTER(NSQAMPSO, ninter))
      TMP_INTER(:,:) = (0d0, 0d0)

      do I = 1,  N_COMB*(N_COMB+1)/2
        INTER_SUM(I) = 0d0
      enddo

      IF (ALPHAS.ne.0d0) then
          pi = 3.141592653589793d0
          G = 2* DSQRT(ALPHAS*pi)
          call UPDATE_AS_PARAM()
      ENDIF

      DO IHEL =1, NB_NHEL
        THISNHEL(:) = NHEL(:, IHEL)
        DO IPART=1,N_CHANGING
            if(THISNHEL(POS(IPART)).NE.ALLOW_HEL(IPART)) GOTO 10 !BYPASS COMPUTATION FOR HELICITY
        ENDDO
        TMP_INTER(:,:) = 0
        call  %(proc_prefix)sGET_ALL_INTER(P, THISNHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, TMP_INTER)
        do J =1, NSQAMPSO
          IF (CHOSEN_SO_CONFIGS(J)) THEN
            do I = 1, N_COMB*(N_COMB+1)/2
              INTER_SUM(I) = INTER_SUM(I) + TMP_INTER(J,I)
            enddo
          endif
        enddo
 10   enddo
      return
      deallocate(TMP_INTER)
      end



      SUBROUTINE  %(proc_prefix)sGET_ALL_INTER(P, NHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, INTER)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER(NSO, (NCOMB*NCOMB+1)/2): all interference term (not the symmetric one)
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: NHEL(%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(OUT) :: INTER(NSQAMPSO,NCOMB*(NCOMB+1)/2)
c
C
C ARGUMENTS
C
      INTEGER NAMPSO, NSQAMPSO
      PARAMETER (NAMPSO=%(nAmpSplitOrders)d, NSQAMPSO=%(nSqAmpSplitOrders)d)
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE COMPLEX INTER(NSQAMPSO,*)
c
c     Intermediate array
c
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER IC(NEXTERNAL)

      DOUBLE COMPLEX AMP(NGRAPHS)
      DOUBLE COMPLEX, ALLOCATABLE:: JAMP(:,:,:)
      integer, save :: s_ncomb = -1
c
c     LOCAL
c
      INTEGER I,J,SOL,N
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX

C ----------
C BEGIN CODE
C ----------
      if (allocated(jamp).and. n_comb.ne.s_ncomb)then
         deallocate(jamp)
      endif
      if (.not.allocated(jamp))then
        allocate(jamp(ncolor, NAMPSO, N_COMB))
        s_ncomb=n_comb
      endif
      IC(:)=1
      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
      do I = 1, N_COMB
         do N = 1, N_CHANGING
            NHEL(POS(N)) = ALLOW_HEL((I-1)*N_CHANGING+N)
         enddo
         call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
         call %(proc_prefix)sGET_JAMP(AMP,JAMP(1,1,I))
      enddo

     SOL = 0
     DO I = 1, N_COMB
        DO J= I, N_COMB
          SOL = SOL +1
          call %(proc_prefix)sGET_INTER(JAMP(1,1,I), JAMP(1,1,J), INTER(1,SOL))
        ENDDO
     ENDDO


     RETURN
     END




      SUBROUTINE %(proc_prefix)sGET_value(P, ALPHAS, NHEL, FLAVOR ,ANS)
C     f2py interface accepting the full FLAVOR(NEXTERNAL) array (back-compat):
C     resolve it to FLAV_IDX and forward to GET_value_idx.
      use model_object
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
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)

      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
      CALL %(proc_prefix)sGET_value_idx(P, ALPHAS, NHEL, FLAV_IDX, ANS)
      return
      end

      SUBROUTINE %(proc_prefix)sGET_value_idx(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
C     f2py interface accepting the flavor index directly.
      use model_object
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
      INTEGER FLAV_IDX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAV_IDX
C     the include file with the values of the parameters and masses
      include "coupl.inc"

      pi = 3.141592653589793d0
      G = 2* DSQRT(ALPHAS*pi)
      call UPDATE_AS_PARAM()
      if (NHEL.ne.0) then
        CALL %(proc_prefix)sSMATRIXHEL(P, NHEL, FLAV_IDX, ANS)
      else
        CALL %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
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

%(broken_sym_function)s


%(flavor_index_function)s


%(flavor_array_function)s
