      SUBROUTINE %(proc_prefix)sSMATRIXHEL(P,HEL, FLAV_IDX, ANS)
      IMPLICIT NONE
C
C CONSTANT
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: HEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX

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
      CALL %(proc_prefix)sSMATRIX(P,FLAV_IDX,ANS)
      USERHEL=-1

      END

      SUBROUTINE %(proc_prefix)sSMATRIX(P, FLAV_IDX, ANS)
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
      INTEGER NPOLENTRIES
      PARAMETER (NPOLENTRIES=(NEXTERNAL+1)*6)
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
      INTEGER HELAVGFACTOR
      PARAMETER (HELAVGFACTOR=%(hel_avg_factor)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
C
C LOCAL VARIABLES
C
      INTEGER NHEL(NEXTERNAL,NCOMB)
C  put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
      REAL*8 T
      REAL*8 %(proc_prefix)sMATRIX
      INTEGER IHEL,IDEN, I, J
C     For a 1>N process, them BEAMTWO_HELAVGFACTOR would be set to 1.
      INTEGER BEAMS_HELAVGFACTOR(2)
      DATA (BEAMS_HELAVGFACTOR(I),I=1,2)/%(beamone_helavgfactor)d,%(beamtwo_helavgfactor)d/
      INTEGER FLAVOR(NEXTERNAL)
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

C
C GLOBAL VARIABLES
C
      INTEGER USERHEL
      COMMON/%(proc_prefix)sHELUSERCHOICE/USERHEL
      DATA USERHEL/-1/
      LOGICAL HELRESET
      common/%(proc_prefix)shelreset/HELRESET
      data HELRESET/.true./

%(helicity_lines)s
%(den_factor_line)s

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
C     FLAV_IDX=0 (or out of range) means GET_FLAVOR_INDEX could not resolve
C     the requested flavor: it is not an allowed combination, so its matrix
C     element is identically zero. Short-circuit before touching the
C     1..NFLAV GOODHEL/NTRY arrays.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS = 0D0
        RETURN
      ENDIF
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
      IF(USERHEL.EQ.-1) NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
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
          DO J=1,NFLAV
            GOODHEL(IHEL,J)=.TRUE.
          ENDDO
        ENDDO
      ENDIF
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (USERHEL.EQ.-1.OR.USERHEL.EQ.IHEL) THEN
                IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 20.OR.USERHEL.NE.-1) THEN
                    IF(NTRY(FLAV_IDX).GE.2.AND.POLARIZATIONS(0,0).ne.-1.and.(.not.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL))) THEN
                        CYCLE
                    ENDIF
                    T=%(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_IDX)
                    IF(POLARIZATIONS(0,0).eq.-1.or.%(proc_prefix)sIS_BORN_HEL_SELECTED(IHEL)) THEN
                        ANS=ANS+T
                    ENDIF
                    IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
                        GOODHEL(IHEL,FLAV_IDX)=.TRUE.
                    ENDIF
                ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)*%(proc_prefix)sBROKEN_SYM(FLAVOR)
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


      REAL*8 FUNCTION %(proc_prefix)sMATRIX(P,NHEL,IC,FLAV_IDX)
      use model_object
C
%(info_lines)s
C
C Returns amplitude squared -- no average over initial state/symmetry factor
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
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL), IC(NEXTERNAL)
      INTEGER FLAV_IDX
C
C LOCAL VARIABLES
C
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(%(ncolortriang)d)
      INTEGER %(proc_prefix)sDENOM
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C
C GLOBAL VARIABLES
C
      include 'coupl.inc'
%(global_variable)s
C COLOR DATA
%(color_data_lines)s
C
C
C ----------
C BEGIN CODE
C ----------
      call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
c      WRITE (*,*) '  -> AMP = ', AMP
      call %(proc_prefix)sGET_JAMP(AMP,JAMP)
c      WRITE (*,*) '  -> JAMP = ', JAMP
      call %(proc_prefix)sGET_MATRIX(JAMP,%(proc_prefix)sMATRIX)
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
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)

%(jamp_lines)s
      END

      SUBROUTINE %(proc_prefix)sGET_MATRIX(JAMP,MATRIX)
C
%(process_lines)s
C
      IMPLICIT NONE
C
C CONSTANTS
C
CF2PY INTENT(OUT) :: MATRIX
CF2PY INTENT(IN) :: JAMP


      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      REAL*8     ZERO,MATRIX
      PARAMETER (ZERO=0D0)
C

C LOCAL VARIABLES
C
      INTEGER I,J
      COMPLEX*16 ZTEMP

      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 JAMP(NCOLOR), TMP_JAMP(%(nb_temp_jamp)i)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
C

C COLOR DATA
C

      MATRIX = 0.D0
      CF_INDEX = 0
      DO I = 1, NCOLOR
        ZTEMP = (0.D0,0.D0)
        DO J = I, NCOLOR
          CF_INDEX = CF_INDEX + 1
          ZTEMP = ZTEMP + %(proc_prefix)sCF(CF_INDEX)*JAMP(J)
        ENDDO
        MATRIX = MATRIX+ZTEMP*DCONJG(JAMP(I))/%(proc_prefix)sDENOM
      ENDDO
      END



      SUBROUTINE %(proc_prefix)sGET_INTER(JAMP_1,JAMP_2, INTER)

CF2PY INTENT(OUT) :: INTER
CF2PY INTENT(IN) :: JAMP_1
CF2PY INTENT(IN) :: JAMP_2

      INTEGER I,J
      INTEGER NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER CF_INDEX
      INTEGER %(proc_prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(proc_prefix)sDENOM, IDEN
      %(den_factor_line)s
      common /%(proc_prefix)scolor_matrix/ %(proc_prefix)sCF,%(proc_prefix)sDENOM
      COMPLEX*16 JAMP_1(NCOLOR),JAMP_2(NCOLOR),INTER

C COLOR DATA
C

      INTER = (0.D0,0.D0)
      CF_INDEX = 0
      DO I = 1, NCOLOR
C        ZTEMP = DCONJG(JAMP_2(I))
         DO J=I, NCOLOR
          CF_INDEX = CF_INDEX +1
          INTER = INTER + %(proc_prefix)sCF(CF_INDEX) * (JAMP_1(J) * DCONJG(JAMP_2(I)) +JAMP_1(I) * DCONJG(JAMP_2(J)))
        ENDDO
      ENDDO
      INTER = INTER/ (2d0*%(proc_prefix)sDENOM*IDEN)

      END



      SUBROUTINE  %(proc_prefix)sGET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, SCALE2, INTER)
c     P momenta
c     NHEL base of helicity that are not changing
c     POS(N_CHNGING): position of the changing helicity
c     n_changing: number of changing helicity
c     ALLOW_HEL(NCOMB, N_CHANGING): combination of helicity to consider (all jamp computed)
c     INTER(NCOMB*(NCOMB+1)/2): all interference term (not the symmetric one)
      use model_object
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: SCALE2
CF2PY INTENT(OUT) :: INTER(N_COMB*(N_COMB+1)/2)
C  SCALE2 is a dummy argument added to have the same syntax as in loop-induced
c
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER THISNHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE PRECISION ALPHAS, SCALE2
      DOUBLE COMPLEX INTER(*)
      INTEGER NINTER
      INTEGER NB_NHEL
      DOUBLE COMPLEX, ALLOCATABLE :: TMP_INTER(:)
      PARAMETER (NB_NHEL=%(ncomb)d)
c LOCAL
      INTEGER I,IHEL,IPART
      DOUBLE PRECISION PI
C
      INTEGER NHEL(NEXTERNAL,NB_NHEL)
C     put in common block to expose this variable to python interface
      COMMON/%(proc_prefix)sPROCESS_NHEL/NHEL
C
C     include coupling definition to update the value of alphas
C
      include 'coupl.inc'

      ninter = N_COMB*(N_COMB+1)/2
      allocate(TMP_INTER(NINTER))
      TMP_INTER(:) = (0d0, 0d0)

      DO I=1, N_COMB*(N_COMB+1)/2
        INTER(I) = 0
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
        TMP_INTER(:) = 0
        call  %(proc_prefix)sGET_ALL_INTER(P, THISNHEL, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, TMP_INTER)
        do I = 1, N_COMB*(N_COMB+1)/2
           INTER(I) = INTER(I) + TMP_INTER(I)
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
c     INTER((NCOMB*NCOMB+1)/2: all interference term (not the symmetric one)
      implicit none
CF2PY INTENT(IN) :: P(0:3,%(nexternal)d)
CF2PY INTENT(IN) :: NHEL(%(nexternal)d)
CF2PY INTENT(IN) :: POS(N_CHANGING)
CF2PY INTENT(IN) :: N_CHANGING
CF2PY INTENT(IN) :: ALLOW_HEL(N_CHANGING*N_COMB)
CF2PY INTENT(IN) :: N_COMB
CF2PY INTENT(IN) :: FLAVOR(%(nexternal)d)
CF2PY INTENT(OUT) :: INTER(NCOMB*(NCOMB+1)/2)
c
C
C ARGUMENTS
C
      INTEGER    NEXTERNAL
      PARAMETER (NEXTERNAL=%(nexternal)d)
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER N_CHANGING, N_COMB
      INTEGER POS(*)
      INTEGER ALLOW_HEL(*)
      INTEGER FLAVOR(NEXTERNAL)
      DOUBLE COMPLEX INTER(*)
c
c     Intermediate array
c
      INTEGER    NGRAPHS
      PARAMETER (NGRAPHS=%(ngraphs)d)
      INTEGER    NCOLOR
      PARAMETER (NCOLOR=%(ncolor)d)
      INTEGER IC(NEXTERNAL)

      DOUBLE COMPLEX AMP(NGRAPHS)
      DOUBLE COMPLEX, ALLOCATABLE, SAVE :: JAMP(:,:)
      INTEGER, SAVE :: S_NCOMB = 0

c
c     LOCAL
c
      INTEGER I,J,SOL,N
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX

      if (allocated(jamp) .and. S_NCOMB.ne.N_COMB) then
           deallocate(jamp)
      endif

     if (.not.allocated(jamp)) then
        S_NCOMB=N_COMB
        allocate(jamp(ncolor, N_COMB))
      endif
C ----------
C BEGIN CODE
C ----------
      IC(:)=1
      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
C     Unresolved flavor (not an allowed combination): the matrix element and
C     therefore all interference terms are zero.
      IF (FLAV_IDX.EQ.0) THEN
         DO I = 1, N_COMB*(N_COMB+1)/2
            INTER(I) = (0d0, 0d0)
         ENDDO
         RETURN
      ENDIF
      do I = 1, N_COMB
         do N = 1, N_CHANGING
            NHEL(POS(N)) = ALLOW_HEL((I-1)*N_CHANGING+N)
         enddo
         call %(proc_prefix)sGET_AMP(P,NHEL,IC,FLAV_IDX,AMP)
         call %(proc_prefix)sGET_JAMP(AMP,JAMP(1,I))
      enddo

    SOL = 0
    DO I = 1, N_COMB
       DO J= I, N_COMB
         SOL = SOL +1
         call %(proc_prefix)sGET_INTER(JAMP(1,I), JAMP(1,J), INTER(SOL))
       ENDDO
    ENDDO


    RETURN
    END




      SUBROUTINE %(proc_prefix)sGET_value(P, ALPHAS, NHEL, FLAVOR ,ANS)
C     f2py interface accepting the full FLAVOR(NEXTERNAL) array (back-compat):
C     resolve it to FLAV_IDX and forward to GET_value_internal. Use
C     GET_value_idx below to pass the flavor index directly.
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
C     ROUTINE FOR F2PY to read the benchmark point.

      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
      CALL  %(proc_prefix)sGET_value_internal(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
      RETURN
      END


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
      INTEGER FLAV_IDX
CF2PY INTENT(OUT) :: ANS
CF2PY INTENT(IN) :: NHEL
CF2PY INTENT(IN) :: P(0:3,NEXTERNAL)
CF2PY INTENT(IN) :: ALPHAS
CF2PY INTENT(IN) :: FLAV_IDX

      CALL  %(proc_prefix)sGET_value_internal(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
      RETURN
      END


      SUBROUTINE %(proc_prefix)sGET_value_internal(P, ALPHAS, NHEL, FLAV_IDX ,ANS)
      use model_object
C     This routine is the real value but can not be in the interface
C     due to f2py not knowing how to handle the couplings common block
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
C     ROUTINE FOR F2PY to read the benchmark point.
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
