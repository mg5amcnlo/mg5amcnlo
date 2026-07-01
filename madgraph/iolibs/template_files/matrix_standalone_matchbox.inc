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
      INTEGER                 NCOMB
      PARAMETER (             NCOMB=%(ncomb)d)
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL),ANS
C     FLAV_IDX (the dummy argument) is declared with NFLAV below.
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER NHEL(NEXTERNAL,NCOMB)
      REAL*8 T
      REAL*8 %(proc_prefix)sMATRIX
      INTEGER IHEL,IDEN, I
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
C FUNCTIONS
C
      INTEGER %(proc_prefix)sBROKEN_SYM
%(helicity_lines)s
%(den_factor_line)s
C ----------
C BEGIN CODE
C ----------
C     FLAV_IDX=0 (or out of range): GET_FLAVOR_INDEX could not resolve the
C     requested flavor -- not an allowed combination, matrix element is zero.
      IF (FLAV_IDX.LT.1 .OR. FLAV_IDX.GT.NFLAV) THEN
        ANS = 0D0
        RETURN
      ENDIF
      CALL %(proc_prefix)sGET_FLAVOR(FLAV_IDX, FLAVOR)
      NTRY(FLAV_IDX)=NTRY(FLAV_IDX)+1
      ANS = 0D0
          DO IHEL=1,NCOMB
             IF (GOODHEL(IHEL,FLAV_IDX) .OR. NTRY(FLAV_IDX) .LT. 20) THEN
                 T=%(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),FLAV_IDX)
               ANS=ANS+T
               IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN
                   GOODHEL(IHEL,FLAV_IDX)=.TRUE.
               ENDIF
             ENDIF
          ENDDO
      ANS=ANS/DBLE(IDEN)*%(proc_prefix)sBROKEN_SYM(FLAVOR)
      END


      REAL*8 FUNCTION %(proc_prefix)sMATRIX(P,NHEL,FLAV_IDX)
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
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER FLAV_IDX
C
C LOCAL VARIABLES
C
      INTEGER FLAVOR(NEXTERNAL)
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER  CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER DENOM, CF_INDEX
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), LNJAMP(NCOLOR)
      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
      type(aloha) W(NWAVEFUNCS)
      COMPLEX*16 DUM0,DUM1
      DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
%(flavor_mask_decl)s
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
%(flavor_mask_setup)s
%(helas_calls)s
%(jamp_lines)s

      %(proc_prefix)sMATRIX = 0.D0
      CF_INDEX =0
      DO I = 1, NCOLOR
          ZTEMP = (0.D0,0.D0)
          DO J = I, NCOLOR
              CF_INDEX = CF_INDEX + 1
              ZTEMP = ZTEMP + CF(CF_INDEX)*JAMP(J)
          ENDDO
          %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX+ZTEMP*DCONJG(JAMP(I))
      ENDDO
      %(proc_prefix)sMATRIX = %(proc_prefix)sMATRIX/DENOM
%(amp2_lines)s
      END


       SUBROUTINE %(proc_prefix)sBORN(P,NHEL,FLAVOR)
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
      REAL*8     ZERO
      PARAMETER (ZERO=0D0)
      COMPLEX*16 IMAG1
      PARAMETER (IMAG1=(0D0,1D0))
C
C ARGUMENTS
C
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER NHEL(NEXTERNAL)
      INTEGER FLAVOR(NEXTERNAL)
C
C LOCAL VARIABLES
C
      INTEGER FLAV_IDX
      INTEGER %(proc_prefix)sGET_FLAVOR_INDEX
      INTEGER I,J
      COMPLEX*16 ZTEMP
      INTEGER CF(NCOLOR*(NCOLOR+1)/2)
      INTEGER CF_INDEX, DENOM
      COMPLEX*16 AMP(NGRAPHS), JAMP(NCOLOR), LNJAMP(NCOLOR)
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP

      COMPLEX*16 TMP_JAMP(%(nb_temp_jamp)i)
       type(aloha) W(NWAVEFUNCS)
       COMPLEX*16 DUM0,DUM1
       DATA DUM0, DUM1/(0d0, 0d0), (1d0, 0d0)/
%(flavor_mask_decl)s
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
      FLAV_IDX = %(proc_prefix)sGET_FLAVOR_INDEX(FLAVOR)
C     Unresolved flavor (not an allowed combination): all color amplitudes
C     are zero.
      IF (FLAV_IDX.EQ.0) THEN
        DO I = 1, NCOLOR
          JAMP(I) = (0D0, 0D0)
          LNJAMP(I) = (0D0, 0D0)
        ENDDO
        RETURN
      ENDIF
%(flavor_mask_setup)s
%(helas_calls)s
%(jamp_lines)s

      END

      SUBROUTINE %(proc_prefix)sGET_JAMP(njamp, ONEJAMP)

      INTEGER     NCOLOR, NJAMP
      PARAMETER (NCOLOR=%(ncolor)d)
      COMPLEX*16  JAMP(NCOLOR), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP

	ONEJAMP = JAMP(njamp+1) ! +1 since njamp start at zero (c convention)
	END

       SUBROUTINE %(proc_prefix)sGET_LNJAMP(njamp, ONEJAMP)

      INTEGER     NCOLOR, NJAMP
      PARAMETER (NCOLOR=%(ncolor)d)
      COMPLEX*16  JAMP(NCOLOR), LNJAMP(NCOLOR), ONEJAMP
      COMMON/%(proc_prefix)sJAMP/JAMP,LNJAMP

	ONEJAMP = LNJAMP(njamp+1) ! +1 since njamp start at zero (c convention)
	END




     SUBROUTINE %(proc_prefix)sGET_NCOLOR(in1, in2, out)

	integer in1, in2, out
	%(color_information)s
     END

     SUBROUTINE %(proc_prefix)sGET_NCOL(ncol)
	integer ncol
	ncol = %(ncolor)d
	return
	end

%(broken_sym_function)s


%(flavor_index_function)s


%(flavor_array_function)s
