      SUBROUTINE %(func_name)s(FLAV_IDX, FLAVOR)
C     Reverse of GET_FLAVOR_INDEX: fill FLAVOR(NEXTERNAL) with the per-leg
C     group-position vector of the FLAV_IDX-th allowed flavor (same table /
C     ordering). FLAV_IDX is expected in [1,NFLAV] (GET_FLAVOR_INDEX never
C     returns 0); the bounds guard below is purely defensive and maps any
C     out-of-range value to the first flavor. Used by the outer entry points
C     (SMATRIX, ...) which receive FLAV_IDX but still need the FLAVOR array
C     (e.g. for BROKEN_SYM).
%(nexternal_decl)s
      INTEGER NFLAV
      PARAMETER (NFLAV=%(nflav)d)
      INTEGER FLAV_IDX
      INTEGER FLAVOR(NEXTERNAL)
CF2PY INTENT(IN) :: FLAV_IDX
CF2PY INTENT(OUT) :: FLAVOR(NEXTERNAL)
      INTEGER FA_I, FA_USE
      INTEGER FA_TABLE(NEXTERNAL, NFLAV)
      DATA FA_TABLE /%(flav_table_data)s/
      FA_USE = FLAV_IDX
      IF (FA_USE .LT. 1 .OR. FA_USE .GT. NFLAV) FA_USE = 1
      DO FA_I = 1, NEXTERNAL
        FLAVOR(FA_I) = FA_TABLE(FA_I, FA_USE)
      ENDDO
      RETURN
      END
