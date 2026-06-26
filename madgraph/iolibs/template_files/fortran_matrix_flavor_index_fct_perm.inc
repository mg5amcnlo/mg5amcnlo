      INTEGER FUNCTION %(func_name)s(FLAVOR)
C     Resolve an external FLAVOR(NEXTERNAL) group-position vector to its
C     1-based index in the allowed-flavor table (same ordering as
C     compute_flavor_masks / the FLAV_TABLE mask columns).
C
C     Permutation-aware variant (decay chains with identical, identically
C     decaying particles, e.g. the two Z systems of z z, z>l+l-, z>l+l-):
C     the mask table keeps only one representative per interchangeable-block
C     permutation class, but a caller (e.g. MadSpin) may present a swapped
C     ordering. FI_TABLE below therefore lists EVERY such permutation and
C     FI_MAP gives the representative's 1-based column each resolves to. A
C     flavor that is not an allowed combination at all returns the 0 sentinel
C     (callers must short-circuit to a zero result; there is no reserved 0).
%(nexternal_decl)s
      INTEGER NLOOKUP
      PARAMETER (NLOOKUP=%(nlookup)d)
      INTEGER FLAVOR(NEXTERNAL)
CF2PY INTENT(IN) :: FLAVOR(NEXTERNAL)
CF2PY INTENT(OUT) :: %(func_name)s
      INTEGER FI_I, FI_J
      LOGICAL FI_MATCH
      INTEGER FI_TABLE(NEXTERNAL, NLOOKUP)
      INTEGER FI_MAP(NLOOKUP)
      DATA FI_TABLE /%(flav_table_data)s/
      DATA FI_MAP /%(flav_index_map)s/
C     0 sentinel for an unresolved (not-in-table) flavor.
      %(func_name)s = 0
      DO FI_I = 1, NLOOKUP
        FI_MATCH = .TRUE.
        DO FI_J = 1, NEXTERNAL
          IF (FLAVOR(FI_J) .NE. FI_TABLE(FI_J, FI_I)) THEN
            FI_MATCH = .FALSE.
            EXIT
          ENDIF
        ENDDO
        IF (FI_MATCH) THEN
          %(func_name)s = FI_MAP(FI_I)
          RETURN
        ENDIF
      ENDDO
      RETURN
      END
