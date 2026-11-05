      INTEGER FUNCTION %(func_name)s(FLAV)
%(nexternal_decl)s
      INTEGER FLAV(NEXTERNAL)
      INTEGER I,J,K,ICOMP
      INTEGER N_TOT, OLD_FACTOR, TOTAL_FACTOR
      INTEGER NCOMP, NENTRIES
      PARAMETER (NCOMP=%(broken_sym_ncomponents)d)
      PARAMETER (NENTRIES=%(broken_sym_nentries)d)
      INTEGER COMP_BEG(NCOMP), COMP_END(NCOMP), COMP_OLD(NCOMP)
      INTEGER PID_LIST(NENTRIES), PID_WORK(NENTRIES)
      INTEGER BLOCK_START(NENTRIES), BLOCK_LEN(NENTRIES)
      LOGICAL SAME_BLOCK
      DATA COMP_BEG /%(broken_sym_component_starts)s/
      DATA COMP_END /%(broken_sym_component_ends)s/
      DATA COMP_OLD /%(broken_sym_component_old_factors)s/
      DATA PID_LIST /%(broken_sym_pid_list)s/
      DATA BLOCK_START /%(broken_sym_block_starts)s/
      DATA BLOCK_LEN /%(broken_sym_block_lengths)s/

      PID_WORK = PID_LIST
      TOTAL_FACTOR = 1
      DO ICOMP=1,NCOMP
         OLD_FACTOR = COMP_OLD(ICOMP)
         IF (COMP_OLD(ICOMP).GT.1) THEN
         DO I=COMP_BEG(ICOMP),COMP_END(ICOMP)
            IF (PID_WORK(I).EQ.0) CYCLE
            N_TOT = 1
            DO J=I+1,COMP_END(ICOMP)
               IF (PID_WORK(I).EQ.PID_WORK(J)) THEN
                  SAME_BLOCK = .TRUE.
                  IF (BLOCK_LEN(I).NE.BLOCK_LEN(J)) SAME_BLOCK = .FALSE.
                  DO K=1,BLOCK_LEN(I)
                     IF (FLAV(BLOCK_START(I)+K-1).NE.FLAV(BLOCK_START(J)+K-1)) THEN
                        SAME_BLOCK = .FALSE.
                     ENDIF
                  ENDDO
                  IF (SAME_BLOCK) THEN
                     PID_WORK(J) = 0
                     N_TOT = N_TOT + 1
                     OLD_FACTOR = OLD_FACTOR/N_TOT
                  ENDIF
               ENDIF
            ENDDO
         ENDDO
         ENDIF
         TOTAL_FACTOR = TOTAL_FACTOR*OLD_FACTOR
      ENDDO
      %(func_name)s = TOTAL_FACTOR
      return
      end
