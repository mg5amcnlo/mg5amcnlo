c Copyright (C) 2020-2026 CERN and UCLouvain.
c Licensed under the GNU Lesser General Public License (version 3 or later).
c Created originally by: O. Mattelaer (Oct 2023) for the MG5aMC CUDACPP plugin.
c Further modified by: O. Mattelaer, A. Valassi (2023-2024).
c Integrated with the MadGraph7 project in Feb 2026.
c       ======================================================
c       *START* Included from CUDACPP template smatrix_multi.f
c       (into function smatrix$i_multi in auto_dsig$i.f)
c       ======================================================
        CALL COUNTERS_SMATRIX1MULTI_STOP( -1 )  ! fortranMEs=-1
#ifdef MG5AMC_MEEXPORTER_CUDACPP
      ENDIF

      IF( FBRIDGE_MODE .EQ. 1 .OR. FBRIDGE_MODE .LT. 0 ) THEN  ! (CppOnly=1 or BothQuiet=-1 or BothDebug=-2)
        IF( LIMHEL.NE.0 ) THEN
          WRITE(6,*) 'ERROR! The cudacpp bridge only supports LIMHEL=0'
          STOP
        ENDIF
        IF ( FIRST ) THEN  ! exclude first pass (helicity filtering) from timers (#461)
          CALL COUNTERS_SMATRIX1MULTI_START( 1, VECSIZE_USED )  ! cudacppHEL=1
          CALL FBRIDGESEQUENCE_NOMULTICHANNEL( FBRIDGE_PBRIDGE,  ! multi channel disabled for helicity filtering
     &      P_MULTI, ALL_G, IFLAV_VEC, HEL_RAND, COL_RAND,
     &      OUT2, SELECTED_HEL2, SELECTED_COL2, .TRUE.)  ! quit after computing helicities
          FIRST = .FALSE.
c         ! This is a workaround for https://github.com/oliviermattelaer/mg5amc_test/issues/22 (see PR #486)
c          IF( FBRIDGE_MODE .EQ. 1 ) THEN  ! (CppOnly=1 : SMATRIX1 is not called at all)
c            CALL RESET_CUMULATIVE_VARIABLE()  ! mimic 'avoid bias of the initialization' within SMATRIX1
c          ENDIF
          CALL FBRIDGEGETNGOODHEL(FBRIDGE_PBRIDGE,NGOODHEL,NTOTHEL)
          IF( NTOTHEL .NE. NCOMB ) THEN
            WRITE(6,*) 'ERROR! Cudacpp/Fortran mismatch',
     &        ' in total number of helicities', NTOTHEL, NCOMB
            STOP
          ENDIF
          WRITE (6,*) 'NGOODHEL =', NGOODHEL
          WRITE (6,*) 'NCOMB =', NCOMB
          CALL COUNTERS_SMATRIX1MULTI_STOP( 1 )  ! cudacppHEL=1
        ENDIF
        CALL COUNTERS_SMATRIX1MULTI_START( 0, VECSIZE_USED )  ! cudacppMEs=0
        IF ( .NOT. MULTI_CHANNEL ) THEN
          CALL FBRIDGESEQUENCE_NOMULTICHANNEL( FBRIDGE_PBRIDGE,  ! multi channel disabled
     &      P_MULTI, ALL_G, IFLAV_VEC, HEL_RAND, COL_RAND,
     &      OUT2, SELECTED_HEL2, SELECTED_COL2, .FALSE.)  ! do not quit after computing helicities
        ELSE
          IF( SDE_STRAT.NE.1 ) THEN
            WRITE(6,*) 'ERROR  ! The cudacpp bridge requires SDE=1'  ! multi channel single-diagram enhancement strategy
            STOP
          ENDIF
          CALL FBRIDGESEQUENCE(FBRIDGE_PBRIDGE, P_MULTI, ALL_G,  ! multi channel enabled
     &      IFLAV_VEC, HEL_RAND, COL_RAND, CHANNELS, OUT2,
     &      SELECTED_HEL2, SELECTED_COL2, .FALSE.)  ! do not quit after computing helicities
        ENDIF
        CALL COUNTERS_SMATRIX1MULTI_STOP( 0 )  ! cudacppMEs=0
      ENDIF

      IF( FBRIDGE_MODE .LT. 0 ) THEN  ! (BothQuiet=-1 or BothDebug=-2)
        DO IVEC=1, VECSIZE_USED
          CBYF1 = OUT2(IVEC)/OUT(IVEC) - 1
          FBRIDGE_NCBYF1 = FBRIDGE_NCBYF1 + 1
          FBRIDGE_CBYF1SUM = FBRIDGE_CBYF1SUM + CBYF1
          FBRIDGE_CBYF1SUM2 = FBRIDGE_CBYF1SUM2 + CBYF1 * CBYF1
          IF( CBYF1 .GT. FBRIDGE_CBYF1MAX ) FBRIDGE_CBYF1MAX = CBYF1
          IF( CBYF1 .LT. FBRIDGE_CBYF1MIN ) FBRIDGE_CBYF1MIN = CBYF1
          IF( FBRIDGE_MODE .EQ. -2 ) THEN  ! (BothDebug=-2)
            WRITE (*,'(I4,2E16.8,F23.11,I3,I3,I4,I4)')
     &        IVEC, OUT(IVEC), OUT2(IVEC), 1+CBYF1,
     &        SELECTED_HEL(IVEC), SELECTED_HEL2(IVEC),
     &        SELECTED_COL(IVEC), SELECTED_COL2(IVEC)
          ENDIF
          IF( ABS(CBYF1).GT.5E-5 .AND. NWARNINGS.LT.20 ) THEN
            NWARNINGS = NWARNINGS + 1
            WRITE (*,'(A,I4,A,I4,2E16.8,F23.11)')
     &        'WARNING! (', NWARNINGS, '/20) Deviation more than 5E-5',
     &        IVEC, OUT(IVEC), OUT2(IVEC), 1+CBYF1
          ENDIF
        END DO
      ENDIF

      IF( FBRIDGE_MODE .EQ. 1 .OR. FBRIDGE_MODE .LT. 0 ) THEN  ! (CppOnly=1 or BothQuiet=-1 or BothDebug=-2)
        DO IVEC=1, VECSIZE_USED
          OUT(IVEC) = OUT2(IVEC)  ! use the cudacpp ME instead of the fortran ME!
          SELECTED_HEL(IVEC) = SELECTED_HEL2(IVEC)  ! use the cudacpp helicity instead of the fortran helicity!
          SELECTED_COL(IVEC) = SELECTED_COL2(IVEC)  ! use the cudacpp color instead of the fortran color!
        END DO
      ENDIF
#endif

      IF ( FIRST_CHID ) THEN
        IF ( MULTI_CHANNEL ) THEN
          WRITE (*,*) 'MULTI_CHANNEL = TRUE'
        ELSE
          WRITE (*,*) 'MULTI_CHANNEL = FALSE'
        ENDIF
        WRITE (*,*) 'CHANNEL_ID =', CHANNELS(1)
        FIRST_CHID = .FALSE.
      ENDIF
c       ======================================================
c       **END** Included from CUDACPP template smatrix_multi.f
c       ======================================================

