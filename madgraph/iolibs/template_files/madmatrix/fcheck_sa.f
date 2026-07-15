C Copyright (C) 2020-2026 CERN and UCLouvain.
C Licensed under the GNU Lesser General Public License (version 3 or later).
C Created originally by: A. Valassi (Feb 2022) for the MG5aMC CUDACPP plugin.
C Further modified by: D. Massaro, A. Thete, A. Valassi (2022-2026).
C Integrated with the MadGraph7 project in Feb 2026.

      PROGRAM FCHECK_SA
      IMPLICIT NONE
      INCLUDE 'fsampler.inc'
      INCLUDE 'fbridge.inc'
      INTEGER*8 SAMPLER, BRIDGE ! 64bit memory addresses
      INTEGER NEVTMAX, NEXTERNAL, NP4
      PARAMETER(NEVTMAX=2048*256, NEXTERNAL=%(nexternal)d, NP4=4)
      CHARACTER*32 ARG0, ARG1, ARG2, ARG3
      INTEGER NARG1, NARG2, NARG3
      INTEGER NEVT, NITER
      INTEGER IEVT, IITER
c     INTEGER IEXTERNAL
      DOUBLE PRECISION MOMENTA(0:NP4-1, NEXTERNAL, NEVTMAX) ! c-array momenta[nevt][nexternal][np4]
      DOUBLE PRECISION GS(NEVTMAX)
      INTEGER IFLAV_VEC(NEVTMAX) ! index of the flavor combination to calculate
      DOUBLE PRECISION RNDHEL(NEVTMAX) ! not yet used
      DOUBLE PRECISION RNDCOL(NEVTMAX) ! not yet used
      DOUBLE PRECISION MES(NEVTMAX)
      INTEGER*4 SELHEL(NEVTMAX) ! not yet used
      INTEGER*4 SELCOL(NEVTMAX) ! not yet used
      DOUBLE PRECISION MES_SUM ! use REAL*16 for quadruple precision
      INTEGER NEVTOK ! exclude nan/abnormal MEs

      IFLAV_VEC(:) = 1
C
C READ COMMAND LINE ARGUMENTS
C (NB: most errors will crash the program !)
C
      IF ( COMMAND_ARGUMENT_COUNT() == 3 ) THEN
        CALL GET_COMMAND_ARGUMENT(1,ARG1)
        CALL GET_COMMAND_ARGUMENT(2,ARG2)
        CALL GET_COMMAND_ARGUMENT(3,ARG3)
        READ (ARG1,'(I4)') NARG1
        READ (ARG2,'(I4)') NARG2
        READ (ARG3,'(I4)') NARG3
        WRITE(6,*) "GPUBLOCKS=  ", NARG1
        WRITE(6,*) "GPUTHREADS= ", NARG2
        WRITE(6,*) "NITERATIONS=", NARG3
        NEVT = NARG1 * NARG2
        NITER = NARG3
        IF ( NEVT > NEVTMAX ) THEN
          WRITE(6,*) "ERROR! NEVT>NEVTMAX"
          STOP
        ENDIF
      ELSE
        CALL GET_COMMAND_ARGUMENT(0,ARG0)
        WRITE(6,*) "Usage: ", TRIM(ARG0),
     &    " gpublocks gputhreads niterations"
        STOP
      ENDIF
C
C USE SAMPLER AND BRIDGE
C
      NEVTOK = 0
      MES_SUM = 0
      CALL FBRIDGECREATE(BRIDGE, NEVT, NEXTERNAL, NP4) ! this must be at the beginning as it initialises the CUDA device
      CALL FSAMPLERCREATE(SAMPLER, NEVT, NEXTERNAL, NP4)
      DO IITER = 1, NITER
        CALL FSAMPLERSEQUENCE(SAMPLER, MOMENTA)
        DO IEVT = 1, NEVT
          GS(IEVT) = 1.2177157847767195 ! fixed G for aS=0.118 (hardcoded for now in check_sa.cc, fcheck_sa.f, runTest.cc)
        END DO
        CALL FBRIDGESEQUENCE_NOMULTICHANNEL(BRIDGE, MOMENTA, GS, ! TEMPORARY? disable multi-channel in fcheck.exe and fgcheck.exe #466
     &    IFLAV_VEC, RNDHEL, RNDCOL, MES, SELHEL, SELCOL, .FALSE.) ! do not quit after computing helicities
        DO IEVT = 1, NEVT
c         DO IEXTERNAL = 1, NEXTERNAL
c           WRITE(6,*) 'MOMENTA', IEVT, IEXTERNAL,
c    &        MOMENTA(0, IEXTERNAL, IEVT),
c    &        MOMENTA(1, IEXTERNAL, IEVT),
c    &        MOMENTA(2, IEXTERNAL, IEVT),
c    &        MOMENTA(3, IEXTERNAL, IEVT)
c         END DO
c         WRITE(6,*) 'MES    ', IEVT, MES(IEVT)
c         WRITE(6,*)
          IF ( .NOT. ISNAN(MES(IEVT)) ) THEN
            NEVTOK = NEVTOK + 1
            MES_SUM = MES_SUM + MES(IEVT)
          ENDIF
        END DO
      END DO
      CALL FSAMPLERDELETE(SAMPLER)
      CALL FBRIDGEDELETE(BRIDGE) ! this must be at the end as it shuts down the CUDA device
      WRITE(6,*) 'Average Matrix Element:', MES_SUM/NEVT/NITER
      WRITE(6,*) 'Abnormal MEs:', NEVT*NITER - NEVTOK
      END PROGRAM FCHECK_SA
