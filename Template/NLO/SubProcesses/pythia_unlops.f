
      subroutine pythia_UNLOPS(p,realemission,passUNLOPScuts)
      implicit none
      include "nexternal.inc"
      include "genps.inc"
      include "run.inc"
c arguments
      double precision p(0:3,nexternal)
      logical realemission,passUNLOPScuts
C VARIABLES TO SPECIFY JETS
      DOUBLE PRECISION PJET(NEXTERNAL,0:3)
      DOUBLE PRECISION PINC(NINCOMING,0:3)
      DOUBLE PRECISION PRADTEMP(0:3)
      DOUBLE PRECISION PRECTEMP(0:3)
      DOUBLE PRECISION PEMTTEMP(0:3)
      INTEGER JETFLAVOUR(NEXTERNAL)
      INTEGER INCFLAVOUR(NINCOMING)
      DOUBLE PRECISION PTMIN, PTMINSAVE, TEMP
      DOUBLE PRECISION PT1, PT2
      INTEGER I, J, K, L, NJETS, NJETS_ABOVE_CUT
C     PYTHIA EVOLUTION PT DEFINITION
      double precision rhopythia
      external rhopythia
c need the particle ID's
      LOGICAL  IS_A_J(NEXTERNAL),IS_A_LP(NEXTERNAL),IS_A_LM(NEXTERNAL)
      LOGICAL  IS_A_PH(NEXTERNAL)
      COMMON /TO_SPECISA/IS_A_J,IS_A_LP,IS_A_LM,IS_A_PH
      integer maxflow
      parameter (maxflow=999)
      integer idup(nexternal,maxproc),mothup(2,nexternal,maxproc),
     &     icolup(2,nexternal,maxflow)
      common /c_leshouche_inc/idup,mothup,icolup
c SCALE (should be moved to the run_card.dat).
      double precision pt_pythia
      include 'cuts.inc'

      pt_pythia=ptj

c Set passcut to true, if not it will be updated below.
      passUNLOPScuts=.true.

C     Reset jet momenta
      NJETS=0
      DO I=1,NEXTERNAL
        JETFLAVOUR(I) = 0
        DO J=0,3 
          PJET(I,J) = 0E0
        ENDDO
      ENDDO

C     Fill incoming particle momenta
c     If a QCD particle, set INCFLAVOR=1
      DO I=1,NINCOMING
         if (LPP(1).NE.0) then
            INCFLAVOUR(I) = 1
         else
            INCFLAVOUR(I) = idup(i,1)
         endif
         DO J=0,3
            PINC(I,J) = P(J,I)
         ENDDO
      ENDDO

C     Fill final jet momenta
c     If a QCD particle, set JETFLAVOR=1
      DO I=NINCOMING+1,NEXTERNAL
         IF(IS_A_J(I)) THEN
            NJETS=NJETS+1
            JETFLAVOUR(NJETS) = 1
            DO J=0,3
               PJET(NJETS,J) = P(J,I)
            ENDDO
         ENDIF
      ENDDO

C     Number of "jets" above the cut.
      NJETS_ABOVE_CUT=0

C     PYTHIA PT SEPARATION CUT
      IF(NJETS.LE.0) then
         write (*,*) 'Error in pythia_UNLOPS: there should always'/
     $        /' be one QCD parton in the final state, because this'/
     $        /' should only be called for the real-emission.',njets
         stop
      endif
      PTMINSAVE = EBEAM(1) + EBEAM(2)
      DO I=1,NJETS
         PTMIN = EBEAM(1) + EBEAM(2)
         PTMINSAVE = MIN(PTMIN,PTMINSAVE)
c-------------------------------------------------------------
C     Compute pythia ISR separation between i-jet and incoming
c-------------------------------------------------------------
         IF ( (LPP(1).NE.0) .OR. (LPP(2).NE.0)) THEN
C     Check separation to first incoming particle
            DO L=0,3
               PRADTEMP(L) = PINC(1,L)
               PEMTTEMP(L) = PJET(I,L)
               PRECTEMP(L) = PINC(2,L)
            ENDDO
            PT1 = RHOPYTHIA(PRADTEMP, PEMTTEMP, PRECTEMP,INCFLAVOUR(1),
     $           -1)
            PTMIN = MIN( PTMIN, PT1 )
C     Check separation to second incoming particle
            DO L=0,3
               PRADTEMP(L) = PINC(2,L)
               PEMTTEMP(L) = PJET(I,L)
               PRECTEMP(L) = PINC(1,L)
            ENDDO
            PT2 = RHOPYTHIA(PRADTEMP, PEMTTEMP, PRECTEMP, INCFLAVOUR(2),
     $           -1)
            PTMIN = MIN( PTMIN, PT2 )
         ENDIF
c-------------------------------------------------------------
C     Compute pythia FSR separation between two jets, without any
C     knowledge of colour connections
c-------------------------------------------------------------
         DO J=1,NJETS
            DO K=1,NJETS
               IF ( I .NE. J .AND. I .NE. K .AND. J .NE. K ) THEN
C     Check separation between final partons i and j, with k as spectator
                  DO L=0,3
                     PRADTEMP(L) = PJET(J,L)
                     PEMTTEMP(L) = PJET(I,L)
                     PRECTEMP(L) = PJET(K,L)
                  ENDDO
                  TEMP = RHOPYTHIA( PRADTEMP, PEMTTEMP, PRECTEMP,
     $                 JETFLAVOUR(J), 1)
                  PTMIN = MIN(PTMIN, TEMP);
               ENDIF
            ENDDO               ! LOOP OVER NJET
         ENDDO                  ! LOOP OVER NJET
c-------------------------------------------------------------
C     Compute pythia FSR separation between two jets, with initial
C     spectator
c-------------------------------------------------------------
         IF ( (LPP(1).NE.0) .OR. (LPP(2).NE.0)) THEN
            DO J=1,NJETS
C     Allow both initial partons as recoiler
               IF ( I .NE. J ) THEN
C     Check with first initial as recoiler
                  DO L=0,3
                     PRADTEMP(L) = PJET(J,L)
                     PEMTTEMP(L) = PJET(I,L)
                     PRECTEMP(L) = PINC(1,L)
                  ENDDO
                  TEMP = RHOPYTHIA( PRADTEMP, PEMTTEMP, PRECTEMP,
     $                 JETFLAVOUR(J), 1);
                  PTMIN = MIN(PTMIN, TEMP);
                  DO L=0,3
                     PRADTEMP(L) = PJET(J,L)
                     PEMTTEMP(L) = PJET(I,L)
                     PRECTEMP(L) = PINC(2,L)
                  ENDDO
                  TEMP = RHOPYTHIA( PRADTEMP, PEMTTEMP, PRECTEMP,
     $                 JETFLAVOUR(J), 1);
                  PTMIN = MIN(PTMIN, TEMP);
               ENDIF
            ENDDO               ! LOOP OVER NJET
         ENDIF
c-------------------------------------------------------------
C     IF ALL SEPARATIONS OF THIS PARTON PASS THE CUT, COUNT AS RESOLVED
C     JET.
c-------------------------------------------------------------
         IF( PTMIN .GT. PT_PYTHIA) THEN
            NJETS_ABOVE_CUT = NJETS_ABOVE_CUT + 1
         ENDIF
         PTMINSAVE = MIN(PTMIN,PTMINSAVE)
      ENDDO                     ! LOOP OVER NJET

C     CHECK COMPATIBILITY WITH CUT, REAL-EMISSION VERSION
      IF( realemission .and.
     &     ((NJETS .GT. 0) .AND. (NJETS_ABOVE_CUT .EQ. NJETS)) ) THEN
         passUNLOPScuts = .FALSE.
         RETURN
      elseif ((.not.realemission) .and. 
     &        (NJETS_ABOVE_CUT .LT. NJETS-1) ) THEN
         passUNLOPScuts = .FALSE.
         RETURN
      ENDIF

      return
      end
     

      DOUBLE PRECISION FUNCTION RHOPYTHIA(PRAD, PEMT, PREC, FLAVRAD,
     $     EMTYPE)
c************************************************************************
c     Returns pythia pT between two particles with prad and pemt
c     with prec as spectator
c************************************************************************
      IMPLICIT NONE
c
c     Arguments
c
      DOUBLE PRECISION PRAD(0:3),PEMT(0:3), PREC(0:3)
      INTEGER FLAVRAD, EMTYPE
c
c     Local
c
      DOUBLE PRECISION Q(0:3),SUM(0:3), qBR(0:3), qAR(0:3)
      DOUBLE PRECISION Q2, m2Rad, m2Dip, m0Rad, qBR2, qAR2, x1, x3, z
      DOUBLE PRECISION TEMP
      INTEGER emtsign

c-----
c  Begin Code
c-----

C     Get sign of emitted momentum
      emtsign = 1
      if(emtype .eq. -1) emtsign = -1

C     Get virtuality
      Q(0) = pRad(0) + emtsign*pEmt(0)
      Q(1) = pRad(1) + emtsign*pEmt(1)
      Q(2) = pRad(2) + emtsign*pEmt(2)
      Q(3) = pRad(3) + emtsign*pEmt(3)
      Q2   = emtsign * ( Q(0)**2 - Q(1)**2 - Q(2)**2 - Q(3)**2 );
C     Mass term of radiator. Ideally, get from FLAVRAD input.
      m0Rad = 0d0
      m2Rad = 0d0
      if( abs(flavRad) .ge. 4 .and. abs(flavRad) .lt. 7)
     &     m2Rad = m0Rad**2

C     Construct 2->3 variables for FSR
      sum(0) = pRad(0) + pRec(0) + pEmt(0)
      sum(1) = pRad(1) + pRec(1) + pEmt(1)
      sum(2) = pRad(2) + pRec(2) + pEmt(2)
      sum(3) = pRad(3) + pRec(3) + pEmt(3)
      m2Dip  = sum(0)**2 - sum(1)**2 - sum(2)**2 - sum(3)**2
      x1     = 2. * ( sum(0)*pRad(0) - sum(1)*pRad(1)
     &             -  sum(2)*pRad(2) - sum(3)*pRad(3) ) / m2Dip
      x3     = 2. * ( sum(0)*pEmt(0) - sum(1)*pEmt(1)
     &             -  sum(2)*pEmt(2) - sum(3)*pEmt(3) ) / m2Dip

C     Construct momenta of dipole before/after splitting for ISR
      qBR(0) = pRad(0) + pRec(0) - pEmt(0)
      qBR(1) = pRad(1) + pRec(1) - pEmt(1)
      qBR(2) = pRad(2) + pRec(2) - pEmt(2)
      qBR(3) = pRad(3) + pRec(3) - pEmt(3)
      qBR2   = qBR(0)**2 - qBR(1)**2 - qBR(2)**2 - qBR(3)**2

      qAR(0) = pRad(0) + pRec(0)
      qAR(1) = pRad(1) + pRec(1)
      qAR(2) = pRad(2) + pRec(2)
      qAR(3) = pRad(3) + pRec(3)
      qAR2   = qAR(0)**2 - qAR(1)**2 - qAR(2)**2 - qAR(3)**2

C     Calculate z of splitting, different for FSR and ISR
      z = x1/(x1+x3)
      if(emtype .eq. -1 ) z = qBR2 / qAR2;

C     Separation of splitting, different for FSR and ISR
      temp = z*(1.-z)
      if(emtype .eq. -1 ) temp = (1.-z)

C     pT^2 = separation*virtuality
      temp = temp*(Q2 - emtsign*m2Rad)
      if(temp .lt. 0.) temp = 0.

C     Return pT
      rhoPythia = dsqrt(temp);

      RETURN
      END
