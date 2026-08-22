      logical function dummy_cuts(p,istatus,ipdg)
C     **************************************************************
C     Make  p p > e+ e- mu+ mu- / a  [QCD]  physically equivalent to
C     p p > z z [QCD]  +  MadSpin, at NLO.
C
C     This is the NLO twin of  ../zz_loopinduced/zz_equivalent_cuts.f
C     and applies the same two cuts, on the same reconstructed lepton
C     pairs.  It is a separate file and not the same one because the
C     two run-card classes give dummy_cuts DIFFERENT signatures:
C
C        LO  / loop induced :  dummy_cuts(P)
C        NLO                :  dummy_cuts(P,ISTATUS,IPDG)
C
C     and the NLO momentum array carries a fifth component (the mass)
C     while the LO one does not.  The NLO signature is the better one
C     to work with: IPDG is handed over per event, so the leptons are
C     located from it directly and nothing has to be inferred from
C     leshouche.inc.  That matters here more than it did at LO -- an
C     NLO process has several FKS subprocesses whose particle
C     orderings need not agree with one another.
C
C     The two cuts, on the RECONSTRUCTED pairs, because this process
C     has no z in its final state:
C
C       1) pt of each (l+ l-) system > the pt threshold.  In
C          p p > z z that threshold is the run card's
C          pt_min_pdg = {23: X}, applied by setcuts.f to both z.
C       2) |m(l+ l-) - MZ| < bwcutoff * WZ on each pair, which is the
C          virtuality window MadSpin samples in at the same BW_cut.
C
C     No number is written here.  The pt threshold is read out of
C     THIS run's own pt_min_pdg entry -- {23: X} is natively inert on
C     a process with no z, exactly as ptheavy was in the loop-induced
C     study, which is what makes it safe to reuse as a carrier -- and
C     bwcutoff comes from run.inc, MZ and WZ from coupl.inc.  So the
C     two sides of the comparison cannot drift apart by someone
C     editing one card and not the other.
C
C     IR safety: only charged leptons enter, photon recombination is
C     off for this process (no QED splittings), and both cuts are
C     smooth functions of the lepton momenta, so no QCD emission can
C     move an event across either boundary.
C     **************************************************************
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'run.inc'
      INCLUDE 'cuts.inc'
      INCLUDE 'coupl.inc'
      INTEGER ISTATUS(NEXTERNAL)
      INTEGER IPDG(NEXTERNAL)
      DOUBLE PRECISION P(0:4,NEXTERNAL)

      INTEGER I, J, IE1, IE2, IM1, IM2
      DOUBLE PRECISION PTMIN, MLO, MHI
      DOUBLE PRECISION EE, PX, PY, PZ, PT2, M2
      LOGICAL FIRSTCALL
      DATA FIRSTCALL /.TRUE./
      SAVE FIRSTCALL, PTMIN, MLO, MHI

      DUMMY_CUTS = .TRUE.

      IF (FIRSTCALL) THEN
C        --- the pt threshold, out of pt_min_pdg = {23: X} ---
         PTMIN = -1D0
         DO J = 1, PDG_CUT(0)
            IF (PDG_CUT(J) .EQ. 23) PTMIN = PTMIN4PDG(J)
         ENDDO
         IF (PTMIN .LT. 0D0) THEN
            WRITE(*,*) 'zz_equivalent_cuts_nlo: pt_min_pdg carries no'
            WRITE(*,*) 'entry for pdg 23, so the pt threshold of the'
            WRITE(*,*) 'p p > z z side cannot be read.  Set'
            WRITE(*,*) 'pt_min_pdg = {23: X} in the run card.'
            STOP 1
         ENDIF
         MLO = MDL_MZ - BWCUTOFF*MDL_WZ
         IF (MLO .LT. 0D0) MLO = 0D0
         MHI = MDL_MZ + BWCUTOFF*MDL_WZ
         WRITE(*,*) 'zz_equivalent_cuts_nlo ACTIVE: pt(ll) > ', PTMIN,
     &        ' GeV ; |m(ll) - ', MDL_MZ, '| < ', BWCUTOFF*MDL_WZ
         FIRSTCALL = .FALSE.
      ENDIF

C     --- locate one each of e+ e- mu+ mu- in THIS phase-space point ---
      IE1 = 0
      IE2 = 0
      IM1 = 0
      IM2 = 0
      DO I = 1, NEXTERNAL
         IF (ISTATUS(I) .NE. 1) CYCLE
         IF (IPDG(I) .EQ. -11) IE1 = I
         IF (IPDG(I) .EQ.  11) IE2 = I
         IF (IPDG(I) .EQ. -13) IM1 = I
         IF (IPDG(I) .EQ.  13) IM2 = I
      ENDDO
      IF (IE1*IE2*IM1*IM2 .EQ. 0) THEN
         WRITE(*,*) 'zz_equivalent_cuts_nlo: could not locate one each'
         WRITE(*,*) 'of e+ e- mu+ mu- among the final state'
         STOP 1
      ENDIF

C     --- the (e+ e-) system ---
      EE = P(0,IE1) + P(0,IE2)
      PX = P(1,IE1) + P(1,IE2)
      PY = P(2,IE1) + P(2,IE2)
      PZ = P(3,IE1) + P(3,IE2)
      PT2 = PX*PX + PY*PY
      M2 = EE*EE - PX*PX - PY*PY - PZ*PZ
      IF (PTMIN .GT. 0D0 .AND. PT2 .LT. PTMIN*PTMIN) THEN
         DUMMY_CUTS = .FALSE.
         RETURN
      ENDIF
      IF (BWCUTOFF .GT. 0D0) THEN
         IF (M2 .LT. MLO*MLO .OR. M2 .GT. MHI*MHI) THEN
            DUMMY_CUTS = .FALSE.
            RETURN
         ENDIF
      ENDIF

C     --- the (mu+ mu-) system ---
      EE = P(0,IM1) + P(0,IM2)
      PX = P(1,IM1) + P(1,IM2)
      PY = P(2,IM1) + P(2,IM2)
      PZ = P(3,IM1) + P(3,IM2)
      PT2 = PX*PX + PY*PY
      M2 = EE*EE - PX*PX - PY*PY - PZ*PZ
      IF (PTMIN .GT. 0D0 .AND. PT2 .LT. PTMIN*PTMIN) THEN
         DUMMY_CUTS = .FALSE.
         RETURN
      ENDIF
      IF (BWCUTOFF .GT. 0D0) THEN
         IF (M2 .LT. MLO*MLO .OR. M2 .GT. MHI*MHI) THEN
            DUMMY_CUTS = .FALSE.
            RETURN
         ENDIF
      ENDIF

      RETURN
      END
