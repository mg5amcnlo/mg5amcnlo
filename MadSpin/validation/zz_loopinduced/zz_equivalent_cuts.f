      logical function dummy_cuts(P)
C     **************************************************************
C     Make  g g > e+ e- mu+ mu- / a  (loop-induced) physically
C     equivalent to  g g > z z  (loop-induced) + MadSpin.
C
C     Two cuts are applied, both on the RECONSTRUCTED lepton pairs,
C     because sample B has no z in its final state:
C
C       1) pt of each (l+ l-) system > ptheavy.  In  g g > z z  the
C          run card's ptheavy acts on the two z directly; at 2 -> 2
C          pt(z1) = pt(z2) exactly (the initial state carries no
C          transverse momentum), so "at least one heavy above
C          ptheavy" and "both above ptheavy" are the same cut, and
C          it maps onto both reconstructed pairs here.
C
C       2) |m(l+ l-) - MZ| < bwcutoff * WZ on each pair.  This is the
C          same virtuality window MadSpin samples in when BW_cut is
C          set to the same number, and it is what makes the fully
C          off-shell calculation comparable to a narrow-window
C          decayed one.
C
C     No number is written here: ptheavy and bwcutoff come from this
C     run's own run_card.dat (cuts.inc / run.inc) and MZ, WZ from its
C     own param_card.dat (coupl.inc).  In sample B ptheavy has no
C     native effect -- setcuts.f flags a particle heavy only if its
C     mass exceeds 10 GeV and every final state here is a massless
C     lepton -- so the value is free to be read and reused.
C
C     Particle ordering is fixed by leshouche.inc of the one and only
C     subprocess P0_gg_llll,
C         IDUP = 21, 21, -11, 11, -13, 13
C     i.e. 1=g 2=g 3=e+ 4=e- 5=mu+ 6=mu-.  It is re-asserted at run
C     time below rather than trusted, and a mismatch stops the run
C     instead of silently cutting the wrong pair.
C     **************************************************************
      IMPLICIT NONE
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE '../../Source/vector.inc'
      INCLUDE 'run.inc'
      INCLUDE 'cuts.inc'
      INCLUDE 'coupl.inc'
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER MAXPROC_LES, MAXSPROC_LES
      PARAMETER (MAXPROC_LES=1, MAXSPROC_LES=1)
      INTEGER IDUP(NEXTERNAL,MAXPROC_LES,MAXSPROC_LES)
      INTEGER MOTHUP(2,NEXTERNAL)
      INTEGER ICOLUP(2,NEXTERNAL,MAXPROC_LES,MAXSPROC_LES)
      INCLUDE 'leshouche.inc'
      INTEGER I, IE1, IE2, IM1, IM2
      REAL*8 PTEE2, PTMM2, MEE2, MMM2, MLO, MHI, PTMIN2
      REAL*8 PX, PY, PZ, EE
      LOGICAL FIRSTCALL
      DATA FIRSTCALL /.TRUE./
      SAVE FIRSTCALL, IE1, IE2, IM1, IM2

      DUMMY_CUTS = .TRUE.

      IF (FIRSTCALL) THEN
         IF (NEXTERNAL .NE. 6) THEN
            WRITE(*,*) 'zz_equivalent_cuts: expected 6 external legs,'
            WRITE(*,*) 'found ', NEXTERNAL
            STOP 1
         ENDIF
         IE1 = 0
         IE2 = 0
         IM1 = 0
         IM2 = 0
         DO I = 3, NEXTERNAL
            IF (IDUP(I,1,1) .EQ. -11) IE1 = I
            IF (IDUP(I,1,1) .EQ.  11) IE2 = I
            IF (IDUP(I,1,1) .EQ. -13) IM1 = I
            IF (IDUP(I,1,1) .EQ.  13) IM2 = I
         ENDDO
         IF (IE1*IE2*IM1*IM2 .EQ. 0) THEN
            WRITE(*,*) 'zz_equivalent_cuts: could not locate one each'
            WRITE(*,*) 'of e+ e- mu+ mu- in leshouche.inc'
            STOP 1
         ENDIF
         WRITE(*,*) 'zz_equivalent_cuts ACTIVE: e+ e- at ', IE1, IE2,
     &        ' mu+ mu- at ', IM1, IM2
         WRITE(*,*) 'zz_equivalent_cuts: pt(ll) > ', PTHEAVY,
     &        ' GeV ; |m(ll) - ', MDL_MZ, '| < ', BWCUTOFF*MDL_WZ
         FIRSTCALL = .FALSE.
      ENDIF

      PTMIN2 = PTHEAVY*PTHEAVY
      MLO = MDL_MZ - BWCUTOFF*MDL_WZ
      IF (MLO .LT. 0D0) MLO = 0D0
      MHI = MDL_MZ + BWCUTOFF*MDL_WZ

C     --- the (e+ e-) system ---
      EE = P(0,IE1) + P(0,IE2)
      PX = P(1,IE1) + P(1,IE2)
      PY = P(2,IE1) + P(2,IE2)
      PZ = P(3,IE1) + P(3,IE2)
      PTEE2 = PX*PX + PY*PY
      MEE2 = EE*EE - PX*PX - PY*PY - PZ*PZ
      IF (PTHEAVY .GT. 0D0 .AND. PTEE2 .LT. PTMIN2) THEN
         DUMMY_CUTS = .FALSE.
         RETURN
      ENDIF
      IF (BWCUTOFF .GT. 0D0) THEN
         IF (MEE2 .LT. MLO*MLO .OR. MEE2 .GT. MHI*MHI) THEN
            DUMMY_CUTS = .FALSE.
            RETURN
         ENDIF
      ENDIF

C     --- the (mu+ mu-) system ---
      EE = P(0,IM1) + P(0,IM2)
      PX = P(1,IM1) + P(1,IM2)
      PY = P(2,IM1) + P(2,IM2)
      PZ = P(3,IM1) + P(3,IM2)
      PTMM2 = PX*PX + PY*PY
      MMM2 = EE*EE - PX*PX - PY*PY - PZ*PZ
      IF (PTHEAVY .GT. 0D0 .AND. PTMM2 .LT. PTMIN2) THEN
         DUMMY_CUTS = .FALSE.
         RETURN
      ENDIF
      IF (BWCUTOFF .GT. 0D0) THEN
         IF (MMM2 .LT. MLO*MLO .OR. MMM2 .GT. MHI*MHI) THEN
            DUMMY_CUTS = .FALSE.
            RETURN
         ENDIF
      ENDIF

      RETURN
      END
