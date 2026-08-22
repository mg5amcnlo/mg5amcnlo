      logical function dummy_cuts(P)
C     **************************************************************
C     CONTROL variant of zz_equivalent_cuts.f: the pt cut on the two
C     reconstructed lepton pairs ONLY, with the Breit-Wigner mass
C     window left off.
C
C     Its whole purpose is to measure what the mass window removes.
C     Running the same process with this file and with the real one
C     differs in exactly one cut, so the ratio of the two cross
C     sections IS the retained fraction of the two Breit-Wigners --
C     which can then be held against MadSpin's own
C     bw_retained_fraction(M_Z, Gamma_Z, 15)**2.  A cut that were
C     silently ignored would make that ratio 1.
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
      REAL*8 PX, PY, PTMIN2
      LOGICAL FIRSTCALL
      DATA FIRSTCALL /.TRUE./
      SAVE FIRSTCALL, IE1, IE2, IM1, IM2

      DUMMY_CUTS = .TRUE.

      IF (FIRSTCALL) THEN
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
            WRITE(*,*) 'zz_ptonly_cuts: could not locate the leptons'
            STOP 1
         ENDIF
         WRITE(*,*) 'zz_ptonly_cuts ACTIVE (CONTROL: no mass window)',
     &        ' pt(ll) > ', PTHEAVY
         FIRSTCALL = .FALSE.
      ENDIF

      IF (PTHEAVY .LE. 0D0) RETURN
      PTMIN2 = PTHEAVY*PTHEAVY

      PX = P(1,IE1) + P(1,IE2)
      PY = P(2,IE1) + P(2,IE2)
      IF (PX*PX + PY*PY .LT. PTMIN2) THEN
         DUMMY_CUTS = .FALSE.
         RETURN
      ENDIF

      PX = P(1,IM1) + P(1,IM2)
      PY = P(2,IM1) + P(2,IM2)
      IF (PX*PX + PY*PY .LT. PTMIN2) THEN
         DUMMY_CUTS = .FALSE.
         RETURN
      ENDIF

      RETURN
      END
