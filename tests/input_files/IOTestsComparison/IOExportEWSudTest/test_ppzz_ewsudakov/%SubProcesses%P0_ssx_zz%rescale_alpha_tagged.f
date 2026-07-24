      DOUBLE PRECISION FUNCTION GET_RESCALE_ALPHA_FACTOR(NTAG,QED_POW)
C     returns the power of ratios of alpha needed to compensate 
C     for the presence of external tagged photons
C     It is automatically written, and it detects whether the
C     starting model is in the Gmu or alpha0 scheme.
C     ntag is the number of tagged photons, qed_pow the power of gweak
C     of the contribution considered
      IMPLICIT NONE
      INTEGER NTAG, QED_POW
      INCLUDE '../../Source/MODEL/input.inc'

      GET_RESCALE_ALPHA_FACTOR = 1D0
      IF (NTAG.EQ.0) RETURN

      GET_RESCALE_ALPHA_FACTOR = 1D0

      RETURN
      END


      DOUBLE PRECISION FUNCTION GET_VIRTUAL_A0GMU_CONV(QED_POW, NTAGPH
     $ , IVIRT, BORN_WGT)
C     returns the piece to compensate the a0<>Gmu change of scheme for
C      the
C     virtuals (single pole if ivirt = 1, finite if ivirt = 0
      IMPLICIT NONE
      INTEGER QED_POW, NTAGPH, IVIRT
      DOUBLE PRECISION BORN_WGT
      INCLUDE '../../Source/MODEL/input.inc'

      IF (IVIRT.EQ.1) THEN
C       single 
        GET_VIRTUAL_A0GMU_CONV = 0D0
      ELSE IF (IVIRT.EQ.0) THEN
C       finite part
        GET_VIRTUAL_A0GMU_CONV = 0D0
      ELSE
        WRITE(*,*) 'Error get_virtual_a0Gmu_conv: Invalid ivirt', IVIRT
      ENDIF

      RETURN
      END

