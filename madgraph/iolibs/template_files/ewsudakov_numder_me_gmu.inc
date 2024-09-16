      subroutine PAR_REN_%(mename)s(P,NHEL,hell,invariants)
C     compute the contribution due to the parameter renormalisation
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P(0:3,NEXTERNAL-1), ANS_SUMMED
      INTEGER NHEL(NEXTERNAL-1),HELL
      DOUBLE PRECISION INVARIANTS(NEXTERNAL-1, NEXTERNAL-1)
      INCLUDE 'coupl.inc'
      INCLUDE '../../Source/MODEL/input.inc'

C     ipara = 1->Gmu; 2->MZ; 3->MW; 4->MT/YMT; 5->MH; 6->tadpole
      INTEGER IMAXPARA, IPARA
      PARAMETER (IMAXPARA=6)

      INCLUDE 'orders.inc'
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_DER(AMP_SPLIT_SIZE,IMAXPARA)
      COMMON /TO_AMP_SPLIT_EWSUD_DER/ AMP_SPLIT_EWSUD_DER
      DOUBLE COMPLEX AMP_SPLIT_EWSUD_DER2(AMP_SPLIT_SIZE,IMAXPARA)
      COMMON /TO_AMP_SPLIT_EWSUD_DER2/ AMP_SPLIT_EWSUD_DER2

      INTEGER I, J

      DOUBLE PRECISION XDELTA, XSAVE, XUP, XDN, MATXUP, MATXDN, MATXCEN
      DOUBLE COMPLEX AMP_SPLIT_UP(AMP_SPLIT_SIZE),
     $  AMP_SPLIT_DN(AMP_SPLIT_SIZE), AMP_SPLIT_CEN(AMP_SPLIT_SIZE)

      DOUBLE COMPLEX AMP_SPLIT_EWSUD(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD/ AMP_SPLIT_EWSUD

      DOUBLE COMPLEX AMP_SPLIT_EWSUD_LO2(AMP_SPLIT_SIZE)
      COMMON /TO_AMP_SPLIT_EWSUD_LO2/ AMP_SPLIT_EWSUD_LO2

      DOUBLE COMPLEX MT_DER_LO2
      COMMON /TO_MT_DER_LO2/MT_DER_LO2 


      DOUBLE COMPLEX AMP_SPLIT_UP2(AMP_SPLIT_SIZE),
     $  AMP_SPLIT_DN2(AMP_SPLIT_SIZE), AMP_SPLIT_CEN2(AMP_SPLIT_SIZE)

      XDELTA = 1D-5

C     check that mdl_mt and mdl_ymt are equal
      IF (MDL_MT.NE.MDL_YMT) THEN
         write(*,*) 'ERROR mdl_mt and mdl_ymt should be equal!', mdl_mt, mdl_ymt
         stop 1
      ENDIF

      IF (MDL_NTADPOLE.NE.1D0) THEN
        WRITE(*,*) 'Set in the paramcard ntadpole to EXACTLY 1 and not', mdl_ntadpole
        STOP 2
      ENDIF

      DO I = 1, IMAXPARA
        IF (I.EQ.1) THEN
          XSAVE = MDL_GF
        ELSE IF (I.EQ.2) THEN
          XSAVE = MDL_MZ
        ELSE IF (I.EQ.3) THEN
          XSAVE = MDL_MW
        ELSE IF (I.EQ.4) THEN
          XSAVE = MDL_MT
        ELSE IF (I.EQ.5) THEN
          XSAVE = MDL_MH
        ELSE IF (I.EQ.6) THEN
          XSAVE = MDL_NTADPOLE
        ENDIF

        XUP = XSAVE * (1D0 + XDELTA)
        XDN = XSAVE * (1D0 - XDELTA)

C        call %(mename)s(P,NHEL,%(hell)s matxcen)
C        amp_split_cen(:) = amp_split_ewsud(:)
        
C       vary the parameter up
        IF (I.EQ.1) THEN
          MDL_GF = XUP
        ELSE IF (I.EQ.2) THEN
          MDL_MZ = XUP
        ELSE IF (I.EQ.3) THEN
          MDL_MW = XUP
        ELSE IF (I.EQ.4) THEN
          MDL_MT = XUP
          MDL_YMT = XUP
        ELSE IF (I.EQ.5) THEN
          MDL_MH = XUP
        ELSE IF (I.EQ.6) THEN
          MDL_NTADPOLE = XUP
        ENDIF

        CALL COUP()
        call %(mename)s(P,NHEL,%(hell)s matxup)
        AMP_SPLIT_UP(:) = AMP_SPLIT_EWSUD(:)
        AMP_SPLIT_UP2(:) = AMP_SPLIT_EWSUD_LO2(:)

C       vary the parameter down
        IF (I.EQ.1) THEN
          MDL_GF = XDN
        ELSE IF (I.EQ.2) THEN
          MDL_MZ = XDN
        ELSE IF (I.EQ.3) THEN
          MDL_MW = XDN
        ELSE IF (I.EQ.4) THEN
          MDL_MT = XDN
          MDL_YMT = XDN
        ELSE IF (I.EQ.5) THEN
          MDL_MH = XDN
        ELSE IF (I.EQ.6) THEN
          MDL_NTADPOLE = XDN
        ENDIF

        CALL COUP()
        call %(mename)s(P,NHEL,%(hell)s matxdn)
        AMP_SPLIT_DN(:) = AMP_SPLIT_EWSUD(:)
        AMP_SPLIT_DN2(:) = AMP_SPLIT_EWSUD_LO2(:)

        AMP_SPLIT_EWSUD_DER(:,I) = (AMP_SPLIT_UP(:)-AMP_SPLIT_DN(:))
     $   /(2*XSAVE*XDELTA)

        AMP_SPLIT_EWSUD_DER2(:,I) = (AMP_SPLIT_UP2(:)-AMP_SPLIT_DN2(:))
     $   /(2*XSAVE*XDELTA)


C       reset the parameter 
        IF (I.EQ.1) THEN
          MDL_GF = XSAVE
        ELSE IF (I.EQ.2) THEN
          MDL_MZ = XSAVE
        ELSE IF (I.EQ.3) THEN
          MDL_MW = XSAVE
        ELSE IF (I.EQ.4) THEN
          MDL_MT = XSAVE
          MDL_YMT = XSAVE
        ELSE IF (I.EQ.5) THEN
          MDL_MH = XSAVE
        ELSE IF (I.EQ.6) THEN
          MDL_NTADPOLE = XSAVE
        ENDIF

        CALL COUP()

      ENDDO

      CALL GET_PAR_REN_GMU(INVARIANTS)

      RETURN
      END




