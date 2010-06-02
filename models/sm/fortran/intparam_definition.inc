ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

c      Parameters that should not be recomputed event by event.
c      
      IF(READLHA) THEN
        AEW = 1.000000D+00/AEWM1
        G = 2.000000D+00*DSQRT(AS)*DSQRT(PI)
        CKM11 = COS(CABI)
        CKM12 = SIN(CABI)
        CKM21 = -SIN(CABI)
        CKM22 = COS(CABI)
        MW = DSQRT(MZ**2/2.000000D+00 + DSQRT(MZ**4/4.000000D
     $   +00 - (AEW*PI*MZ**2)/(GF*DSQRT(2.000000D+00))))
        EE = 2.000000D+00*DSQRT(AEW)*DSQRT(PI)
        SW2 = 1.000000D+00 - MW**2/MZ**2
        CW = DSQRT(1.000000D+00 - SW2)
        SW = DSQRT(SW2)
        G1 = EE/CW
        GW = EE/SW
        V = (2.000000D+00*MW*SW)/EE
        LAM = MH**2/(2.000000D+00*V**2)
        YB = (YMB*DSQRT(2.000000D+00))/V
        YC = (YMC*DSQRT(2.000000D+00))/V
        YT = (YMT*DSQRT(2.000000D+00))/V
        YTAU = (YMTAU*DSQRT(2.000000D+00))/V
        MUH = DSQRT(LAM*V**2)
        COMPLEXI = (0D0, 1.000000D+00)
        GW__EXP__2 = GW**2
        CW__EXP__2 = CW**2
        EE__EXP__2 = EE**2
        SW__EXP__2 = SW**2
        SQRT__2 =  DSQRT(2.000000D+00)
        CONJG__CKM11 =  CONJG(CKM11)
        CONJG__CKM12 =  CONJG(CKM12)
        CONJG__CKM21 =  CONJG(CKM21)
        CONJG__CKM22 =  CONJG(CKM22)
      ENDIF
c      
c      Parameters that should be recomputed at an event by even basis.
c      
      G__EXP__2 = G**2
c      
c      Definition of the EW coupling used in the write out of aqed
c      
      GAL(1) = 1D0
      GAL(2) = 1D0
      
c      
c      Definition of DUM symbols
c      
      DUM0 = 0
      DUM1 = 1
      
