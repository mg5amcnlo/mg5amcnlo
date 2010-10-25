c This file has been generated automatically by FeynRules

c Version: 1.4   Date: 25. 10. 2010,    11:39

c Internal parameters definition :

c Parameters that should not be recomputed event by event.
      if(readlha) then
      aEW = 1/aEWM1
      sw2 = 1 - MW**2/MZ**2
      ee = 3.5449077018110318*sqrt(aEW)
      cw = sqrt(1. - 1.*sw2)
      sw = sqrt(sw2)
      gw = ee/sw
      g1 = ee/cw
      G = 3.5449077018110318*sqrt(aS)
      v = (2*MW*sw)/ee
      lam = MH**2/(2.*v**2)
      muH = sqrt(lam*v**2)
      ye = 0
      ym = 0
      ytau = (sqrt(2.)*ymtau)/v
      yu = 0
      yc = (sqrt(2.)*ymc)/v
      yt = (sqrt(2.)*ymt)/v
      yd = 0
      ys = 0
      yb = (sqrt(2.)*ymb)/v
      CKM11 = Cos(cabi)
      CKM12 = Sin(cabi)
      CKM13 = 0
      CKM21 = -Sin(cabi)
      CKM22 = Cos(cabi)
      CKM23 = 0
      CKM31 = 0
      CKM32 = 0
      CKM33 = 1
      Sqrt2 = sqrt(2.)
      SqrtPi = sqrt(3.141592653589793)
      Pgw2 = gw**2
      Psw2 = sw**2
      Pcw2 = cw**2
      CONJCKM23 = conjg(CKM23)
      CONJCKM33 = conjg(CKM33)
      CONJCKM13 = conjg(CKM13)
      CONJCKM21 = conjg(CKM21)
      CONJCKM31 = conjg(CKM31)
      CONJCKM11 = conjg(CKM11)
      CONJCKM22 = conjg(CKM22)
      CONJCKM32 = conjg(CKM32)
      CONJCKM12 = conjg(CKM12)
      endif
c Parameters that should be recomputed at an event by even basis.
      GGT1 = sqrt(G**2)


c Definition of the EW coupling used in the write out of aqed
      gal(1) = ee
      gal(2) = ee


c Definition of DUM symbols
      DUM0 = 0
      DUM1 = 1


