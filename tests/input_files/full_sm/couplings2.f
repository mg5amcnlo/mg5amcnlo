      subroutine coup2(readlha)

      implicit none
      logical readlha

      include 'input.inc'
      include 'coupl.inc'
      include 'intparam_definition.inc'


c Interactions associated with 2
      if(readlha) then
      MGVX3 = -(gw*sw)
      MGVX5 = cw*gw
      MGVX4 = Pgw2*Psw2
      MGVX7 = cw*Pgw2*sw
      MGVX6 = -Pgw2
      MGVX8 = Pcw2*Pgw2

      MGVX10 = -6*lam*v
      MGVX9 = -6*lam
      MGVX12 = (ee**2*v)/(2.*Psw2)
      MGVX14 = ee**2*v + (ee**2*Pcw2*v)/(2.*Psw2) + (ee**2*Psw2*v)/
     +(2.*Pcw2)
      MGVX11 = ee**2/(2.*Psw2)
      MGVX13 = ee**2 + (ee**2*Pcw2)/(2.*Psw2) + (ee**2*Psw2)/(2.*Pc
     +w2)

      MGVX24(1) = ee
      MGVX24(2) = ee
      MGVX30(1) = -(ee/(Sqrt2*sw))
      MGVX30(2) = 0
      MGVX57(1) = (cw*ee)/(2.*sw) - (ee*sw)/(2.*cw)
      MGVX57(2) = -((ee*sw)/cw)
      MGVX63(1) = -(cw*ee)/(2.*sw) - (ee*sw)/(2.*cw)
      MGVX63(2) = 0
      MGVX69(1) = -(ye/Sqrt2)
      MGVX69(2) = -(ye/Sqrt2)
      MGVX70(1) = -(ym/Sqrt2)
      MGVX70(2) = -(ym/Sqrt2)
      MGVX71(1) = -(ytau/Sqrt2)
      MGVX71(2) = -(ytau/Sqrt2)
      MGVX21(1) = ee/3.
      MGVX21(2) = ee/3.
      MGVX27(1) = (-2*ee)/3.
      MGVX27(2) = (-2*ee)/3.
      MGVX68(1) = -(yb/Sqrt2)
      MGVX68(2) = -(yb/Sqrt2)
      MGVX73(1) = -(yc/Sqrt2)
      MGVX73(2) = -(yc/Sqrt2)
      MGVX66(1) = -(yd/Sqrt2)
      MGVX66(2) = -(yd/Sqrt2)
      MGVX67(1) = -(ys/Sqrt2)
      MGVX67(2) = -(ys/Sqrt2)
      MGVX74(1) = -(yt/Sqrt2)
      MGVX74(2) = -(yt/Sqrt2)
      MGVX72(1) = -(yu/Sqrt2)
      MGVX72(2) = -(yu/Sqrt2)
      MGVX33(1) = -((CKM23*ee)/(Sqrt2*sw))
      MGVX33(2) = 0
      MGVX34(1) = -((CKM21*ee)/(Sqrt2*sw))
      MGVX34(2) = 0
      MGVX35(1) = -((CKM22*ee)/(Sqrt2*sw))
      MGVX35(2) = 0
      MGVX36(1) = -((CKM33*ee)/(Sqrt2*sw))
      MGVX36(2) = 0
      MGVX37(1) = -((CKM31*ee)/(Sqrt2*sw))
      MGVX37(2) = 0
      MGVX38(1) = -((CKM32*ee)/(Sqrt2*sw))
      MGVX38(2) = 0
      MGVX39(1) = -((CKM13*ee)/(Sqrt2*sw))
      MGVX39(2) = 0
      MGVX40(1) = -((CKM11*ee)/(Sqrt2*sw))
      MGVX40(2) = 0
      MGVX41(1) = -((CKM12*ee)/(Sqrt2*sw))
      MGVX41(2) = 0
      MGVX45(1) = -((CONJCKM23*ee)/(Sqrt2*sw))
      MGVX45(2) = 0
      MGVX46(1) = -((CONJCKM33*ee)/(Sqrt2*sw))
      MGVX46(2) = 0
      MGVX47(1) = -((CONJCKM13*ee)/(Sqrt2*sw))
      MGVX47(2) = 0
      MGVX48(1) = -((CONJCKM21*ee)/(Sqrt2*sw))
      MGVX48(2) = 0
      MGVX49(1) = -((CONJCKM31*ee)/(Sqrt2*sw))
      MGVX49(2) = 0
      MGVX50(1) = -((CONJCKM11*ee)/(Sqrt2*sw))
      MGVX50(2) = 0
      MGVX51(1) = -((CONJCKM22*ee)/(Sqrt2*sw))
      MGVX51(2) = 0
      MGVX52(1) = -((CONJCKM32*ee)/(Sqrt2*sw))
      MGVX52(2) = 0
      MGVX53(1) = -((CONJCKM12*ee)/(Sqrt2*sw))
      MGVX53(2) = 0
      MGVX54(1) = (cw*ee)/(2.*sw) + (ee*sw)/(6.*cw)
      MGVX54(2) = -(ee*sw)/(3.*cw)
      MGVX60(1) = -(cw*ee)/(2.*sw) + (ee*sw)/(6.*cw)
      MGVX60(2) = (2*ee*sw)/(3.*cw)
      endif

      return
      end
