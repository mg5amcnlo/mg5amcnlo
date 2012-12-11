      integer function cnv2000to2004(id,mconv)
C...convert (mconv=1) from PDG 2000 & StdHep 5.01 numbers
C                     to PDG 2004 & StdHep 5.02 numbers
C...     or (mconv=2) from PDG 2004 numbers to PDG 2000 numbers
C
C           id = particle identification number
C
#include "stdlun.inc"

      integer id,mconv

C... itab2000(I) converts all defined PDG 2000 particle ID's to the 2004 numbers
      parameter (mxlist = 66)
      integer itab2000(mxlist), itab2004(mxlist)

      data itab2000(1)  /  3000331/	! eta(tech)0
      data itab2000(2)  /  3100111/	! pi(1)(tech)22
      data itab2000(3)  /  3200111/	! pi(8)(tech)22
      data itab2000(4)  /  3100113/	! rho(tech)11
      data itab2000(5)  /  3200113/	! rho(tech)12
      data itab2000(6)  /  3300113/	! rho(tech)21
      data itab2000(7)  /  3400113/	! rho(tech)22
      data itab2000(8)  /        0/	! rho(1900)0
      data itab2000(9)  /        0/	! rho(1900)+
      data itab2000(10) /  9030113/	! rho(2150)0
      data itab2000(11) /  9030213/	! rho(2150)+
      data itab2000(12) /  9000115/	! a(2)(1660)0
      data itab2000(13) /  9000215/	! a(2)(1660)+
      data itab2000(14) /  9010115/	! a(2)(1750)0
      data itab2000(15) /  9010215/	! a(2)(1750)+
      data itab2000(16) /  9020115/	! pi(2)(2100)0
      data itab2000(17) /  9020215/	! pi(2)(2100)+
      data itab2000(18) /        0/	! pi(3)(1990)0
      data itab2000(19) /        0/	! pi(3)(1990)+
      data itab2000(20) /  9000117/	! pi(3)(2250)0
      data itab2000(21) /  9000217/	! pi(3)(2250)+
      data itab2000(22) /        0/	! f(0)(2330)
      data itab2000(23) /        0/	! h(1)(1595)
      data itab2000(24) /   100225/	! f(2)(1810)
      data itab2000(25) /  9030225/	! f(2)(1950)
      data itab2000(26) /        0/	! f(2)(1910)
      data itab2000(27) /  9040225/	! f(2)(2150)
      data itab2000(28) /  9050225/	! f(2)(2300)
      data itab2000(29) /  9060225/	! f(2)(2340)
      data itab2000(30) /        0/	! pi'_tech0
      data itab2000(31) /  3000221/	! eta_tech
      data itab2000(32) /        0/	! V_8
      data itab2000(33) /        0/	! pi_tech_1_22
      data itab2000(34) /        0/	! pi_tech_8_22
      data itab2000(35) /        0/	! rho_tech_11
      data itab2000(36) /        0/	! rho_tech_12
      data itab2000(37) /        0/	! rho_tech_21
      data itab2000(38) /        0/	! rho_tech_22
      data itab2000(39) /        0/	! Theta+
      data itab2000(40) /        0/	! Phi--
      data itab2000(41) /   200111/	! pi(1800)
      data itab2000(42) /   200211/	! pi(1800)
      data itab2000(43) /  9020113/	! a(1)(1640)
      data itab2000(44) /  9020213/	! a(1)(1640)
      data itab2000(45) /        0/	! eta(1405)
      data itab2000(46) /  9020221/	! f(0)(1500)
      data itab2000(47) /  9030221/	! f(0)(2020)
      data itab2000(48) /  9040221/	! f(0)(2060)
      data itab2000(49) /        0/	! f(0)(2100)
      data itab2000(50) /  9050221/	! f(0)(2200)
      data itab2000(51) /  9060221/	! eta(2225)
      data itab2000(52) /        0/	! f(0)(2330)
      data itab2000(53) /   200221/	! eta(1760)
      data itab2000(54) /   100335/	! f(2)(2010)
      data itab2000(55) /  9000339/	! f(J)(2220)
      data itab2000(56) /  9000229/	! f(4)(2300)
      data itab2000(57) /        0/	! K*(0)(800)
      data itab2000(58) /        0/	! K*(0)(800)
      data itab2000(59) /   200311/	! K(1830)
      data itab2000(60) /   200321/	! K(1830)
      data itab2000(61) /  9000311/	! K*(0)(1950)
      data itab2000(62) /  9000321/	! K*(0)(1950)
      data itab2000(63) /  9000313/	! K(1)(1650)
      data itab2000(64) /  9000323/	! K(1)(1650)
      data itab2000(65) /   100315/	! K*(2)(1980)
      data itab2000(66) /   100325/	! K*(2)(1980)

      data itab2004(1)  /  3100221/	! eta(tech)0
      data itab2004(2)  /  3060111/	! pi(1)(tech)22
      data itab2004(3)  /  3160111/	! pi(8)(tech)22
      data itab2004(4)  /  3130113/	! rho(tech)11
      data itab2004(5)  /  3140113/	! rho(tech)12
      data itab2004(6)  /  3150113/	! rho(tech)21
      data itab2004(7)  /  3160113/	! rho(tech)22
      data itab2004(8)  /  9020113/	! rho(1900)0
      data itab2004(9)  /  9020213/	! rho(1900)+
      data itab2004(10) /  9030113/	! rho(2150)0
      data itab2004(11) /  9030213/	! rho(2150)+
      data itab2004(12) /        0/	! a(2)(1700)0 from a(2)(1660)
      data itab2004(13) /        0/	! a(2)(1700)+
      data itab2004(14) /        0/	! a(2)(1700)0 from a(2)(1750)
      data itab2004(15) /        0/	! a(2)(1700)+
      data itab2004(16) /  9000115/	! pi(2)(2100)0
      data itab2004(17) /  9000215/	! pi(2)(2100)+
      data itab2004(18) /  9000117/	! pi(3)(1990)0
      data itab2004(19) /  9000217/	! pi(3)(1990)+
      data itab2004(20) /  9010117/	! pi(3)(2250)0
      data itab2004(21) /  9010217/	! pi(3)(2250)+
      data itab2004(22) /  9070221/	! f(0)(2330)
      data itab2004(23) /        0/	! h(1)(1595)
      data itab2004(24) /  9030225/	! f(2)(1810)
      data itab2004(25) /  9050225/	! f(2)(1950)
      data itab2004(26) /  9040225/	! f(2)(1910)
      data itab2004(27) /  9070225/	! f(2)(2150)
      data itab2004(28) /  9080225/	! f(2)(2300)
      data itab2004(29) /  9090225/	! f(2)(2340)
      data itab2004(30) /  3000221/	! pi'_tech0
      data itab2004(31) /  3010221/	! eta_tech
      data itab2004(32) /  3100021/	! V_8
      data itab2004(33) /  3060111/	! pi_tech_1_22
      data itab2004(34) /  3160111/	! pi_tech_8_22
      data itab2004(35) /  3130113/	! rho_tech_11
      data itab2004(36) /  3140113/	! rho_tech_12
      data itab2004(37) /  3150113/	! rho_tech_21
      data itab2004(38) /  3160113/	! rho_tech_22
      data itab2004(39) /  9221132/	! Theta+
      data itab2004(40) /  9331122/	! Phi--
      data itab2004(41) /  9010111/	! pi(1800)
      data itab2004(42) /  9010211/	! pi(1800)
      data itab2004(43) /        0/	! a(1)(1640)
      data itab2004(44) /        0/	! a(1)(1640)
      data itab2004(45) /  9020221/	! eta(1405)
      data itab2004(46) /  9030221/	! f(0)(1500)
      data itab2004(47) /  9040221/	! f(0)(2020)
      data itab2004(48) /        0/	! f(0)(2060)
      data itab2004(49) /  9050221/	! f(0)(2100)
      data itab2004(50) /  9060221/	! f(0)(2200)
      data itab2004(51) /        0/	! eta(2225)
      data itab2004(52) /  9070221/	! f(0)(2330)
      data itab2004(53) /        0/	! eta(1760)
      data itab2004(54) /  9060225/	! f(2)(2010)
      data itab2004(55) /  9000229/	! f(J)(2220)
      data itab2004(56) /  9010229/	! f(4)(2300)
      data itab2004(57) /  9000311/	! K*(0)(800)
      data itab2004(58) /  9000321/	! K*(0)(800)
      data itab2004(59) /        0/	! K(1830)
      data itab2004(60) /        0/	! K(1830)
      data itab2004(61) /  9010311/	! K*(0)(1950)
      data itab2004(62) /  9010321/	! K*(0)(1950)
      data itab2004(63) /        0/	! K(1)(1650)
      data itab2004(64) /        0/	! K(1)(1650)
      data itab2004(65) /        0/	! K*(2)(1980)
      data itab2004(66) /        0/	! K*(2)(1980)

      save itab2000,itab2004

      cnv2000to2004=id
      ida=iabs(id)

      if(ida.eq.0)then
D        write(lnhout,1001) 
      elseif(mconv.eq.1)then
        do i = 1,mxlist
           if(ida.eq.itab2000(i))then
              cnv2000to2004=isign(itab2004(i),id)
           endif
        enddo
D        if(cnv2000to2004.eq.0 ) write(lnhout,1002) id
      elseif(mconv.eq.2)then
        do i = 1,mxlist
           if(ida.eq.itab2004(i))then
	      if( itab2000(i) .gt. 0 ) then
                 cnv2000to2004=isign(itab2000(i),id)
	      else
                 cnv2000to2004=0
	      endif
           endif
        enddo
D        if(cnv2000to2004.eq.0 ) write(lnhout,1003) id
      endif
      return
 1001 format(' cnv2000to2004 1: particle id is zero')
 1002 format(' cnv2000to2004 4: ID ',i7,
     1            ' is not a defined PDG 2004 state')
 1003 format(' cnv2000to2004 3: ID ',i7,
     1            ' is not a defined PDG 2000 state')
      end
