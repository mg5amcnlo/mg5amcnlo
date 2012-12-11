      integer function cnv2000to2002(id,mconv)
C...convert (mconv=1) from PDG 2000 & StdHep 5.01 numbers
C                     to PDG 2002 & StdHep 5.02 numbers
C...     or (mconv=2) from PDG 2002 numbers to PDG 2000 numbers
C
C           id = particle identification number
C
#include "stdlun.inc"

      integer id,mconv

C... itab2000(I) converts all defined PDG 2000 particle ID's to the 2002 numbers
      parameter (mxlist = 29)
      integer itab2000(mxlist), itab2002(mxlist)

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

      data itab2002(1)  /  3100221/	! eta(tech)0
      data itab2002(2)  /  3060111/	! pi(1)(tech)22
      data itab2002(3)  /  3160111/	! pi(8)(tech)22
      data itab2002(4)  /  3130113/	! rho(tech)11
      data itab2002(5)  /  3140113/	! rho(tech)12
      data itab2002(6)  /  3150113/	! rho(tech)21
      data itab2002(7)  /  3160113/	! rho(tech)22
      data itab2002(8)  /  9030113/	! rho(1900)0
      data itab2002(9)  /  9030213/	! rho(1900)+
      data itab2002(10) /  9040113/	! rho(2150)0
      data itab2002(11) /  9040213/	! rho(2150)+
      data itab2002(12) /   100115/	! a(2)(1700)0 from a(2)(1660)
      data itab2002(13) /   100215/	! a(2)(1700)+
      data itab2002(14) /   100115/	! a(2)(1700)0 from a(2)(1750)
      data itab2002(15) /   100215/	! a(2)(1700)+
      data itab2002(16) /  9000115/	! pi(2)(2100)0
      data itab2002(17) /  9000215/	! pi(2)(2100)+
      data itab2002(18) /  9000117/	! pi(3)(1990)0
      data itab2002(19) /  9000217/	! pi(3)(1990)+
      data itab2002(20) /  9010117/	! pi(3)(2250)0
      data itab2002(21) /  9010217/	! pi(3)(2250)+
      data itab2002(22) /  9070221/	! f(0)(2330)
      data itab2002(23) /  9010223/	! h(1)(1595)
      data itab2002(24) /  9030225/	! f(2)(1810)
      data itab2002(25) /   100225/	! f(2)(1950)
      data itab2002(26) /  9040225/	! f(2)(1910)
      data itab2002(27) /  9050225/	! f(2)(2150)
      data itab2002(28) /  9060225/	! f(2)(2300)
      data itab2002(29) /  9070225/	! f(2)(2340)


      save itab2000,itab2002

      cnv2000to2002=id
      ida=iabs(id)

      if(ida.eq.0)then
D        write(lnhout,1001) 
      elseif(mconv.eq.1)then
        do i = 1,mxlist
           if(ida.eq.itab2000(i))then
              cnv2000to2002=isign(itab2002(i),id)
           endif
        enddo
D        if(cnv2000to2002.eq.0 ) write(lnhout,1002) id
      elseif(mconv.eq.2)then
        do i = 1,mxlist
           if(ida.eq.itab2002(i))then
	      if( itab2000(i) .gt. 0 ) then
                 cnv2000to2002=isign(itab2000(i),id)
	      else
                 cnv2000to2002=0
	      endif
           endif
        enddo
D        if(cnv2000to2002.eq.0 ) write(lnhout,1003) id
      endif
      return
 1001 format(' CNV2000TO2002 1: particle id is zero')
 1002 format(' CNV2000TO2002 4: ID ',i7,
     1            ' is not a defined PDG 2002 state')
 1003 format(' CNV2000TO2002 3: ID ',i7,
     1            ' is not a defined PDG 2000 state')
      end
