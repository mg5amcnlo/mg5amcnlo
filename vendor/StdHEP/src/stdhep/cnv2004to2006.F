      integer function cnv2004to2006(id,mconv)
C...convert (mconv=1) from PDG 2004 & StdHep 5.01 numbers
C                     to PDG 2006 & StdHep 5.02 numbers
C...     or (mconv=2) from PDG 2006 numbers to PDG 2004 numbers
C
C           id = particle identification number
C
#include "stdlun.inc"

      integer id,mconv

C... itab2004(I) converts all defined PDG 2004 particle ID's to the 2006 numbers
C... itab2006(I) converts all defined PDG 2006 particle ID's to the 2004 numbers
      parameter (mxlist = 28)
      integer itab2004(mxlist), itab2006(mxlist)

      data itab2004(1)  /        0/	! a(1)(1640)0
      data itab2004(2)  /        0/	! a(1)(1640)+
      data itab2004(3)  /  9020113/	! rho(1900)0
      data itab2004(4)  /  9020213/	! rho(1900)+
      data itab2004(5)  /  9030113/	! rho(2150)0
      data itab2004(6)  /  9030213/	! rho(2150)+
      data itab2004(7)  /        0/	! a(2)(1700)0
      data itab2004(8)  /        0/	! a(2)(1700)+
      data itab2004(9)  /  9000115/	! pi(2)(2100)0
      data itab2004(10) /  9000215/	! pi(2)(2100)+
      data itab2004(11) /        0/	! eta(1760)
      data itab2004(12) /        0/	! eta(2225)
      data itab2004(13) /  9040221/	! f(0)(2020)
      data itab2004(14) /  9050221/	! f(0)(2100)
      data itab2004(15) /  9060221/	! f(0)(2200)
      data itab2004(16) /  9070221/	! f(0)(2330)
      data itab2004(17) /        0/	! h(1)(1595)
      data itab2004(18) /        0/	! K(1830)
      data itab2004(19) /        0/	! K(1830)
      data itab2004(20) /  9010311/	! K*(0)(1950)
      data itab2004(21) /  9010321/	! K*(0)(1950)
      data itab2004(22) /        0/	! K(1)(1650)
      data itab2004(23) /        0/	! K(1)(1650)
      data itab2004(24) /        0/	! K*(2)(1980)
      data itab2004(25) /        0/	! K*(2)(1980)
      data itab2004(26) /  9010315/	! K(2)(2250)
      data itab2004(27) /  9010325/	! K(2)(2250)
      data itab2004(28) /        0/	! chi(c2)(2P)

      data itab2006(1)  /  9020113/	! a(1)(1640)0
      data itab2006(2)  /  9020213/	! a(1)(1640)+
      data itab2006(3)  /  9030113/	! rho(1900)0
      data itab2006(4)  /  9030213/	! rho(1900)+
      data itab2006(5)  /  9040113/	! rho(2150)0
      data itab2006(6)  /  9040213/	! rho(2150)+
      data itab2006(7)  /  9000115/	! a(2)(1700)0
      data itab2006(8)  /  9000215/	! a(2)(1700)+
      data itab2006(9)  /  9010115/	! pi(2)(2100)0
      data itab2006(10) /  9010215/	! pi(2)(2100)+
      data itab2006(11) /  9040221/	! eta(1760)
      data itab2006(12) /  9080221/	! eta(2225)
      data itab2006(13) /  9050221/	! f(0)(2020)
      data itab2006(14) /  9060221/	! f(0)(2100)
      data itab2006(15) /  9070221/	! f(0)(2200)
      data itab2006(16) /        0/	! f(0)(2330)
      data itab2006(17) /  9010223/	! h(1)(1595)
      data itab2006(18) /  9010311/	! K(1830)0
      data itab2006(19) /  9010321/	! K(1830)+
      data itab2006(20) /  9020311/	! K*(0)(1950)
      data itab2006(21) /  9020321/	! K*(0)(1950)
      data itab2006(22) /  9000313/	! K(1)(1650)
      data itab2006(23) /  9000323/	! K(1)(1650)
      data itab2006(24) /  9010315/	! K*(2)(1980)
      data itab2006(25) /  9010325/	! K*(2)(1980)
      data itab2006(26) /  9020315/	! K(2)(2250)
      data itab2006(27) /  9020325/	! K(2)(2250)
      data itab2006(28) /   100445/	! chi(c2)(2P)



      save itab2004,itab2006

      cnv2004to2006=id
      ida=iabs(id)

      if(ida.eq.0)then
D        write(lnhout,1001) 
      elseif(mconv.eq.1)then
        do i = 1,mxlist
           if(ida.eq.itab2004(i))then
              cnv2004to2006=isign(itab2006(i),id)
           endif
        enddo
D        if(cnv2004to2006.eq.0 ) write(lnhout,1002) id
      elseif(mconv.eq.2)then
        do i = 1,mxlist
           if(ida.eq.itab2006(i))then
	      if( itab2004(i) .gt. 0 ) then
                 cnv2004to2006=isign(itab2004(i),id)
	      else
                 cnv2004to2006=0
	      endif
           endif
        enddo
C  check for R-Hadrons
	k100=mod(ida/1000000,1000)
	k1000=mod(ida/1000000,10000)
	k1000=mod(ida/1000000,100000)
	if(k100.eq.109)then
          cnv2004to2006=0
	elseif(k1000.eq.1009 .or. k1000.eq.1006)then
          cnv2004to2006=0
	elseif(k10000.eq.10009 .or. k10000.eq.10006)then
          cnv2004to2006=0
	endif
D        if(cnv2004to2006.eq.0 ) write(lnhout,1003) id
      endif
      return
 1001 format(' cnv2004to2006 1: particle id is zero')
 1002 format(' cnv2004to2006 4: ID ',i7,
     1            ' is not a defined PDG 2006 state')
 1003 format(' cnv2004to2006 3: ID ',i7,
     1            ' is not a defined PDG 2004 state')
      end
