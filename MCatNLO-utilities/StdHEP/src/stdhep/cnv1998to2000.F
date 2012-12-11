      integer function cnv1998to2000(id,mconv)
C...convert (mconv=1) from PDG 1998 & StdHep 4.09 numbers
C                     to PDG 2000 & StdHep 4.10 numbers
C...     or (mconv=2) from PDG 2000 numbers to PDG 1998 numbers
C
C           id = particle identification number
C
#include "stdlun.inc"

      integer id,mconv

C... itab1998(I) converts all defined PDG 1998 particle ID's to the 2000 numbers
      parameter (mxlist = 18)
      integer itab1998(mxlist), itab2000(mxlist)

      data itab1998(1)  /          0/	! pi(1)(1600)0
      data itab1998(2)  /          0/	! pi(1)(1600)+
      data itab1998(3)  /          0/	! a(1)(1640)0
      data itab1998(4)  /          0/	! a(1)(1640)+
      data itab1998(5)  /    9010113/	! rho(2150)0
      data itab1998(6)  /    9010213/	! rho(2150)+
      data itab1998(7)  /          0/	! a(2)(1660)0
      data itab1998(8)  /          0/	! a(2)(1660)+
      data itab1998(9)  /          0/	! a(2)(1750)0
      data itab1998(10) /          0/	! a(2)(1750)+
      data itab1998(11) /    9000113/	! pi(2)(2100)0
      data itab1998(12) /    9000213/	! pi(2)(2100)+
      data itab1998(13) /    9030221/	! f(0)(1710)
      data itab1998(14) /    9040221/	! f(0)(2020)
      data itab1998(15) /    9050221/	! f(0)(2060)
      data itab1998(16) /    9060221/	! f(0)(2200)
      data itab1998(17) /    9070221/	! eta(2225)
      data itab1998(18) /          0/	! psi(3836)

      data itab2000(1)  /    9010113/	! pi(1)(1600)0
      data itab2000(2)  /    9010213/	! pi(1)(1600)+
      data itab2000(3)  /    9020113/	! a(1)(1640)0
      data itab2000(4)  /    9020213/	! a(1)(1640)+
      data itab2000(5)  /    9030113/	! rho(2150)0
      data itab2000(6)  /    9030213/	! rho(2150)+
      data itab2000(7)  /    9000115/	! a(2)(1660)0
      data itab2000(8)  /    9000215/	! a(2)(1660)+
      data itab2000(9)  /    9010115/	! a(2)(1750)0
      data itab2000(10) /    9010215/	! a(2)(1750)+
      data itab2000(11) /    9020115/	! pi(2)(2100)0
      data itab2000(12) /    9020215/	! pi(2)(2100)+
      data itab2000(13) /      10331/	! f(0)(1710)
      data itab2000(14) /    9030221/	! f(0)(2020)
      data itab2000(15) /    9040221/	! f(0)(2060)
      data itab2000(16) /    9050221/	! f(0)(2200)
      data itab2000(17) /    9060221/	! eta(2225)
      data itab2000(18) /    9000445/	! psi(3836)


      save itab1998,itab2000

      cnv1998to2000=id
      ida=iabs(id)

      if(ida.eq.0)then
D        write(lnhout,*) ' CNV1998TO2000 1: particle id is zero'
      elseif(mconv.eq.1)then
        do i = 1,mxlist
           if(ida.eq.itab1998(i))then
              cnv1998to2000=isign(itab2000(i),id)
           endif
        enddo
D        if(cnv1998to2000.eq.0 ) write(lnhout,1002) id
      elseif(mconv.eq.2)then
        do i = 1,mxlist
           if(ida.eq.itab2000(i))then
	      if( itab1998(i) .gt. 0 ) then
                 cnv1998to2000=isign(itab1998(i),id)
	      else
                 cnv1998to2000=0
	      endif
           endif
        enddo
D        if(cnv1998to2000.eq.0 ) write(lnhout,1001) id
      endif
      return
 1001 format(' CNV1998TO2000 3: ID ',i7,
     1            ' is not a defined PDG 1998 state')
 1002 format(' CNV1998TO2000 4: ID ',i7,
     1            ' is not a defined PDG 2000 state')
      end
