      integer function dpmtran(id,mconv)
C...convert (mconv=1) from DPM numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to DPM numbering scheme
C
C           ID = particle identification number

#include "stdlun.inc"

      dpmtran=id

      ida=iabs(id)
      j1=mod(ida,10)
      i1=mod(ida/10,10)
      i2=mod(ida/100,10)
      i3=mod(ida/1000,10)
      i4=mod(ida/10000,10)
      ksusy = mod(ida/1000000,10)
      ku = mod(ida/10000000,10)
      kqn=mod(ida/1000000000,10)

      if(ida.eq.0)then
        write(lnhout,*) ' DPMTRAN 1: particle ID is zero'
      elseif(mconv.LT.1 .OR. mconv.GT.2)then
        dpmtran=0
        write(lnhout,*) ' DPMTRAN 2: unallowed conversion option'
C...mesons 
      elseif(i1.ne.0 .and. i2.ne.0 .and. i3.eq.0)then
        if(mconv.eq.1)then
          if(ida.eq.30323) dpmtran=isign(100323,id)
          if(ida.eq.30313) dpmtran=isign(100313,id)
          if(ida.eq.20111) dpmtran=isign(100111,id)
          if(ida.eq.20211) dpmtran=isign(100211,id)
          if(ida.eq.20321) dpmtran=isign(100321,id)
          if(ida.eq.20311) dpmtran=isign(100311,id)
          if(ida.eq.10111) dpmtran=isign(9000111,id)
          if(ida.eq.10211) dpmtran=isign(9000211,id)
          if(ida.eq.10221) dpmtran=isign(9010221,id)
          if(ida.eq.10331) dpmtran=isign(10221,id)
          if(ida.eq.20333) dpmtran=isign(9000223,id)
	elseif(mconv.eq.2)then
          if(ida.eq.100323) dpmtran=isign(30323,id)
          if(ida.eq.100313) dpmtran=isign(30313,id)
          if(ida.eq.100111) dpmtran=isign(20111,id)
          if(ida.eq.100211) dpmtran=isign(20211,id)
          if(ida.eq.100321) dpmtran=isign(20321,id)
          if(ida.eq.100311) dpmtran=isign(20311,id)
          if(ida.eq.10111) dpmtran=0
          if(ida.eq.10211) dpmtran=0
          if(ida.eq.9000111) dpmtran=isign(10111,id)
          if(ida.eq.9000211) dpmtran=isign(20211,id)
          if(ida.eq.9010221) dpmtran=isign(10221,id)
          if(ida.eq.10221) dpmtran=isign(10331,id)
          if(ida.eq.20333) dpmtran=0
          if(ida.eq.9000223) dpmtran=isign(20333,id)
	endif
      endif

      return
      end
