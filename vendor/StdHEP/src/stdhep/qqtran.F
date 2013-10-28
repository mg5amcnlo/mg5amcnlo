      integer function qqtran(ID,mconv)
C...convert (mconv=1) from QQ numbering scheme to PDG numbering scheme
C...     or (mconv=2) from PDG numbering scheme to QQ numbering scheme
C
C           ID = particle identification number

#include "stdlun.inc"
#include "qqpars.inc"
#include "qqprop.inc"

      integer I,IT,ID,mconv

      if(mconv.EQ.1)then
C...convert from QQ numbering scheme to PDG numbering scheme
        qqtran=IDPDGQ(ID)
        if(qqtran.EQ.0)
     1     write(lnhout,*) ' QQTRAN: particle ID translation is zero'
      elseif(mconv.EQ.2)then
C...convert from PDG numbering scheme to QQ numbering scheme
        qqtran = -21
        do IT=NPMNQQ,NPMXQQ
          if(ID.EQ.IDPDGQ(IT)) qqtran = IT
        enddo
        if(ID.EQ.0 .or. qqtran.eq.-21 )then
          qqtran = -21 
          write(lnhout,*) ' QQTRAN: particle ID translation is invalid'
        endif
      else
C...invalid option
        qqtran=0
        write(lnhout,*) ' QQTRAN: unallowed conversion option'
      endif
      return
      end
