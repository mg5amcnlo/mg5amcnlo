      subroutine stdxend(istream)
c
c  end xdr tape writing
c
      implicit none
#include "mcfio.inc"
#include "stdlun.inc"
      integer istream,ieff,inum
c
      call mcfio_InfoStreamInt(istream, MCFIO_NUMWORDS, inum)
      call mcfio_InfoStreamInt(istream, MCFIO_EFFICIENCY, ieff)
      call mcfio_close(istream)
c$$$ THIS HAS BEEN CHANGED BY PT
c$$$      write(lnhout,1001) inum,ieff
      write(99,1001) inum,ieff
      return
1001  format(/10x,'STDXEND: ',i10,' words i/o with ',i8,' efficiency ')
      end
