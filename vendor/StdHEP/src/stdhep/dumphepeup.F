      subroutine dumphepeup(iev)
      
#include "hepeup.inc"
#include "stdlun.inc"

      integer iev, i, j
      
      write(lnhdmp,1001) iev
      write(lnhdmp,1002) nup, idprup, xwgtup, scalup, aqedup, aqcdup
      do i=1,nup
          write(lnhdmp,1003) i, idup(i), istup(i),
     1                    mothup(1,i),mothup(2,i),
     2	                  icolup(1,i), icolup(2,i), 
     3                    (pup(j,i),j=1,5), vtimup(i), spinup(i)
      enddo
      
      return
 1001 format('dump hepeup event ',i8)
 1002 format(2i11,4e20.12)
 1003 format(i8,6i11,7e20.12)
      end
