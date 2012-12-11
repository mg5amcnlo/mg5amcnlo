      subroutine dumpheprup
      
#include "heprup.inc"
#include "stdlun.inc"

      integer iev, i, j
      
      write(lnhdmp,1001) 
      write(lnhdmp,1002) idbmup(1),idbmup(2),ebmup(1),ebmup(2),
     1                   pdfgup(1),pdfgup(2),pdfsup(1),pdfsup(2),
     2                   idwtup,nprup
      do i=1,nprup
          write(lnhdmp,1003) i, xsecup(i),xerrup(i),
     1                          xmaxup(i),lprup(i)
      enddo
      
      return
 1001 format('dump hepeup event ')
 1002 format(2i11,2e20.12,6i11)
 1003 format(i8,3e20.12,i11)
      end
