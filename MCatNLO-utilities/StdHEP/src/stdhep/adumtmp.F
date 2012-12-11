      subroutine adumtmp

C...Purpose: The linker picks up stdtmp from the first available .o file,
C            which used to be stdtcopy and then became luntmp.
C            This dummy should now be the first available place to find
C            stdtmp.
C

#include "stdhep.inc"
#include "hepev4.inc"
#include "stdtmp.inc"
#include "stdlun.inc"

      write (lnhout,801)
  801 format(/5X,'ADUMTMP: This routine helps avoid link problems')
      return
      end
