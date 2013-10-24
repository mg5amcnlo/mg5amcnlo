      subroutine pythia2ev4
c
c  fill hepev4 (Les Houches block) with info from pythia
c

#include "stdhep.inc"
#include "hepev4.inc"
#include "pypars.inc"
#include "pyint1.inc"
#include "pyint5.inc"

      integer i
      logical lfirst
      data lfirst/.TRUE./
      save lfirst

C...print version number if this is the first call
      if(lfirst)then
        call stdversn
        if(mconv.EQ.1) NEVHEP = 0
        lfirst=.FALSE.
      endif

c
C.....running estimate of total cross section ... normalize by 
C......number of generated events?
C      eventweightlh = pari(1)/ngen(0,3)
      eventweightlh = xsec(idruplh,3)
C.....values from the hard scattering
      alphaqedlh = vint(57)
      alphaqcdlh = vint(58)
      idruplh = mint(1)
C.....NOTE... all of these have units GeV**2, not GeV.  Document this.
C.....parton showering scale
      scalelh(1) = vint(56)
C.....s-hat
      scalelh(2)=vint(44)
C.....t-hat
      scalelh(3)=vint(45)
C.....u-hat
      scalelh(4)=vint(46)
C.....pt2
      scalelh(5)=vint(48)

      do i=1,nhep
          icolorflowlh(1,i) = 0
          icolorflowlh(2,i) = 0
	  spinlh(1,i) = 9
	  spinlh(2,i) = 9
	  spinlh(3,i) = 9
      enddo

      return
      end



