      subroutine fill_MC_mshell()
      implicit none
      integer i
c Monte Carlo masses: use PDG conventions.
c May be given in input eventually
      double precision mcmass(-16:21)
      common/cmcmass/mcmass
c
      character*10 MonteCarlo
      common/cMonteCarloType/MonteCarlo
c
      do i=-16,21
        mcmass(i)=-1.d10
      enddo
      if (MonteCarlo(1:7).eq.'HERWIG6') then
         include "MCmasses_HERWIG6.inc"
      elseif(MonteCarlo(1:8).eq.'HERWIGPP')then
         include "MCmasses_HERWIGPP.inc"
      elseif(MonteCarlo(1:8).eq.'PYTHIA6Q')then
         include "MCmasses_PYTHIA6Q.inc"
      elseif(MonteCarlo(1:9).eq.'PYTHIA6PT')then
         include "MCmasses_PYTHIA6PT.inc"
      elseif(MonteCarlo(1:7).eq.'PYTHIA8')then
         include "MCmasses_PYTHIA8.inc"
      else
         write (*,*) 'Wrong MC ', MonteCarlo, ' in fill_MC_mshell'
         stop
      endif
      do i=-5,-1
         mcmass(i)=mcmass(-i)
      enddo
      do i=-16,-11
         mcmass(i)=mcmass(-i)
      enddo
      return      
      end
