      subroutine MW_setrun
c----------------------------------------------------------------------
c     Sets the run parameters reading them from the MadWeight_card.dat and
c         from run_card.dat
c
c MadWeight_CARD
c 1. Run
c 2. permutations
c 3. likelihood parameter
c
c RUN_CARD
c 1. PDF set
c 2. Collider parameters
c 3. cuts
c---------------------------------------------------------------------- 
      implicit none

      include 'madweight_param.inc'

      call setrun
     
      include 'madweight_card.inc'
 

      return
      end
