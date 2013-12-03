c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

      program helas_couplings
      
      include 'coupl.inc'
      include 'formats.inc'
      
      call setpara('param_card.dat',.true.)

      open(unit=1,file="helas_couplings.txt")
      include "helas_couplings.inc"
      
      close(1)
      
      END

