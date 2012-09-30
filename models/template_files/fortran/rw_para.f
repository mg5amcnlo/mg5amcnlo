c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

      subroutine setpara(param_name)
      implicit none

      character*(*) param_name
      logical readlha

      %(includes)s

      integer maxpara
      parameter (maxpara=5000)
      
      integer npara
      character*20 param(maxpara),value(maxpara)

      %(load_card)s
      include 'param_read.inc'
      call coup()

      return

      end



