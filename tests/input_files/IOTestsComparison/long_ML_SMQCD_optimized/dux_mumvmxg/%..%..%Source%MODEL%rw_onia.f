c************************************************************************
c**                                                                    **
c**           MadGraph/MadEvent Interface to FeynRules                 **
c**                                                                    **
c**          C. Duhr (Louvain U.) - M. Herquet (NIKHEF)                **
c**                                                                    **
c************************************************************************

      subroutine setonia(onia_name)
      implicit none

      character*(*) onia_name
      logical readlha

      include 'ldme.inc'

      integer maxpara
      parameter (maxpara=5000)
      
      integer nonia
      character*20 onia(maxpara),value(maxpara)

      call LHA_loadcard(onia_name,nonia,onia,value)

      include 'onia_read.inc'

      return

      end
