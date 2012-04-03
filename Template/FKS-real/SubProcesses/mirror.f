c This file contains subroutines and functions related to the process
c  mirroring in MadFKS. Written by MZ

      subroutine get_mirror_rescale(rescale)
      implicit none
      double precision rescale, sumprob, selproc(2) 
      double precision dlum
      integer imirror, imirror_bak
      common/cmirror/imirror
      include "mirrorprocs.inc"
      imirror_bak=imirror
      sumprob=0d0
      do imirror=1,2
        if (imirror.eq.1.or.mirrorproc) then
          selproc(imirror)=dlum()
          sumprob=sumprob+selproc(imirror)
        endif
      enddo
      imirror=imirror_bak
      if (selproc(imirror).gt.1d-8) then
        rescale=sumprob/selproc(imirror)
      else
        rescale=0d0
      endif
      return
      end

