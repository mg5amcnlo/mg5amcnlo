c Dummy analysis routines used for linking when not doing Fixed Order
c calculations or when having fo_analysis_format=none in the
c FO_analyse_card.
      subroutine analysis_begin(nwgt,weights_info)
      implicit none
      integer nwgt
      character*(*) weights_info(*)
      call inihist
      end
      subroutine analysis_end(xnorm)
      implicit none
      double precision xnorm
      end
      subroutine analysis_fill(p,istatus,ipdg,wgts,ibody)
      implicit none
      include 'nexternal.inc'
      integer istatus(nexternal)
      integer iPDG(nexternal)
      double precision p(0:4,nexternal)
      double precision wgts(*)
      integer ibody
      end
