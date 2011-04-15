      subroutine sborn_sf(p_born,m,n,wgt)
      implicit none
      include "nexternal.inc"
      double precision p_born(0:3,nexternal-1),wgt
      double complex wgt1(2)
      integer m,n
 
      integer i,j
      integer ii( %(nborns)d )
      data (ii(i), i=1, %(nborns)d  ) /   %(iborns)s  /                                                 
      integer ii_fks( %(nborns)d )
      data (ii_fks(i), i=1, %(nborns)d ) /   %(ifkss)s  /                                                 
      integer mm( %(nborns)d )
      data (mm(i), i=1, %(nborns)d ) /   %(mms)s  /                                                 
      integer nn( %(nborns)d )
      data (nn(i), i=1, %(nborns)d ) /   %(nns)s  /                                                 
 
      integer i_fks,j_fks
      common/fks_indices/i_fks,j_fks
 
%(iflines)s
         wgt=0d0
c         write (*,*)
c     &      "No corresponding color linked Born found in sborn_sf"
      endif
 
      return
      end
