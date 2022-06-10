!**************************************************************
!    run90.inc 
!**************************************************************
!
!     Collider
!
      integer                                        lpp(2)
      double precision    ebeam(2), xbk(2),q2fact(2)
      common/to_collider/ ebeam   , xbk   ,q2fact,   lpp

      integer nuclearA_beam1, nuclearA_beam2, nuclearZ_beam1, nuclearZ_beam2
      common/ion_beam/nuclearA_beam1,nuclearA_beam2, nuclearZ_beam1,nuclearZ_beam2