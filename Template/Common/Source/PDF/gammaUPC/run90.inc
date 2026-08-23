!**************************************************************
!    run90.inc 
!**************************************************************
!
!     Collider
!
      integer                                        lpp(2)
      double precision    ebeamMG5(2), xbk(2),q2fact(2)
      common/to_collider/ ebeamMG5   , xbk   ,q2fact,   lpp


!
!     block for heavy ion beam
!
      integer nb_proton(2), nb_neutron(2)
      common/to_heavyion_pdg/ nb_proton, nb_neutron
