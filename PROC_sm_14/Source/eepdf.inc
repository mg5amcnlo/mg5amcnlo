      ! Some stuff relevant for the dressed-lepton luminosity
      !
      ! the number of components
      integer n_ee
      parameter (n_ee = 4)
      ! arrays to store the components before combining them
      ! note this common is very quickly overwritten do not use
      ! use such common outside of auto_dsig.f
      double precision ee_components(n_ee)
      common / to_ee_components / ee_components


