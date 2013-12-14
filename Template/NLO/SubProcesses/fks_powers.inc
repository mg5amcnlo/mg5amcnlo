      double precision a_f_damp, c_f_damp, one_f_damp
      parameter (a_f_damp=1d0)
      parameter (c_f_damp=1d0)
      parameter (one_f_damp=2d0)

      double precision fks_a, fks_b
      parameter (fks_a=1.5d0)
      parameter (fks_b=1.5d0)

      double precision a_h_damp, one_h_damp
      parameter (a_h_damp=1d0)
      parameter (one_h_damp=1.d-2)

      logical useenergy
      parameter (useenergy=.true.)

      logical usebeta
      parameter (usebeta=.true.)

      double precision deltaO, deltaI, xicut
      parameter (deltaO=1.0d0)
      parameter (deltaI=1.0d0)
      parameter (xicut=0.5d0)

      double precision deltaS, xiScut, xiBSVcut
      parameter (deltaS=1.0d0)
      parameter (xiScut=0.5d0)
      parameter (xiBSVcut=1.0d0)

      double precision deltaminY,skewY,alphaY
      parameter (alphaY=2.0d0)
      parameter (skewY=10d0)
      parameter (deltaminY=0.95d0)

