c*************************************************************************
c     Phase-space generated invariant masses, used to feed the exact
c     (squared) invariant mass of two final-state particles into the
c     propagator denominator of the offshell HELAS currents (FIXP2).
c
c     GENERATED_INV_MASS(i,j,ivec) holds the invariant mass SQUARED of the
c     s-channel pseudo-particle that is the fusion of the two final-state
c     external particles i and j, for the phase-space point stored in the
c     vector slot ivec. It is zero whenever the generating channel does
c     not build such a (exactly two final-state particle) s-channel node,
c     in which case the HELAS routine falls back to recomputing p^2 from
c     the momenta.
c
c     GENERATED_INV_MASS_IVEC is the vector slot currently being filled by
c     the phase-space generator (set in dsample.f before x_to_f_arg).
c
c     Requires nexternal.inc (NEXTERNAL) and vector.inc (VECSIZE_MEMMAX)
c     to be included beforehand.
c*************************************************************************
      integer generated_inv_mass_ivec
      common/to_inv_mass_ivec/generated_inv_mass_ivec
      double precision generated_inv_mass(nexternal,nexternal,
     &     VECSIZE_MEMMAX)
      common/to_generated_inv_mass/generated_inv_mass
