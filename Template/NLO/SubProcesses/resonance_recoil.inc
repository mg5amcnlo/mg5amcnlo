c Radiation frame for the current FKS map. Physical momenta and flux
c always refer to the full process; only xi, y and the radiation measure
c use the rest frame of resonance_momentum when resonance_recoil is true.
      logical resonance_recoil,resonance_members(nexternal)
      double precision resonance_momentum(0:3),resonance_mass2
      common /c_resonance_recoil/ resonance_momentum,resonance_mass2,
     $     resonance_recoil,resonance_members
