c Radiation frame for the current FKS map. Physical momenta and flux
c always refer to the full process; only xi, y and the radiation measure
c use the rest frame of resonance_momentum when resonance_recoil is true.
c For FI recoil this is the emitter plus the unused beam momentum;
c initial_recoil_leg identifies that beam and the descendant mask is empty.
      logical resonance_recoil,resonance_members(nexternal)
      double precision resonance_momentum(0:3),resonance_mass2
      common /c_resonance_recoil/ resonance_momentum,resonance_mass2,
     $     resonance_recoil,resonance_members
      integer initial_recoil_leg
      common /c_initial_recoil/ initial_recoil_leg
