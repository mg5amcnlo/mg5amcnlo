      real function stdchg(id)

C...Purpose: to give the charge for a particle/parton.

C      id     = particle ID
C      stdchg = particle charge (real number)

      integer hepchg
      external hepchg

      integer id,ich

      ich = hepchg(id)
      stdchg = float(ich)/3.
      
      return
      end
