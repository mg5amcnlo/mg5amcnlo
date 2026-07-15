C Copyright (C) 2020-2026 CERN and UCLouvain.
C Licensed under the GNU Lesser General Public License (version 3 or later).
C Created originally by: A. Valassi (Feb 2022) for the MG5aMC CUDACPP plugin.
C Further modified by: A. Valassi (2022-2024).
C Integrated with the MadGraph7 project in Feb 2026.
C
C Create a Sampler and return its pointer
C - PSAMPLER:  the memory address of the C++ Sampler
C - NEVT:      the number of events in the Fortran arrays
C - NPAR:      the number of external particles in the Fortran arrays (KEPT FOR SANITY CHECKS ONLY: remove it?)
C - NP4:       the number of momenta components, usually 4, in the Fortran arrays (KEPT FOR SANITY CHECKS ONLY: remove it?)
C
      INTERFACE
         SUBROUTINE FSAMPLERCREATE(PSAMPLER, NEVT, NPAR, NP4)
         INTEGER*8 PSAMPLER
         INTEGER*4 NEVT
         INTEGER*4 NPAR
         INTEGER*4 NP4
         END SUBROUTINE FSAMPLERCREATE
      END INTERFACE
      
C
C Delete a Sampler.
C - PSAMPLER:  the memory address of the C++ Sampler
C
      INTERFACE
         SUBROUTINE FSAMPLERDELETE(PSAMPLER)
         INTEGER*8 PSAMPLER
         END SUBROUTINE FSAMPLERDELETE
      END INTERFACE
      
C
C Execute the matrix-element calculation "sequence" via a Sampler on GPU/CUDA or CUDA/C++.
C - PSAMPLER:  the memory address of the C++ Sampler
C - MOMENTA:   the output 4-momenta Fortran array
C
      INTERFACE
         SUBROUTINE FSAMPLERSEQUENCE(PSAMPLER, MOMENTA)
         INTEGER*8 PSAMPLER
         DOUBLE PRECISION MOMENTA(*)
         END SUBROUTINE FSAMPLERSEQUENCE
      END INTERFACE
