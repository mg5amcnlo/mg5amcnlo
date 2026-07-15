C Copyright (C) 2020-2026 CERN and UCLouvain.
C Licensed under the GNU Lesser General Public License (version 3 or later).
C Created originally by: A. Valassi (Feb 2022) for the MG5aMC CUDACPP plugin.
C Further modified by: D. Massaro, A. Thete, A. Valassi (2022-2026).
C Integrated with the MadGraph7 project in Feb 2026.
C
C Create a Bridge and return its pointer
C - PBRIDGE: the memory address of the C++ Bridge
C - NEVT:    the number of events in the Fortran arrays
C - NPAR:    the number of external particles in the Fortran arrays (KEPT FOR SANITY CHECKS ONLY: remove it?)
C - NP4:     the number of momenta components, usually 4, in the Fortran arrays (KEPT FOR SANITY CHECKS ONLY: remove it?)
C
      INTERFACE
         SUBROUTINE FBRIDGECREATE(PBRIDGE, NEVT, NPAR, NP4)
         INTEGER*8 PBRIDGE
         INTEGER*4 NEVT
         INTEGER*4 NPAR
         INTEGER*4 NP4
         END SUBROUTINE FBRIDGECREATE
      END INTERFACE
      
C
C Delete a Bridge.
C - PBRIDGE: the memory address of the C++ Bridge
C
      INTERFACE
         SUBROUTINE FBRIDGEDELETE(PBRIDGE)
         INTEGER*8 PBRIDGE
         END SUBROUTINE FBRIDGEDELETE
      END INTERFACE
      
C
C Execute the matrix-element calculation "sequence" via a Bridge on GPU/CUDA or CUDA/C++.
C - PBRIDGE:   the memory address of the C++ Bridge
C - MOMENTA:   the input 4-momenta Fortran array
C - GS:        the input Gs (running QCD coupling constant alphas) Fortran array
C - IFLAV_VEC: the input array of flavor indices for the flavor combinations
C - RNDHEL:    the input random number Fortran array for helicity selection
C - RNDCOL:    the input random number Fortran array for color selection
C - CHANID:    the input array of channels (Feynman diagrams) to enhance
C - MES:       the output matrix element Fortran array
C - SELHEL:    the output selected helicity Fortran array
C - SELCOL:    the output selected color Fortran array
C - HELONLY:   input flag, quit after computing good helicities?
C
      INTERFACE
         SUBROUTINE FBRIDGESEQUENCE(PBRIDGE, MOMENTA, GS,
     &     IFLAV_VEC, RNDHEL, RNDCOL, CHANID, MES, SELHEL,
     &     SELCOL, HELONLY)
         INTEGER*8 PBRIDGE
         DOUBLE PRECISION MOMENTA(*)
         DOUBLE PRECISION GS(*)
         INTEGER*4 IFLAV_VEC(*)
         DOUBLE PRECISION RNDHEL(*)
         DOUBLE PRECISION RNDCOL(*)
         INTEGER*4 CHANID(*)
         DOUBLE PRECISION MES(*)
         INTEGER*4 SELHEL(*)
         INTEGER*4 SELCOL(*)
         LOGICAL HELONLY
         END SUBROUTINE FBRIDGESEQUENCE
      END INTERFACE

C
C Execute the matrix-element calculation "sequence" via a Bridge on GPU/CUDA or CUDA/C++.
C - PBRIDGE:   the memory address of the C++ Bridge
C - MOMENTA:   the input 4-momenta Fortran array
C - GS:        the input Gs (running QCD coupling constant alphas) Fortran array
C - IFLAV_VEC: the input array of flavor indices for the flavor combinations
C - RNDHEL:    the input random number Fortran array for helicity selection
C - RNDCOL:    the input random number Fortran array for color selection
C - MES:       the output matrix element Fortran array
C - SELHEL:    the output selected helicity Fortran array
C - SELCOL:    the output selected color Fortran array
C - HELONLY:   input flag, quit after computing good helicities?
C
      INTERFACE
         SUBROUTINE FBRIDGESEQUENCE_NOMULTICHANNEL(PBRIDGE, MOMENTA, GS,
     &     IFLAV_VEC, RNDHEL, RNDCOL, MES, SELHEL, SELCOL, HELONLY)
         INTEGER*8 PBRIDGE
         DOUBLE PRECISION MOMENTA(*)
         DOUBLE PRECISION GS(*)
         INTEGER*4 IFLAV_VEC(*)
         DOUBLE PRECISION RNDHEL(*)
         DOUBLE PRECISION RNDCOL(*)
         DOUBLE PRECISION MES(*)
         INTEGER*4 SELHEL(*)
         INTEGER*4 SELCOL(*)
         LOGICAL HELONLY
         END SUBROUTINE FBRIDGESEQUENCE_NOMULTICHANNEL
      END INTERFACE

C
C Retrieve the number of good helicities for helicity filtering in the Bridge.
C - PBRIDGE:  the memory address of the C++ Bridge
C - NGOODHEL: the output number of good helicities
C - NTOTHEL:  the output total number of helicities in cudacpp (aka NCOMB in Fortran)
C
      INTERFACE
         SUBROUTINE FBRIDGEGETNGOODHEL(PBRIDGE, NGOODHEL, NTOTHEL)
         INTEGER*8 PBRIDGE
         INTEGER*4 NGOODHEL
         INTEGER*4 NTOTHEL
         END SUBROUTINE FBRIDGEGETNGOODHEL
      END INTERFACE
