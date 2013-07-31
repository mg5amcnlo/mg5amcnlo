!====================================================================
!
!  Define common block with all general parameters used by MadFKS 
!  See their definitions in the file FKS_params.dat.
!
!====================================================================

      character*64 paramFileName
      parameter ( paramFileName='FKS_params.dat')

      real*8 IRPoleCheckThreshold,PrecisionVirtualAtRunTime

      integer NHelForMCoverHels

      common /FKSPARAMS/NHelForMCoverHels,IRPoleCheckThreshold,
     &	     PrecisionVirtualAtRunTime
