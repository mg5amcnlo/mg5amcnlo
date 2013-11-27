!====================================================================
!
!  Define common block with all general parameters used by MadFKS 
!  See their definitions in the file FKS_params.dat.
!
!====================================================================

      character*64 paramFileName
      parameter ( paramFileName='FKS_params.dat')

      real*8 IRPoleCheckThreshold,Virt_fraction,
     &	     PrecisionVirtualAtRunTime,Min_virt_fraction

      integer NHelForMCoverHels

      common /FKSPARAMS/IRPoleCheckThreshold,virt_fraction,
     &       PrecisionVirtualAtRunTime,min_virt_fraction,
     &       NHelForMCoverHels
