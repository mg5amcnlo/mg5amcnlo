!====================================================================
!
!  Define common block with all general parameters used by MadFKS 
!  See their definitions in the file FKS_params.dat.
!
!====================================================================

      character*64 paramFileName
      parameter ( paramFileName='FKS_params.dat')

	  integer maxContribsSelected, maxCouplingsSelected
	  parameter( maxContribsSelected=100, maxCouplingsSelected=100 )
	  integer maxContribType, maxCouplingTypes
	  parameter ( maxContribType=15, maxCouplingTypes=20 )

      real*8 IRPoleCheckThreshold,Virt_fraction,
     &	     PrecisionVirtualAtRunTime,Min_virt_fraction

      integer NHelForMCoverHels, VetoedContributionTypes(0:maxContribsSelected),
     &        SelectedContributionTypes(0:maxContribsSelected),
     &        QED_squared_selected, QCD_squared_selected,
     &        SelectedCouplingOrders(maxCouplingTypes,0:maxCouplingsSelected)

      logical separate_flavour_configs, IncludeBornContributions

      common /FKSPARAMS/IRPoleCheckThreshold,virt_fraction,
     &       PrecisionVirtualAtRunTime,min_virt_fraction,
     &       NHelForMCoverHels,separate_flavour_configs,
     &       QCD_squared_selected, QED_squared_selected,
     &       VetoedContributionTypes, SelectedContributionTypes,
     &       SelectedCouplingOrders
