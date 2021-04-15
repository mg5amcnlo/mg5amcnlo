!====================================================================
!
!  Define common block with all general parameters used by MadFKS 
!  See their definitions in the file FKS_params.dat.
!
!====================================================================
module FKSParams
  character(len=64), parameter ::  paramFileName='FKS_params.dat'
  integer,parameter :: maxContribsSelected=100, &
                       maxCouplingsSelected=100, &
                       maxContribType=15, &
                       maxCouplingTypes=20
  real*8 :: IRPoleCheckThreshold,Virt_fraction, PrecisionVirtualAtRunTime,Min_virt_fraction
  integer  :: NHelForMCoverHels,VetoedContributionTypes(0:maxContribsSelected), &
              SelectedContributionTypes(0:maxContribsSelected),QED_squared_selected, &
              SelectedCouplingOrders(maxCouplingTypes,0:maxCouplingsSelected), &
              QCD_squared_selected
  logical :: separate_flavour_configs,IncludeBornContributions,use_poly_virtual

contains

  subroutine FKSParamReader(filename, printParam, force)
    ! Reads the file 'filename' and sets the parameters found in that file.
    implicit none
    logical, save :: HasReadOnce=.False.,paramPrinted=.false.
    logical :: force,couldRead,printParam
    character(*) :: filename
    CHARACTER(len=64) :: buff, buff2, mode
    include "orders.inc"
    integer :: i,j
    couldRead=.False.
    if (HasReadOnce.and..not.force) then
       goto 901
    endif
! Make sure to have default parameters if not set in the FKSParams.dat card
! (if it is an old one for instance)
    call DefaultFKSParam()
! Overwrite the default parameters from file:
    open(68, file=fileName, err=676, action='READ')
    do
       read(68,*,end=999) buff
       if(index(buff,'#').eq.1) then
          if (buff .eq. '#IRPoleCheckThreshold') then
             read(68,*,end=999) IRPoleCheckThreshold
             if (IRPoleCheckThreshold .lt. -1.01d0 ) then
                stop 'IRPoleCheckThreshold must be >= -1.0d0.'
             endif
          elseif (buff .eq. '#PrecisionVirtualAtRunTime') then
             read(68,*,end=999) PrecisionVirtualAtRunTime
             if (IRPoleCheckThreshold .lt. -1.01d0 ) then
                stop 'PrecisionVirtualAtRunTime must be >= -1.0d0.'
             endif
          else if (buff .eq. '#NHelForMCoverHels') then
             read(68,*,end=999) NHelForMCoverHels
             if (NHelForMCoverHels .lt. -1) then
                stop 'NHelForMCoverHels must be >= -1.'
             endif
          else if (buff .eq. '#QCD^2==') then
             read(68,*,end=999) QCD_squared_selected
             if (QCD_squared_selected .lt. -1) then
                stop 'QCD_squared_selected must be >= -1.'
             endif
          else if (buff .eq. '#QED^2==') then
             read(68,*,end=999) QED_squared_selected
             if (QED_squared_selected .lt. -1) then
                stop 'QED_squared_selected must be >= -1.'
             endif
          else if (buff .eq. '#VirtualFraction') then
             read(68,*,end=999) Virt_fraction
             if (Virt_fraction .lt. 0 .or. virt_fraction .gt.1) then
                stop 'VirtualFraction should be a fraction between 0 and 1'
             endif
          else if (buff .eq. '#MinVirtualFraction') then
             read(68,*,end=999) Min_Virt_fraction
             if (min_virt_fraction .lt. 0 .or. min_virt_fraction .gt.1) then
                stop 'VirtualFraction should be a fraction between 0 and 1'
             endif
          else if (buff .eq. '#SeparateFlavourConfigurations') then
             read(68,*,end=999) separate_flavour_configs
          else if (buff .eq. '#UsePolyVirtual') then
             read(68,*,end=999) use_poly_virtual
          else if (buff .eq. '#VetoedContributionTypes') then
             read(68,*,end=999) VetoedContributionTypes(0)
             if (VetoedContributionTypes(0) .lt. 0.or. &
                  VetoedContributionTypes(0) .gt. maxContribsSelected) then
                write(*,*) 'VetoedContributionTypes length should be >= 0 and <=', &
                     maxContribsSelected
                stop 'Format error in FKS_params.dat.'
             endif
             read(68,*,end=999) (VetoedContributionTypes(I),I=1,VetoedContributionTypes(0))
             do I=1,VetoedContributionTypes(0)
                if (VetoedContributionTypes(I).lt.1.or. &
                     VetoedContributionTypes(I).gt.maxContribType) then
                   write(*,*) 'VetoedContributionTypes must be >=1 and <=',maxContribType
                   stop 'Format error in FKS_params.dat.'
                endif
             enddo
             do I=VetoedContributionTypes(0)+1,maxContribsSelected
                VetoedContributionTypes(I)=-1
             enddo

          else if (buff .eq. '#SelectedContributionTypes') then
             read(68,*,end=999) SelectedContributionTypes(0)
             if (SelectedContributionTypes(0) .lt. 0 .or. &
                  SelectedContributionTypes(0) .gt. maxContribsSelected) then
                write(*,*) 'SelectedContributionTypes length should be >= 0 and <=', &
                     maxContribsSelected
                stop 'Format error in FKS_params.dat.'
             endif
             read(68,*,end=999) (SelectedContributionTypes(I),I=1,SelectedContributionTypes(0))
             do I=1,SelectedContributionTypes(0)
                if (SelectedContributionTypes(I).lt.1.or. &
                     SelectedContributionTypes(I).gt.maxContribType) then
                   write(*,*) 'SelectedContributionTypes must be >=1 and <=',maxContribType
                   stop 'Format error in FKS_params.dat.'
                endif
             enddo
             do I=SelectedContributionTypes(0)+1,maxContribsSelected
                SelectedContributionTypes(I)=-1
             enddo
          else if (buff .eq. '#SelectedCouplingOrders') then
             read(68,*,end=999) SelectedCouplingOrders(1,0)
             if (SelectedCouplingOrders(1,0) .lt. 0 .or. &
                  SelectedCouplingOrders(1,0) .gt. maxCouplingsSelected) then
                write(*,*) 'SelectedCouplingOrders length should be >= 0 and <=', &
                     maxCouplingsSelected
                stop 'Format error in FKS_params.dat.'                
             endif
             do j = 2, maxCouplingTypes
                SelectedCouplingOrders(j,0) = SelectedCouplingOrders(1,0)
             enddo
             do j = 1, SelectedCouplingOrders(1,0)
                read(68,*,end=999) (SelectedCouplingOrders(i,j),i=1,nsplitorders)
                do i=nsplitorders+1,maxCouplingTypes
                   SelectedCouplingOrders(i,j)=-1
                enddo
             enddo
          else
             write(*,*) 'The parameter name ',buff(2:),'is not reckognized.'
             stop 'Format error in FKS_params.dat.'
          endif
       endif
    enddo
999 continue
    couldRead=.True.
    goto 998      

676 continue
    write(*,*) 'ERROR :: MadFKS parameter file ',fileName, &
         ' could not be found or is malformed. Please specify it.'
    stop 
    !   Below is the code if one desires to let the code continue with
    !   a non existing or malformed parameter file
    write(*,*) 'WARNING :: The file ',fileName,' could not be ', &
         ' open or did not contain the necessary information. The ', &
         ' default MadFKS parameters will be used.'
    call DefaultFKSParam()
    goto 998
998 continue

    if(printParam.and..not.paramPrinted) then
       write(*,*) &
            '==============================================================='
       if (couldRead) then      
          write(*,*) 'INFO: MadFKS read these parameters from ',filename
       else
          write(*,*) 'INFO: MadFKS used the default parameters.'
       endif
       write(*,*) &
            '==============================================================='
       write(*,*) ' > IRPoleCheckThreshold      = ',IRPoleCheckThreshold
       write(*,*) ' > PrecisionVirtualAtRunTime = ',PrecisionVirtualAtRunTime
       if (SelectedContributionTypes(0).gt.0) then
          write(*,*) ' > SelectedContributionTypes = ', &
               (SelectedContributionTypes(I),I=1,SelectedContributionTypes(0))
       else
          write(*,*) ' > SelectedContributionTypes = All'
       endif
       if (VetoedContributionTypes(0).gt.0) then
          write(*,*) ' > VetoedContributionTypes   = ', &
               (VetoedContributionTypes(I),I=1,VetoedContributionTypes(0))
       else
          write(*,*) ' > VetoedContributionTypes   = None'
       endif
       if (QCD_squared_selected.eq.-1) then
          write(*,*) ' > QCD_squared_selected      = All'
       else
          write(*,*) ' > QCD_squared_selected      = ',QCD_squared_selected
       endif
       if (QED_squared_selected.eq.-1) then
          write(*,*) ' > QED_squared_selected      = All'
       else
          write(*,*) ' > QED_squared_selected      = ',QED_squared_selected
       endif
       if (SelectedCouplingOrders(1,0).gt.0) then
          do j=1,SelectedCouplingOrders(1,0)
             write(*,*) ' > SelectedCouplingOrders(',j,') = ', &
                  (SelectedCouplingOrders(i,j),i=1,nsplitorders)
          enddo
       else
          write(*,*) ' > SelectedCouplingOrders    = All'
       endif
       write(*,*) ' > NHelForMCoverHels         = ',NHelForMCoverHels
       write(*,*) ' > VirtualFraction           = ',Virt_fraction
       write(*,*) ' > MinVirtualFraction        = ',Min_virt_fraction
       write(*,*) ' > SeparateFlavourConfigs    = ',separate_flavour_configs
       write(*,*) ' > UsePolyVirtual            = ',use_poly_virtual
       write(*,*) &
            '==============================================================='
       paramPrinted=.TRUE.
    endif

    close(68)
    HasReadOnce=.TRUE.
901 continue
  end subroutine FKSParamReader

  subroutine DefaultFKSParam() 
    ! Sets the default parameters
    implicit none
    integer i,j
    IRPoleCheckThreshold=1.0d-5
    NHelForMCoverHels=5
    PrecisionVirtualAtRunTime=1d-3
    Virt_fraction=1d0
    QED_squared_selected=-1
    QCD_squared_selected=-1
    Min_virt_fraction=0.005d0
    separate_flavour_configs=.false.
    use_poly_virtual=.true.
    IncludeBornContributions=.true.
    SelectedContributionTypes(0)=0
    VetoedContributionTypes(0)=0
    do i=1, maxContribsSelected
       SelectedContributionTypes(I)=-1
       VetoedContributionTypes(I)=-1
    enddo
    do j=1,maxCouplingTypes
       SelectedCouplingOrders(j,0) = 0
    enddo
    do j=1,maxCouplingsSelected
       do i=1,maxCouplingTypes
          SelectedCouplingOrders(i,j) = -1
       enddo
    enddo
  end subroutine DefaultFKSParam

end module FKSParams
