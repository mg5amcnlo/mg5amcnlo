      subroutine FKSParamReader(filename, printParam, force) 

      implicit none

      logical HasReadOnce, force
      data HasReadOnce/.FALSE./


      character(*) filename
      CHARACTER*64 buff, buff2, mode
      include "FKSParams.inc"
      include "orders.inc"
      
      integer i,j

      logical printParam, couldRead, paramPrinted
      data paramPrinted/.FALSE./
      couldRead=.False.
!     Default parameters

      if (HasReadOnce.and..not.force) then
          goto 901
      endif

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
                stop 'VirtualFraction should be a fraction'/
     $               /' between 0 and 1'
             endif 
           else if (buff .eq. '#MinVirtualFraction') then
             read(68,*,end=999) Min_Virt_fraction
             if (min_virt_fraction .lt. 0 .or. min_virt_fraction .gt.1)
     $            then
                stop 'VirtualFraction should be a fraction'/
     $               /' between 0 and 1'
             endif 
           else if (buff .eq. '#SeparateFlavourConfigurations') then
             read(68,*,end=999) separate_flavour_configs

           else if (buff .eq. '#VetoedContributionTypes') then
             read(68,*,end=999) VetoedContributionTypes(0)
             if (VetoedContributionTypes(0) .lt. 0.or.
     &           VetoedContributionTypes(0) .gt. maxContribsSelected) then
                write(*,*) 'VetoedContributionTypes length should be '/
     &               /'>= 0 and <=',maxContribsSelected
                stop 'Format error in FKS_params.dat.'
             endif
             read(68,*,end=999) (VetoedContributionTypes(I),I=1,
     $          VetoedContributionTypes(0))
             do I=1,VetoedContributionTypes(0)
               if (VetoedContributionTypes(I).lt.1.or.
     $             VetoedContributionTypes(I).gt.maxContribType) then
                write(*,*) 'VetoedContributionTypes must be >=1 and '/
     &               /'<=',maxContribType
                stop 'Format error in FKS_params.dat.'
               endif
             enddo
             do I=VetoedContributionTypes(0)+1,maxContribsSelected
               VetoedContributionTypes(I)=-1
             enddo

           else if (buff .eq. '#SelectedContributionTypes') then
             read(68,*,end=999) SelectedContributionTypes(0)
             if (SelectedContributionTypes(0) .lt. 0 .or.
     &           SelectedContributionTypes(0) .gt. maxContribsSelected) then
                write(*,*) 'SelectedContributionTypes length should be '/
     &               /'>= 0 and <=',maxContribsSelected
                stop 'Format error in FKS_params.dat.'
             endif
             read(68,*,end=999) (SelectedContributionTypes(I),I=1,
     $          SelectedContributionTypes(0))
             do I=1,SelectedContributionTypes(0)
               if (SelectedContributionTypes(I).lt.1.or.
     $             SelectedContributionTypes(I).gt.maxContribType) then
                write(*,*) 'SelectedContributionTypes must be >=1 and '/
     &               /'<=',maxContribType
                stop 'Format error in FKS_params.dat.'
               endif
             enddo
             do I=SelectedContributionTypes(0)+1,maxContribsSelected
               SelectedContributionTypes(I)=-1
             enddo

           else if (buff .eq. '#SelectedCouplingOrders') then
             read(68,*,end=999) SelectedCouplingOrders(1,0)
             if (SelectedCouplingOrders(1,0) .lt. 0 .or.
     &           SelectedCouplingOrders(1,0) .gt. maxCouplingsSelected) then
                write(*,*) 'SelectedCouplingOrders length should be >='/
     &               /' 0 and <=',maxCouplingsSelected
                stop 'Format error in FKS_params.dat.'                
             endif
             do j = 2, maxCouplingTypes
               SelectedCouplingOrders(j,0) = SelectedCouplingOrders(1,0)
             enddo
             do j = 1, SelectedCouplingOrders(1,0)
               read(68,*,end=999) (SelectedCouplingOrders(i,j),i=1,
     $            nsplitorders)
               do i=nsplitorders+1,maxCouplingTypes
                 SelectedCouplingOrders(i,j)=-1
               enddo
             enddo

           else
             write(*,*) 'The parameter name ',buff(2:),
     &' is not reckognized.'
             stop 'Format error in FKS_params.dat.'
           endif
         endif
      enddo
  999 continue
      couldRead=.True.
      goto 998      

  676 continue
      write(*,*) 'ERROR :: MadFKS parameter file ',fileName,
     &' could not be found or is malformed. Please specify it.'
      stop 
C     Below is the code if one desires to let the code continue with
C     a non existing or malformed parameter file
      write(*,*) 'WARNING :: The file ',fileName,' could not be ',
     & ' open or did not contain the necessary information. The ',
     & ' default MadFKS parameters will be used.'
      call DefaultFKSParam()
      goto 998

  998 continue

      if(printParam.and..not.paramPrinted) then
      write(*,*)
     & '==============================================================='
      if (couldRead) then      
        write(*,*) 'INFO: MadFKS read these parameters from '
     &,filename
      else
        write(*,*) 'INFO: MadFKS used the default parameters.'
      endif
      write(*,*)
     & '==============================================================='
      write(*,*) ' > IRPoleCheckThreshold      = ',IRPoleCheckThreshold
      write(*,*) ' > PrecisionVirtualAtRunTime = '
     $     ,PrecisionVirtualAtRunTime
      if (SelectedContributionTypes(0).gt.0) then
        write(*,*) ' > SelectedContributionTypes = ',
     &   (SelectedContributionTypes(I),I=1,SelectedContributionTypes(0))
      else
        write(*,*) ' > SelectedContributionTypes = All'
      endif
      if (VetoedContributionTypes(0).gt.0) then
        write(*,*) ' > VetoedContributionTypes   = ',
     &    (VetoedContributionTypes(I),I=1,VetoedContributionTypes(0))
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
          write(*,*) ' > SelectedCouplingOrders(',j,') = ',
     &      (SelectedCouplingOrders(i,j),i=1,nsplitorders)
        enddo
      else
        write(*,*) ' > SelectedCouplingOrders    = All'
      endif
      write(*,*) ' > NHelForMCoverHels         = ',NHelForMCoverHels
      write(*,*) ' > VirtualFraction           = ',Virt_fraction
      write(*,*) ' > MinVirtualFraction        = ',Min_virt_fraction
      write(*,*) ' > SeparateFlavourConfigs    = ',separate_flavour_configs
      write(*,*)
     & '==============================================================='
      paramPrinted=.TRUE.
      endif

      close(68)
      HasReadOnce=.TRUE.
  901 continue
      end

      subroutine DefaultFKSParam() 

      implicit none
      integer i,j
      include "FKSParams.inc"

      IRPoleCheckThreshold=1.0d-5
      NHelForMCoverHels=5
      PrecisionVirtualAtRunTime=1d-3
      Virt_fraction=1d0
      QED_squared_selected=-1
      QCD_squared_selected=-1
      Min_virt_fraction=0.005d0
      separate_flavour_configs=.false.
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
      end
