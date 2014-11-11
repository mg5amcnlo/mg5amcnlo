      subroutine FKSParamReader(filename, printParam, force) 

      implicit none

      logical HasReadOnce, force
      data HasReadOnce/.FALSE./


      character(*) filename
      CHARACTER*64 buff, buff2, mode
      include "FKSParams.inc"

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
           else
             write(*,*) 'The parameter name ',buff(2:),
     &' is not reckognized.'
             stop
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
      write(*,*) ' > NHelForMCoverHels         = ',NHelForMCoverHels
      write(*,*) ' > VirtualFraction           = ',Virt_fraction
      write(*,*) ' > MinVirtualFraction        = ',Min_virt_fraction
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
      
      include "FKSParams.inc"

      IRPoleCheckThreshold=1.0d-5
      NHelForMCoverHels=5
      PrecisionVirtualAtRunTime=1d-3
      Virt_fraction=1d0
      Min_virt_fraction=0.005d0
      end
