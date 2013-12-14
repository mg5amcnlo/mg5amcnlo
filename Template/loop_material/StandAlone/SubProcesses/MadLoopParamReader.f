      subroutine MadLoopParamReader(filename, printParam) 

      implicit none

      CHARACTER*64 fileName, buff, buff2, mode
      include "MadLoopParams.inc"

      logical printParam, couldRead, paramPrinted
      data paramPrinted/.FALSE./
      couldRead=.False.
!     Default parameters

      open(666, file=fileName, err=676, status='OLD', action='READ')
      do
         read(666,*,end=999) buff
         if(index(buff,'#').eq.1) then

           if (buff .eq. '#CTModeInit') then
             read(666,*,end=999) CTModeInit
             if (CTModeInit .lt. 0 .or.
     &           CTModeInit .gt. 6 ) then
               stop 'CTModeInit must be >= 0 and <=6.'
             endif
           
           else if (buff .eq. '#CTModeRun') then
             read(666,*,end=999) CTModeRun
             if (CTModeRun .lt. -1 .or.
     &           CTModeRun .gt. 6 ) then
               stop 'CTModeRun must be >= -1 and <=6.'
             endif 

           else if (buff .eq. '#NRotations_DP') then
             read(666,*,end=999) NRotations_DP
             if (NRotations_DP .lt. 0 .or.
     &           NRotations_DP .gt. 2 ) then
               stop 'NRotations_DP must be >= 0 and <=2.'
             endif

           else if (buff .eq. '#NRotations_QP') then
             read(666,*,end=999) NRotations_QP
             if (NRotations_QP .lt. 0 .or.
     &           NRotations_QP .gt. 2 ) then
               stop 'NRotations_QP must be >= 0 and <=2.'
             endif

           else if (buff .eq. '#MLStabThres') then
             read(666,*,end=999) MLStabThres
             if (MLStabThres.lt.0.0d0) then
                 stop 'MLStabThres must be >= 0'
             endif
          
           else if (buff .eq. '#CTLoopLibrary') then
             read(666,*,end=999) CTLoopLibrary
             if (CTLoopLibrary.lt.2 .or.
     &           CTLoopLibrary.gt.3) then
                 stop 'CTLoopLibrary must be >= 2 and <=3.'
             endif

           else if (buff .eq. '#CTStabThres') then
             read(666,*,end=999) CTStabThres
             if (CTStabThres.le.0.0d0) then
                 stop 'CTStabThres must be > 0'
             endif

           else if (buff .eq. '#ZeroThres') then
             read(666,*,end=999) ZeroThres
             if (ZeroThres.le.0.0d0) then
                 stop 'ZeroThres must be > 0'
             endif

           else if (buff .eq. '#CheckCycle') then
             read(666,*,end=999) CheckCycle
             if (CheckCycle.lt.1) then
                 stop 'CheckCycle must be >= 1'
             endif

           else if (buff .eq. '#MaxAttempts') then
             read(666,*,end=999) MaxAttempts
             if (MaxAttempts.lt.1) then
                 stop 'MaxAttempts must be >= 1'
             endif

          else if (buff .eq. '#UseLoopFilter') then
             read(666,*,end=999) UseLoopFilter

          else if (buff .eq. '#DoubleCheckHelicityFilter') then
             read(666,*,end=999) DoubleCheckHelicityFilter

          else if (buff .eq. '#LoopInitStartOver') then
             read(666,*,end=999) LoopInitStartOver

          else if (buff .eq. '#HelInitStartOver') then
             read(666,*,end=999) HelInitStartOver

          else if (buff .eq. '#ImprovePSPoint') then
             read(666,*,end=999) ImprovePSPoint
             if (ImprovePSPoint .lt. -1 .or.
     &           ImprovePSPoint .gt. 2 ) then
               stop 'ImprovePSPoint must be >= -1 and <=2.'
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
      write(*,*) '##E00 Error:: MadLoop parameter file ',fileName,
     &' could not be found or is malformed. Please specify it.'
      stop 
C     Below is the code if one desires to let the code continue with
C     a non existing or malformed parameter file
      write(*,*) '##I01 INFO :: The file ',fileName,' could not be ',
     & ' open or did not contain the necessary information. The ',
     & ' default MadLoop parameters will be used.'
      call DefaultParam()
      goto 999

  998 continue

      if(printParam.and..not.paramPrinted) then
      write(*,*)
     & '==============================================================='
      if (couldRead) then      
        write(*,*) 'INFO: MadLoop read these parameters from '
     &,filename
      else
        write(*,*) 'INFO: MadLoop used the default parameters.'
      endif
      write(*,*)
     & '==============================================================='
      write(*,*) ' > CTModeRun                 = ',CTModeRun
      write(*,*) ' > MLStabThres               = ',MLStabThres
      write(*,*) ' > NRotations_DP             = ',NRotations_DP
      write(*,*) ' > NRotations_QP             = ',NRotations_QP
      write(*,*) ' > CTStabThres               = ',CTStabThres
      write(*,*) ' > CTLoopLibrary             = ',CTLoopLibrary
      write(*,*) ' > CTModeInit                = ',CTModeInit
      write(*,*) ' > CheckCycle                = ',CheckCycle
      write(*,*) ' > MaxAttempts               = ',MaxAttempts
      write(*,*) ' > UseLoopFilter             = ',UseLoopFilter
      write(*,*) ' > ImprovePSPoint            = ',ImprovePSPoint
      write(*,*) ' > DoubleCheckHelicityFilter = ',
     &DoubleCheckHelicityFilter
      write(*,*) ' > LoopInitStartOver         = ',LoopInitStartOver
      write(*,*) ' > HelInitStartOver          = ',HelInitStartOver
      write(*,*) ' > ZeroThres                 = ',ZeroThres
      write(*,*)
     & '==============================================================='
      paramPrinted=.TRUE.
      endif

      close(666)

      end

      subroutine DefaultParam() 

      implicit none
      
      include "MadLoopParams.inc"

      CTModeInit=0
      CTModeRun=-1
      NRotations_DP=1
      NRotations_QP=0
      MLStabThres=1.0d-3
      CTStabThres=1.0d-2
      CTLoopLibrary=3
      CheckCycle=3
      MaxAttempts=10
      UseLoopFilter=.False.
      DoubleCheckHelicityFilter=.True.
      LoopInitStartOver=.False.
      HelInitStartOver=.False.
      ZeroThres=1.0d-9
      ImprovePSPoint=2

      end
