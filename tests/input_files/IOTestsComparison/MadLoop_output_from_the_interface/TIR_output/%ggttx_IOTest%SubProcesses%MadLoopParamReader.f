      subroutine MadLoopParamReader(filename, printParam) 

      implicit none

      CHARACTER(512) fileName, buff, buff2, mode
      CHARACTER*20 MLReductionLib_str,MLReductionLib_str_save
      CHARACTER*2  MLReductionLib_char
      INTEGER MLRed,i,j,k

      include "MadLoopParams.inc"

      logical printParam, couldRead, paramPrinted, find
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

           else if (buff .eq. '#OSThres') then
              read(666,*,end=999) OSThres
              if (OSThres.le.0.0d0) then
                 stop 'OSThres must be > 0'
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

          else if (buff .eq. '#WriteOutFilters') then
             read(666,*,end=999) WriteOutFilters

          else if (buff .eq. '#UseQPIntegrandForNinja') then
             read(666,*,end=999) UseQPIntegrandForNinja

          else if (buff .eq. '#UseQPIntegrandForCutTools') then
             read(666,*,end=999) UseQPIntegrandForCutTools

          else if (buff .eq. '#ImprovePSPoint') then
             read(666,*,end=999) ImprovePSPoint
             if (ImprovePSPoint .lt. -1 .or.
     &           ImprovePSPoint .gt. 2 ) then
               stop 'ImprovePSPoint must be >= -1 and <=2.'
             endif

          else if (buff .eq. '#HelicityFilterLevel') then
             read(666,*,end=999) HelicityFilterLevel
             if (HelicityFilterLevel .lt. 0 .or.
     &           HelicityFilterLevel .gt. 2 ) then
               stop 'HelicityFilterLevel must be >= 0 and <=2.'
             endif

          else if (buff .eq. '#MLReductionLib') then
             read(666,*,end=999) MLReductionLib_str
             MLReductionLib(1:6)=0
             MLReductionLib_str_save=MLReductionLib_str
             j=0
             DO
                i=index(MLReductionLib_str,'|')
                IF(i.EQ.0)THEN
                   MLReductionLib_char=MLReductionLib_str
                ELSE
                   MLReductionLib_char=MLReductionLib_str(:i-1)
                ENDIF
                IF(MLReductionLib_char.EQ.'1 ')THEN
                   MLRed=1
                ELSEIF(MLReductionLib_char.EQ.'2 ')THEN
                   MLRed=2
                ELSEIF(MLReductionLib_char.EQ.'3 ')THEN
                   MLRed=3
                ELSEIF(MLReductionLib_char.EQ.'4 ')THEN
                   MLRed=4
                ELSEIF(MLReductionLib_char.EQ.'5 ')THEN
                   MLRed=5
                ELSEIF(MLReductionLib_char.EQ.'6 ')THEN
                   MLRed=6
                ELSE
                   PRINT *, 'MLReductionLib is wrong: '//
     $                  TRIM(MLReductionLib_str_save)
                   STOP
                ENDIF
                find=.FALSE.
                DO k=1,j
                   IF(MLReductionLib(k).EQ.MLRed)THEN
                      find=.TRUE.
                      EXIT
                   ENDIF
                ENDDO
                IF(.NOT.find)THEN
                   j=j+1
                   MLReductionLib(j)=MLRed
                ENDIF
                IF(i.EQ.0)THEN
                   EXIT
                ELSE
                   MLReductionLib_str=MLReductionLib_str(i+1:)
                ENDIF
             ENDDO
          else if (buff .eq. '#IREGIRECY') then
             read(666,*,end=999) IREGIRECY
          else if (buff .eq. '#IREGIMODE') then
             read(666,*,end=999) IREGIMODE
             if (IREGIMODE .lt. 0 .or.
     &            IREGIMODE .gt.2) then
                stop 'IREGIMODE must be >=0 and <=2.'
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
      write(*,*) ' > MLReductionLib            = '
     $     //TRIM(MLReductionLib_str_save)
      write(*,*) ' > IREGIMODE                 = ',IREGIMODE
      write(*,*) ' > IREGIRECY                 = ',IREGIRECY
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
      write(*,*) ' > HelicityFilterLevel       = ',HelicityFilterLevel
      write(*,*) ' > ImprovePSPoint            = ',ImprovePSPoint
      write(*,*) ' > DoubleCheckHelicityFilter = ',
     &DoubleCheckHelicityFilter
      write(*,*) ' > LoopInitStartOver         = ',LoopInitStartOver
      write(*,*) ' > HelInitStartOver          = ',HelInitStartOver
      write(*,*) ' > ZeroThres                 = ',ZeroThres
      write(*,*) ' > OSThres                   = ',OSThres
      write(*,*) ' > WriteOutFilters           = ',WriteOutFilters
      write(*,*) ' > UseQPIntegrandForNinja    = ',
     &UseQPIntegrandForNinja
      write(*,*) ' > UseQPIntegrandForCutTools = ',
     &UseQPIntegrandForCutTools
      write(*,*)
     & '==============================================================='
      paramPrinted=.TRUE.
      endif

      close(666)

      end

      subroutine DefaultParam() 

      implicit none
      
      include "MadLoopParams.inc"

      MLReductionLib(1)=6
      MLReductionLib(2:6)=0
      IREGIMODE=2
      IREGIRECY=.TRUE.
      CTModeInit=0
      CTModeRun=-1
      NRotations_DP=1
      NRotations_QP=0
      MLStabThres=1.0d-3
      CTStabThres=1.0d-2
      CTLoopLibrary=3
      CheckCycle=3
      MaxAttempts=10
      HelicityFilterLevel=2
      UseLoopFilter=.False.
      DoubleCheckHelicityFilter=.True.
      LoopInitStartOver=.False.
      HelInitStartOver=.False.
      WriteOutFilters=.True.
      ZeroThres=1.0d-9
      OSThres=1.0d-13
      ImprovePSPoint=2
      UseQPIntegrandForCutTools=.True.
      UseQPIntegrandForNinja=.False.

      end
