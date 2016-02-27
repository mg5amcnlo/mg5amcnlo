!====================================================================
!
!  Define common block with all general parameters used by MadLoop
!  See their definitions in the file MadLoopParams.dat
!
!====================================================================
!
	  integer CTModeInit,CTModeRun,CheckCycle,MaxAttempts,
     &CTLoopLibrary,NRotations_DP,NRotations_QP,ImprovePSPoint,
     &MLReductionLib(7),IREGIMODE,HelicityFilterLevel

      real*8 MLStabThres,CTStabThres,ZeroThres,OSThres

      logical UseLoopFilter,LoopInitStartOver,DoubleCheckHelicityFilter
	  logical HelInitStartOver,IREGIRECY,WriteOutFilters
      logical UseQPIntegrandForNinja, UseQPIntegrandForCutTools

      common /MADLOOP/CTModeInit,CTModeRun,NRotations_DP,NRotations_QP,
     &ImprovePSPoint,CheckCycle, MaxAttempts,UseLoopFilter,MLStabThres,
     &CTStabThres,CTLoopLibrary,LoopInitStartOver,
     &DoubleCheckHelicityFilter,ZeroThres,OSThres,HelInitStartOver,
     &MLReductionLib,IREGIMODE,HelicityFilterLevel,IREGIRECY,
     &WriteOutFilters,UseQPIntegrandForNinja,UseQPIntegrandForCutTools
