!====================================================================
!
!  Define common block with all general parameters used by MadLoop
!  See their definitions in the file MadLoopParams.dat
!
!====================================================================
!
	  integer CTModeInit,CTModeRun,CheckCycle,MaxAttempts,
     &CTLoopLibrary,NRotations_DP,NRotations_QP,ImprovePSPoint

      real*8 MLStabThres,CTStabThres,ZeroThres

      logical UseLoopFilter,LoopInitStartOver,DoubleCheckHelicityFilter
	  logical HelInitStartOver

      common /MADLOOP/CTModeInit,CTModeRun,NRotations_DP,NRotations_QP,
     &ImprovePSPoint,CheckCycle, MaxAttempts,UseLoopFilter,MLStabThres,
     &CTStabThres,CTLoopLibrary,LoopInitStartOver,
     &DoubleCheckHelicityFilter,ZeroThres,HelInitStartOver
