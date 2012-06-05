!====================================================================
!
!  Define common block with all general parameters used by MadLoop
!  See their definitions in the file MadLoopParams.dat
!
!====================================================================
!
	  integer CTModeInit,CTModeRun,CheckCycle,MaxAttempts,
     &CTLoopLibrary

      real*8 MLStabThres,CTStabThres,ZeroThres

      logical UseLoopFilter,LoopInitStartOver

      common /MADLOOP/CTModeInit,CTModeRun,CheckCycle,
     &MaxAttempts,UseLoopFilter,MLStabThres,CTStabThres,
     &CTLoopLibrary,LoopInitStartOver,ZeroThres
