!====================================================================
!
!  Define common block with all general parameters used by MadLoop
!  See their definitions in the file MadLoopParams.dat
!
!====================================================================
!
	  integer CTModeInit,CTModeRun,CheckCycle,MaxAttempts,
     &CTLoopLibrary,NRotations_DP,NRotations_QP

      real*8 MLStabThres,CTStabThres,ZeroThres

      logical UseLoopFilter,LoopInitStartOver,DoubleCheckHelicityFilter

      common /MADLOOP/CTModeInit,CTModeRun,NRotations_DP,NRotations_QP,
     &CheckCycle, MaxAttempts,UseLoopFilter,MLStabThres,CTStabThres,
     &CTLoopLibrary,LoopInitStartOver,DoubleCheckHelicityFilter,
     &ZeroThres
