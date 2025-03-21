C     General Pythia control
C     is_pythia_active =
C        -1 : Pythia unavailable
C         0 : Pythia available but not initialized yet
C         1 : Pythia available and already initialized
      integer is_pythia_active
	  character pythia_cmd_file(500)
      common/pythia_control/is_pythia_active, pythia_cmd_file
