#!/bin/csh
# run the standard qq translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup qq 
set echo

  mv qqExample.lpt qqExample.lpt.bak
  ./qqExample
