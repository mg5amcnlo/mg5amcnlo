#!/bin/csh
# run the standard jetset translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup lund 
set echo

  mv madToXDR.lpt madToXDR.lpt.bak
  mv madToXDR.io  madToXDR.io.bak
 ./madToXDR
