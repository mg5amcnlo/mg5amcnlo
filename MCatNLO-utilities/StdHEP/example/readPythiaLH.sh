#!/bin/csh
# run the standard translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
set echo

  mv readPythiaLH.lpt readPythiaLH.lpt.bak
 ./readPythiaLH
