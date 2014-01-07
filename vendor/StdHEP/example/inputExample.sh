#!/bin/csh
# run the standard translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
set echo

  mv inputExample.lpt inputExample.lpt.bak
 ./inputExample
