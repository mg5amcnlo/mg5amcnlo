#!/bin/csh
# run the standard jetset translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup lund 
set echo

  mv pythiaLHExample.lpt pythiaLHExample.lpt.bak
  mv pythiaLHExample.io  pythiaLHExample.io.bak
 ./pythiaLHExample
