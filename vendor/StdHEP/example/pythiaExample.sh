#!/bin/csh
# run the standard jetset translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup lund 
set echo

  mv pythiaExample.lpt pythiaExample.lpt.bak
  mv pythiaExample.io  pythiaExample.io.bak
 ./pythiaExample
