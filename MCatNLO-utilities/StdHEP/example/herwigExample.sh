#!/bin/csh
# run the standard herwig translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup herwig

set echo
  mv herwigExample.lpt herwigExample.lpt.bak
  mv herwigExample.io  herwigExample.io.bak
 ./herwigExample
