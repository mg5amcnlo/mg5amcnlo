#!/bin/csh
# run the standard isajet translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup isajet
set echo

  mv isajetExample.lpt isajetExample.lpt.bak
  mv isajetExample.io  isajetExample.io.bak
  cp $STDHEP_DIR/example/isajetExample.cmd .
 ./isajetExample
