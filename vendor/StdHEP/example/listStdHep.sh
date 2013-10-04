#!/bin/csh
# run the standard translation test executable
# the environment variables must be defined as below:
#   setup listStdHep 
set echo

  mv listStdHep.lpt listStdHep.lpt.bak
 ./listStdHep
