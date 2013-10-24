#!/bin/csh
# run the standard translation test executable
# the environment variables must be defined as below:
#   setup listStdHepConv 
set echo

  mv listStdHepConv.lpt listStdHepConv.lpt.bak
 ./listStdHepConv
