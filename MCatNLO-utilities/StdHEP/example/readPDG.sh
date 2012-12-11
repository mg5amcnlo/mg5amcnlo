#!/bin/csh
# run the standard translation test executable
# the environment variables must be defined as below:
#   setup readPDG 
set echo

  mv readPDG.lpt readPDG.lpt.bak
 ./readPDG
