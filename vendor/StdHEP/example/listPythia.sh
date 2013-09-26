#!/bin/csh
# the environment variables must be defined as below:
#   setup stdhep 
#   setup lund
set echo

mv listPythia.lpt listPythia.lpt.bak
./listPythia
