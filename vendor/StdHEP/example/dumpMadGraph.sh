#!/bin/csh
# run the standard jetset translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup lund 
set echo

  mv dumpMadGraph.lpt dumpMadGraph.lpt.bak
  mv dumpMadGraph.dmp  dumpMadGraph.dmp.bak
 ./dumpMadGraph
