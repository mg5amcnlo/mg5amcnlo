#!/bin/csh
# run the standard jetset translation test executable
# the environment variables must be defined as below:
#   setup cern 
#   setup stdhep 
#   setup lund 
set echo

  mv pythiaMadGraph.lpt pythiaMadGraph.lpt.bak
  mv pythiaMadGraph.io  pythiaMadGraph.io.bak
 ./pythiaMadGraph
