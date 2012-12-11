#!/bin/csh
#
# Source this file to get necessary products for building stdhep.
# NOTE: if your copy of stdhep is a ups product, 
#       you do NOT need to source this file
#
## setup stdhep 
# setenv STDHEP_DIR `pwd` in the top directory
 setup cern
# if ( `ups list gtools | grep current | wc -l` > 0) then
#   setup gtools		# for gmake if not on Linux
# endif
 setup isajet v7_72a
 setup herwig v6_510a
 setup qq v9_2b
 setup pythia v6_413
 # histo is optional
 setup histo
 # choose which compiler to use
 # setup gcc v3_4_3

exit
