#!/bin/csh

set echo

setenv PDG_MASS_TBL $STDHEP_DIR/mass_width_2006.csv

mv testcsvpdg.lpt testcsvpdg.lpt.bak
./testcsvpdg
