#!/bin/bash

set_environment () {

  echo " Set environment variables: SO FAR NOTHING DONE HERE"

}

run () {

 echo ""
 echo "-----------------------------------------------"
 echo "-----------------------------------------------"
 echo "----- This script installs the HEP tools: -----"
 echo "--- BOOST, LHAPDF, HEPMC, PYTHIA --------------"
 echo "-----------------------------------------------"
 echo "-----------------------------------------------"

 local TOOLS="/nfs/farm/g/theory/qcdsim/sp/HEP2"
 mkdir -p $TOOLS

 echo ""
 echo "-- Begin installing BOOST --------------------"
 chmod +x installBOOST.sh
 ./installBOOST.sh $TOOLS/BOOST 1.59.0

 echo ""
 echo "-- Begin installing LHAPDF --------------------"
 chmod +x installLHAPDF6.sh
 ./installLHAPDF6.sh $TOOLS/BOOST $TOOLS/LHAPDF6 6.1.5

 echo ""
 echo "-- Begin installing HEPMC ---------------------"
 chmod +x installHEPMC2.sh
 ./installHEPMC2.sh $TOOLS/HEPMC2 2.06.09

 echo ""
 echo "-- Begin installing PYTHIA --------------------"
 chmod +x installPYTHIA8.sh
 GZIP="/nfs/farm/g/theory/qcdsim/sp/gensoft"
 ./installPYTHIA8.sh $TOOLS/HEPMC2 $GZIP $TOOLS/BOOST $TOOLS/LHAPDF6 $TOOLS/PYTHIA8 8210

 echo ""
 echo "-- Finished installing HEP tools ---------------"
 echo "-----------------------------------------------"
 echo "-----------------------------------------------"

}

set_environment
run "$@"
