#!/bin/bash

set_environment () {

  echo " Set environment variables"

  # Here, define your installation paths, versions etc.
  #INSTALLD=/nfs/farm/g/theory/qcdsim/sp/HEP/HEPMC2
  INSTALLD="$1"
  VERSION="$2"
  TARBALL="$3"
  USR=/usr
  LOCAL=$USR/local/
  export LD_LIBRARY_PATH=/usr/lib:/usr/lib64:$LOCAL/lib:$LOCAL/64/lib:$LOCAL/lib64:$LD_LIBRARY_PATH

  # set SLC5 platform name:
  LCG_PLATFORM=i686
  if [[ "$(uname -m)" == "x86_64" ]] ; then
    LCG_PLATFORM=x86_64
  fi


  # Set flag to correct compiler, good if more than one compiler is available.
  # Script will also work with this commented !change!
#  export LDFLAGS="-L=/usr/local/lib/gcc/i686-pc-linux-gnu/4.3.2"
#  export LDFLAGS="-L=/usr/lib64/gcc/x86_64-suse-linux/4.5"

#  version="2.06.09"

}

run () {
  echo " Unpack HEPMC"
  tar xvzf ${TARBALL}

  echo " Enter HEPMC directory"
  cd HepMC-${VERSION}/

  echo " Configure HEPMC"
  ./configure --prefix=$INSTALLD --with-momentum=GEV --with-length=MM

  echo " Compile HEPMC"
  make

  echo " Install HEPMC"
  make install

  echo " Finished HEPMC installation"
  cd ..

}

set_environment "$@"
run "$@"

