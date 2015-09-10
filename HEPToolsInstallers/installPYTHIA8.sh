#!/bin/bash

set_environment () {

  echo " Set environment variables"

  # Here, define your installation paths, versions etc.
  #INSTALLPATH=/nfs/farm/g/theory/qcdsim/sp/HEP/PYTHIA8
#  LHAPDF6PATH=/nfs/farm/g/theory/qcdsim/sp/HEP/LHAPDF6
#  HEPMC2PATH=/nfs/farm/g/theory/qcdsim/sp/HEP/HEPMC2
#  HEPMC2INCLUDEPATH=$HEPMC2PATH/include
#  GZIPPATH=/nfs/farm/g/theory/qcdsim/sp/gensoft
#  BOOSTPATH=/nfs/farm/g/theory/qcdsim/sp/HEP/BOOST

  HEPMC2PATH="$1"
  HEPMC2INCLUDEPATH=$HEPMC2PATH/include
  GZIPPATH="$2"
  BOOSTPATH="$3"
  LHAPDF6PATH="$4"
  INSTALLPATH="$5"
  VERSION="$6"
#  version="8210"

}

run () {

  workd=$(pwd)

  echo " Download PYTHIA8 version $VERSION"
  mkdir $INSTALLPATH
  cd $INSTALLPATH
  wget http://home.thep.lu.se/~torbjorn/pythia8/pythia${VERSION}.tgz

  echo " Unpack PYTHIA8"
  tar xvzf pythia${VERSION}.tgz

  echo " Enter PYTHIA8 directory"
  cd pythia${VERSION}/

  echo " Configure PYTHIA8"
  make distclean
  configStr="./configure --prefix=$INSTALLPATH --with-hepmc2=$HEPMC2PATH --with-hepmc2-include=$HEPMC2INCLUDEPATH --with-lhapdf6=$LHAPDF6PATH --with-lhapdf6-plugin=LHAPDF6.h --with-gzip=$GZIPPATH --with-boost=$BOOSTPATH"
  echo "$configStr"
  $configStr

  echo " Compile PYTHIA8"
  make
  make install

  echo " Compile PYTHIA8 examples"
  cd $INSTALLPATH/share/Pythia8/examples
  ls -1 main*.cc | while read line
  do
    make "$(echo "$line" | sed "s,\.cc,,g")"  
  done

  echo " Finished LHAPDF installation"
  cd $workd

}

set_environment "$@"
run "$@"
