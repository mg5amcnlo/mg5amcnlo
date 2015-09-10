#!/bin/bash

set_environment () {

  echo " Set environment variables"

  # Here, define your installation paths, versions etc.
  gccversion="$(gcc -dumpversion)"
  BOOST="$1"
  INSTALLD="$2"
  VERSION="$3"
  LOCAL=$INSTALLD
  export LHAPATH=$LOCAL/share/LHAPDF
  export LD_LIBRARY_PATH="/nfs/farm/g/theory/qcdsim/shoeche/tools/gcc/4.8.0/lib64:/nfs/farm/g/theory/qcdsim/shoeche/tools/gcc/4.8.0/lib:/nfs/farm/g/theory/qcdsim/shoeche/root/lib:/nfs/farm/g/theory/qcdsim/shoeche/tools/lib64:/nfs/farm/g/theory/qcdsim/shoeche/tools/lib:/nfs/farm/g/theory/qcdsim/sp/hepsoft/RIVET2/lib:/usr/lib64/openmpi/lib"

  # set SLC5 platform name:
  LCG_PLATFORM=i686
  if [[ "$(uname -m)" == "x86_64" ]] ; then
    LCG_PLATFORM=x86_64
  fi
}

run () {

  workd=$(pwd)

  echo " Download LHAPDF $VERSION"
  mkdir $INSTALLD
  cd $INSTALLD
  wget http://www.hepforge.org/archive/lhapdf/LHAPDF-${VERSION}.tar.gz

  echo " Unpack LHAPDF"
  tar xvzf LHAPDF-${VERSION}.tar.gz

  echo " Enter LHAPDF directory"
  cd LHAPDF-${VERSION}/

  echo " Configure LHAPDF"
  LIBRARY_PATH=$LD_LIBRARY_PATH ./configure CXXFLAGS="-static-libstdc++" --prefix=$LOCAL --bindir=$LOCAL/bin --datadir=$LOCAL/share --libdir=$LOCAL/lib --disable-python --with-boost=$BOOST --enable-static

  echo " Compile LHAPDF"
  LIBRARY_PATH=$LD_LIBRARY_PATH make

  echo " Install LHAPDF"
  LIBRARY_PATH=$LD_LIBRARY_PATH make install

  echo "copy index and conf file"
  cd $INSTALLD
  index="$(find . -name 'pdfsets.index')"
  cp $index $INSTALLD/share/LHAPDF/
  conf="$(find . -name 'lhapdf.conf')"
  cp $conf $INSTALLD/share/LHAPDF/

  echo " Get LHAPDF sets"
  cd $INSTALLD/share/LHAPDF
  wget --no-parent --recursive --level=1 -e robots=off -A.tar.gz -nd https://www.hepforge.org/archive/lhapdf/pdfsets/6.1/
  ls -1 *.tar.gz | while read line; do tar xvfz $line; done

  echo " Finished LHAPDF installation"
  cd $workd

}

set_environment "$@"
run "$@"

