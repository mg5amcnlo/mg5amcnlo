
workd=$(pwd)

####################################################
####################################################

#rm -rf Pythia/pythia-src Pythia/pythia-build
#mkdir -p Pythia

#rm -rf Pythia/HepMC-src Pythia/HepMC-build
#wget lcgapp.cern.ch/project/simu/HepMC/download/HepMC-2.06.09.tar.gz
#tar xf HepMC-2.06.09.tar.gz
#mv HepMC-2.06.09 Pythia/HepMC-src
#cd Pythia/HepMC-src
#./configure --prefix=$workd/Pythia/HepMC-build --with-momentum=GEV --with-length=MM
#make
#make check
#make install
#cd $workd

#sudo apt-get install libbz2-dev
#rm -rf Pythia/boost-src Pythia/boost-build
#wget http://sourceforge.net/projects/boost/files/boost/1.55.0/boost_1_55_0.tar.gz
#tar xf boost_1_55_0.tar.gz
#mv boost_1_55_0 Pythia/boost-src
#cd Pythia/boost-src
#./bootstrap.sh
#./b2 install --prefix=$workd/Pythia/boost-build

#tar xvfz pythia8.tar.gz
#mv pythia8 Pythia/pythia-src
cd Pythia/pythia-src

#make distclean

configStr="./configure"
configStr+=" --prefix=$workd/Pythia/pythia-build"
configStr+=" --with-hepmc2=/nfs/"
configStr+=" --enable-debug --with-gzip=/usr/local"
configStr+=" --with-boost=/usr/local"

echo "$configStr --cxx-common='-ldl -fPIC -lstdc++ -DHEPMC2HACK'"
$configStr --cxx-common='-ldl -fPIC -lstdc++ -DHEPMC2HACK'

#./configure --with-hepmc2=/home/prestel/work/2018/HEPMC2 --with-lhapdf6=/home/prestel/work/2018/LHAPDF6 --with-gzip=/home/prestel/work/2018/ZLIB --with-boost=/home/prestel/work/2018/BOOST --with-fastjet3=/home/prestel/work/2018/RIVET --cxx-common="-ldl -fPIC -lstdc++ --std=c++14"

make
make install

####################################################
####################################################

cd $workd

####################################################
####################################################

#rm -fr Dire/dire-src Dire/dire-build
#mkdir -p Dire
#rm -fr Dire/dire-build

#tar xvfz DIRE-2.001alpha.tar.gz
#mv DIRE-2.001alpha Dire/dire-src
cd Dire/dire-src

#make clean

configStr="./configure"
configStr+=" --prefix=$workd/Dire/dire-build"
configStr+=" --with-pythia8=$workd/Pythia/pythia-src"

echo "$configStr"
$configStr

make
make install

####################################################
####################################################
# now create files for sudakov generation

cd $workd/../SudGen
echo "WORK=$workd/../" > makefile.inc
echo "PYTHIA8INCLUDE=\$(WORK)/Pythia/Pythia/pythia-src/include" >> makefile.inc
echo "PYTHIA8LIB=\$(WORK)/Pythia/Pythia/pythia-src/lib" >> makefile.inc
echo "DIRELIB=\$(WORK)/Pythia/Pythia/pythia-src/../../Dire/dire-src/lib" >> makefile.inc
echo "PYTHIA8FLAGS=-lstdc++ -lz -ldl -fPIC" >> makefile.inc

rm gridsudgen
make gridsudgen

####################################################
####################################################

cd $workd

