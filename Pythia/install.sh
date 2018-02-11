
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
configStr+=" --with-hepmc2=/home/prestel/work/2018/HEPMC2"
configStr+=" --enable-debug --with-gzip=/home/prestel/work/2018/ZLIB"
configStr+=" --with-boost=/home/prestep/work/2018/BOOST"

echo "$configStr --cxx-common='-ldl -fPIC -lstdc++ -DHEPMC2HACK'"
$configStr --cxx-common='-ldl -fPIC -lstdc++ -DHEPMC2HACK'

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

make distclean

configStr="./configure"
configStr+=" --prefix=$workd/Dire/dire-build"
configStr+=" --with-pythia8=$workd/Pythia/pythia-src"

echo "$configStr"
$configStr

make
make install

####################################################
####################################################

cd $workd

