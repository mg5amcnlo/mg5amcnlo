# compile custom pythia version shipped with the code.

workd=$(pwd)

Pythia=Pythia/pythia8245MultiscalesFix

cd $Pythia

make distclean

configStr="./configure"
configStr+=" --prefix=$(pwd)"
configStr+=" --with-hepmc2=/usr/local/"

echo "$configStr --cxx-common='-ldl -fPIC -lstdc++ --std=c++14'"
$configStr --cxx-common='-ldl -fPIC -lstdc++ --std=c++14'

make
make install

####################################################
####################################################

cd $workd

####################################################
####################################################
# now create files for sudakov generation
# Note: not part of the release

#cd $workd/../SudGen
#echo "WORK=$workd/" > makefile.inc
#echo "PYTHIA8INCLUDE=\$(WORK)/$Pythia/include" >> makefile.inc
#echo "PYTHIA8LIB=\$(WORK)/$Pythia/lib" >> makefile.inc
#echo "PYTHIA8FLAGS=-lstdc++ -lz -ldl -fPIC" >> makefile.inc
#
#rm gridsudgen
#make gridsudgen

####################################################
####################################################

cd $workd

