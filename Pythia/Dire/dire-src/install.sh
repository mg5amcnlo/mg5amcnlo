

#make distclean

configStr="./configure"
configStr+=" --prefix=/nfs/farm/g/theory/qcdsim/sp/hepsoft/diretest/dire-build"
#./configure --prefix=/nfs/farm/g/theory/qcdsim/sp/hepsoft/diretest/dire-build --with-pythia8=/nfs/farm/g/theory/qcdsim/sp/hepsoft/pythia/Pythia8/pythia8/pythia82/branches/sp-dev
configStr+=" --with-pythia8=/nfs/farm/g/theory/qcdsim/sp/hepsoft/pythia/Pythia8/pythia8/pythia82/branches/sp-dev-new"

echo "$configStr"
$configStr

make
make install
