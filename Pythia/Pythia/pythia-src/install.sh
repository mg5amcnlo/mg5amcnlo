
workd=/nfs/farm/g/theory/qcdsim/sp
make distclean

configStr="./configure"
configStr+=" --prefix=$(pwd)"
configStr+=" --with-hepmc2=$workd/hepsoft/built-with-devtools-2/RIVET241 --with-hepmc2-include=$workd/hepsoft/built-with-devtools-2/RIVET241/include"
configStr+=" --with-lhapdf6=$workd/hepsoft/built-with-devtools-2 --with-lhapdf6-plugin=LHAPDF6.h"
configStr+=" --enable-debug --with-gzip=$workd/gensoft"
configStr+=" --with-boost=$workd/gensoft/built-with-devtools-2/BOOST"
configStr+=" --with-promc=$workd/hepsoft/PROMC/promc"
configStr+=" --with-fastjet3=/nfs/farm/g/theory/qcdsim/sp/hepsoft/fastjet-3.2.2/fastjet-install"

echo "$configStr --cxx-common='-ldl -fPIC -lstdc++ -DHEPMC2HACK'"
$configStr --cxx-common='-ldl -fPIC -lstdc++ -DHEPMC2HACK'

make
make install
