

make distclean

configStr="./configure"
configStr+=" --prefix=$(pwd)/../dire-build"
configStr+=" --with-pythia8=$(pwd)/../../Pythia/pythia-src"

echo "$configStr"
$configStr

make
make install
