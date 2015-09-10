#!/bin/bash

run () {

    BOOSTINSTALLD="$1"
    VERSION="$2"
	TARBALLPATH="$3"
    VVERSION="$(echo $2 | sed 's/\./_/g')"
    mkdir -p $BOOSTINSTALLD
    cd $BOOSTINSTALLD
    echo "Download boost"
    tar xvzf ${TARBALLPATH}
	echo "Enter boost directory and install boost"
    cd boost_${VVERSION}
    ./bootstrap.sh
    ./b2 install --prefix=$BOOSTINSTALLD
    cd ..
    echo "Finished installing boost"

}

run "$@"
