#! /bin/bash
# This script will build all Docker conatiners from configuration in
# this directory. Caution, this might be a long process, and take up a
# lot of disk space. Alternatively, pass the configuration files to
# build as arguments to the script, e.g. "./build.sh doxygen.dev".

IMAGES=$@
if [ -z $IMAGES ]; then IMAGES=`ls *.dev`; fi
for IMAGE in $IMAGES; do
    if [ -f $IMAGE ]; then
	docker build --no-cache -t pythia8/dev:${IMAGE%.*} -f $IMAGE .\
	       > ${IMAGE%.*}.log &
    fi
done
