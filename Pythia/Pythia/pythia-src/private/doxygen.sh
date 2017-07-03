#!/usr/bin/env bash
# Copyright (C) 2017 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL version 2, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, January 2015.

# This is a simple script to create Doxygen documentation for Pythia.
# It should be run from the main directory, but is located in private/.

# Check Doxygen and SED exist.
if ! type "sed" &> /dev/null; then
    echo "Error: SED not found."; exit; fi
if ! type "doxygen" &> /dev/null; then
    echo "Error: DOXYGEN not found."; exit; fi

# Remove and create new doxygen structure.
rm -rf doxygen
mkdir -p doxygen/include
DIRS="include/Pythia8 include/Pythia8Plugins src"
for DIR in $DIRS; do cp -r $DIR doxygen/$DIR; done

# Handle in-place SED for OSX.
ARCH=$(uname | grep -i -o -e Linux -e Darwin)
ARCH=$(echo $ARCH | awk '{print toupper($0)}')
if [ "$ARCH" == "DARWIN" ]; then SED="sed -i .sed"; else SED="sed -i"; fi

# Convert comments to Doxygen style comments.
FILES=`ls doxygen/src/*.cc doxygen/include/*/*.h`
$SED "s|//\+|//|g"            $FILES    # Ensure no more than // (i.e. ///).
$SED "s|//|///|g"             $FILES    # Change all // to ///.
$SED "s|\( *\)///--|\1//--|g" $FILES    # Change ///- lines to //-.
$SED "s|\( *\)///==|\1//==|g" $FILES    # Change ///= lines to //=.

# Set PYTHIA version in doxygen.cfg.
VER=$(grep Number share/Pythia8/xmldoc/Version.xml | grep -o 8.[0-9][0-9][0-9])
$SED "s|\(PROJECT_NUMBER *= *\)[^ ]*|\1$VER|g" doxygen.cfg

# Create the documentation and clean up.
doxygen private/doxygen.cfg
$SED "s|///|//|g" doxygen/*source.html  # Convert source comments back to //.
for DIR in $DIRS; do rm -rf doxygen/$DIR; done
rm -rf doxygen.cfg.sed doxygen/include
