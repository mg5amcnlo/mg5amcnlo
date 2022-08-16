#!/usr/bin/env bash
# Copyright (C) 2021 Torbjorn Sjostrand.
# PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
# Author: Philip Ilten, August 2020.

# This is a simple script to prepare Pythia for release. Execute as
# "./private/release.sh". Pass as an argument the release number, e.g. 303.

# Handle in-place SED for OSX.
ARCH=$(uname | grep -i -o -e Linux -e Darwin)
ARCH=$(echo $ARCH | awk '{print toupper($0)}')
if [ "$ARCH" == "DARWIN" ]; then SED="sed -i .sed"; else SED="sed -i"; fi
if ! type "sed" &> /dev/null; then
    echo "Error: SED not found."; exit; fi
TOP=$PWD
VER=$1
if [ -z $VER ]; then echo "./private/release.sh <VER>"; exit; fi

# Generate the Python interface.
echo "(1) Generating the Python interface, see python.log."
cd plugins/python
./generate > python.log
cd $TOP

# Update the version number and date.
echo "(2) Updating the version number and date."
SRC=include/Pythia8/Pythia.h
sed -i "s|\(PYTHIA_VERSION 8\).*|\1.$VER|g" $SRC
sed -i "s|\(PYTHIA_VERSION_INTEGER 8\).*|\1$VER|g" $SRC
SRC=src/Pythia.cc
sed -i "s|\(const double Pythia::VERSIONNUMBERCODE = 8\).*|\1.$VER;|g" $SRC
XML=share/Pythia8/xmldoc/Version.xml
DATE=`date +%Y%m%d`
sed -i "s|\(Number\" default=\"8\).*\"|\1.$VER\"|g" $XML
sed -i "s|\(Date\" default=\"\).*\"|\1$DATE\"|g" $XML
XML=share/Pythia8/xmldoc/UpdateHistory.xml
DATE=`date +"%d %B %Y"`
sed -i "s|\(<h3>8\.$VER\).*|\1: $DATE</h3>|g" $XML
DATE=`date +"%Y"`
FILES=`grep -lR "Copyright (C)" *`
for FILE in $FILES; do
    sed -i "s|Copyright (C) 20[0-9][0-9]|Copyright (C) $DATE|g" $FILE; done

# Check the XML.
echo "(3) Checking the XML formatting, see checkXML.log."
./private/checkXML.py > checkXML.log

# Run trimfile.
echo "(4) Running trimfile, see trim*.log."
cd private
make trimfile > /dev/null
cd $TOP
./private/trimfile -d src > trimSrc.log
./private/trimfile -d examples > trimExamples.log
./private/trimfile -d include/Pythia8 > trimInclude.log
./private/trimfile -d include/Pythia8Plugins > trimPlugins.log
./private/trimfile -d share/Pythia8/xmldoc > trimXML.log

# Convert the XML.
echo "(5) Converting the XML, see convertXML.log."
cd private
make convertXML > /dev/null
cd $TOP
./private/convertXML > convertXML.log

# Create the XML index.
echo "(6) Indexing the XML."
./private/indexXML.py

# Create the example index.
echo "(7) Indexing the examples."
./private/indexMains.py

# Create the doxygen.
echo "(8) Generating Doxygen, see doxygen.log."
./private/doxygen.sh > doxygen.log
tar -czf doxygen8$VER.tgz doxygen
rm -rf doxygen

# Package the release.
echo "(9) Package the release."
REL=pythia8$VER
rsync -a --exclude ".git*" --exclude "doxygen*" --exclude "$REL*" . ./$REL
cd $REL
make distclean > /dev/null
find . -type f -name "*.log" -print0 | xargs -0 rm -f
rm -rf .git .gitignore private
chmod a-w share/Pythia8/xmldoc/*
cd $TOP
tar -czf $REL.tgz $REL
rm -rf $REL
