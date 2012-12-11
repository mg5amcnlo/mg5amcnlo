#!/bin/sh

echo "testCInOut.sh: rename old files"
mv testCInOut.lpt      testCInOut.lpt.bak
mv testCInOutWrite.txt testCInOutWrite.txt.bak
mv testCInOutRead.txt  testCInOutRead.txt.bak
mv testCInOut.io       testCInOut.io.bak

echo "testCInOut.sh: run testCInOut"
./testCInOut >& testCInOut.lpt 

echo "testCInOut.sh: compare output"
cmd1=`diff -q -b testCInOutRead.txt testCInOutWrite.txt`
if [ -n "$cmd1" ]
then
  echo $cmd1
  exit 1;
else
  echo "testCInOut.sh: TEST IS SUCCESSFUL"
fi

exit 0;
