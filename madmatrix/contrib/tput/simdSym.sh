#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Apr 2021) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2021-2024).
# Integrated with the MadGraph7 project in Feb 2026.

#----------------------------------------------------------------------------------------------
# See http://sponce.web.cern.ch/sponce/CSC/slides/PracticalVectorization.booklet.pdf
# See https://software.intel.com/sites/landingpage/IntrinsicsGuide
# [NB: the trailing 'd' is for double; a trailing 's' would be for single precision]
#----------------------------------------------------------------------------------------------
# https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=addsd&expand=154
# (x1) : addsd xmm
# https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=addpd&expand=121,124,127
# (x2) : addpd xmm
# (x4) : vaddpd ymm
# (x8) : vaddpd zmm
#----------------------------------------------------------------------------------------------
# https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mulsd&expand=3949
# (x1) : mulsd xmm
# https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mulpd&expand=3919,3922,3925
# (x2) : mulpd xmm
# (x4) : vmulpd ymm
# (x8) : vmulpd zmm
#----------------------------------------------------------------------------------------------
# Below NNN is 132, 213, 231
# https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=vfmadd&expand=2537,2541,2545
# (x2) : vfmaddNNNpd xmm
# (x4) : vfmaddNNNpd ymm
# (x8) : vfmaddNNNpd zmm
#----------------------------------------------------------------------------------------------

# Auxiliary function of mainCountSyms
# Env: $dump must point to the objdump file
function countSyms() {
  if [ "$dump" == "" ] || [ ! -e $dump ]; then echo "ERROR! File '$dump' not found"; exit 1; fi 
  for sym in "$@"; do
    # Use cut -f3- to print only the assembly code after two leading fields separated by tabs
    printf "(%24s : %4d) " "'$sym'" $(cat $dump | awk '/^ +[[:xdigit:]]+:\t/' | cut -f3- | egrep "$sym" | wc -l)
  done; printf "\n"
}
#dump=$1; shift; countSyms $* # for debugging (./simdSym.sh ./build.none/CPPProcess_cpp.o.objdump 'addsd.*xmm')

function mainCountSyms() {
  # Command line arguments: select file
  if [ "$1" == "" ] || [ "$2" != "" ]; then
    echo "Usage:   $0 <filename>"
    echo "Example: $0 ./check.exe"
    exit 1
  fi
  file=$1
  # Disassemble selected file
  dump=${file}.objdump
  objdump -d -C $file > ${dump}
  # Count symbols in selected file
  countSyms '^[^v].*ss.*xmm' '^[^v].*sd.*xmm'
  countSyms '^[^v].*ps.*xmm' '^[^v].*pd.*xmm'
  countSyms '^v.*ps.*xmm' '^v.*pd.*xmm'
  countSyms '^v.*ps.*ymm' '^v.*pd.*ymm'
  ###countSyms '===' '^vpcmp[n]*eqq.*(x|y)mm'
  countSyms '===' '^vpcmpeqq.*(x|y)mm'
  countSyms '===' '^vpcmpneqq.*(x|y)mm'
  countSyms '^v.*32x2.*(x|y)mm' '^v.*64x2.*(x|y)mm'
  ###countSyms '^v.*32x2.*xmm' '^v.*64x2.*xmm'
  ###countSyms '^v.*32x2.*ymm' '^v.*64x2.*ymm'
  countSyms '^v.*dqa32.*(x|y)mm' '^v.*dqa64.*(x|y)mm'
  ###countSyms '^v.*dqa32.*xmm' '^v.*dqa64.*xmm'
  ###countSyms '^v.*dqa32.*ymm' '^v.*dqa64.*ymm'
  countSyms '^v.*zmm' '^v.*zmm'
  ###countSyms '^v.*ps.*zmm' '^v.*pd.*zmm'
}
#mainCountSyms $*

#----------------------------------------------------------------------------------------------

# Auxiliary function of mainListSyms
# Env: $dump must point to the objdump file
function listSyms() {
  if [ "$dump" == "" ] || [ ! -e $dump ]; then echo "ERROR! File '$dump' not found"; exit 1; fi 
  # Use cut -f3- to print only the assembly code after two leading fields separated by tabs
  cat $dump | awk '/^ +[[:xdigit:]]+:\t/' | cut -f3- | egrep '((x|y|z)mm| vs)' | sed -r 's/ .*(%(x|y|z)mm|vs).*/ \1/g' | sort | uniq -c | awk '{printf "%5s %-15s %5d\n",$3,$2,$1}' | sort -k 1,2
}
###dump=$1; shift; listSyms $* # for debugging

function mainListSyms() {
  # Command line arguments: --disassemble-all?
  dis=-d
  if [ "$1" == "-D" ]; then
    dis=-D
    shift
  fi
  # Command line arguments: select file
  if [ "$1" == "" ] || [ "$2" != "" ]; then
    echo "Usage:   $0 <filename>"
    echo "Example: $0 ./check.exe"
    exit 1
  fi
  file=$1
  # Disassemble selected file
  dump=${file}.objdump
  objdump $dis -C $file > ${dump}
  # List symbols in selected file
  listSyms
}
mainListSyms $*

#----------------------------------------------------------------------------------------------

function mainCompareSyms() {
  allFileSyms=""
  for avx in none sse4 avx2 512y 512z; do
    file=./build.$avx/CPPProcess_cpp.o
    fileSymsRaw=$file.symlist.raw
    mainListSyms $file > $fileSymsRaw
    ls -l $fileSymsRaw
    allFileSyms="$allFileSyms $fileSymsRaw"
  done
  allSymList=all.symlist
  cat $allFileSyms | sort -u | awk -vf1= -vf2= -vf3=0 '{if ($1==f1 && $2==f2){f3+=$3} else {if(f1!=""){print f1,f2,f3};f1=$1;f2=$2;f3=$3}}END{print f1,f2,f3}' | sort -u -k 1,2 > $allSymList
  for avx in none sse4 avx2 512y 512z; do
    file=./build.$avx/CPPProcess_cpp.o
    fileSymsRaw=$file.symlist.raw
    fileSyms=$file.symlist
    cat $fileSymsRaw | awk -vall=$allSymList -vfmt="%5s %15s %5d\n" -vtot=0 -veof=1 '{f1=$1; f2=$2; f3=$3; tot+=f3; while(f3>0){getline < all; if($1==f1 && $2==f2){printf fmt,$1,$2,f3; f3=0} else {printf fmt,$1,$2,0}}} END{while(getline < all){printf fmt,$1,$2,0};printf fmt,"TOTAL","",tot}' > $fileSyms
    ls -l $fileSyms
  done
}
#mainCompareSyms

#----------------------------------------------------------------------------------------------

