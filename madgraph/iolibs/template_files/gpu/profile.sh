#!/bin/bash

usage(){
  echo "Usage (GUI analysis): $0 [-cc] [-l label] [-p #blocks #threads #iterations]"
  echo "Usage (CL analysis):  $0 -nogui [-p #blocks #threads #iterations]"
  exit 1
}

# Default options
tag=cu
###cuargs="16384 32 12" # NEW DEFAULT 20.08.10 (faster on local, and allows comparison to global and shared memory)
###ccargs="  256 32 12" # Similar to cuda config, but faster than using "16384 32 12"
cuargs="16384 32 2" # faster tests
ccargs="  256 32 2" # faster tests
args=
label=

# Command line arguments
while [ "$1" != "" ]; do
  # Profile C++ instead of cuda
  if [ "$1" == "-cc" ]; then
    if [ "$tag" != "nogui" ]; then
      tag=cc
      shift
    else
      echo "ERROR! Incompaticle options -gui and -cc"
      usage
    fi
  # Fast no-GUI profiling with ncu
  elif [ "$1" == "-nogui" ]; then
    if [ "$tag" != "cc" ]; then
      tag=nogui
      shift
    else
      echo "ERROR! Incompaticle options -gui and -cc"
      usage
    fi
  # Override blocks/threads/iterations
  # (NB do not exceed 12 iterations: profiling overhead per iteration is huge)
  elif [ "$1" == "-p" ]; then
    if [ "$4" != "" ]; then
      args="$2 $3 $4"    
      shift 4
    else
      usage
    fi
  # Label
  elif [ "$1" == "-l" ]; then
    if [ "$2" != "" ]; then
      label="$2"
      shift 2
    else
      usage
    fi
  # Invalid arguments
  else
    usage
  fi
done

if [ "$tag" == "cc" ]; then
  if [ "$args" == "" ]; then args=$ccargs; fi
  cmd="./check.exe -p $args"
else
  if [ "$args" == "" ]; then args=$cuargs; fi
  cmd="./gcheck.exe -p $args"
fi

ncu="ncu"
nsys="nsys"
ncugui="ncu-ui &"
nsysgui="nsight-sys &"
host=$(hostname)
if [ "${host%%cern.ch}" != "${host}" ] && [ "${host##b}" != "${host}" ]; then
  ncu=/usr/local/cuda-11.0/bin/ncu
  ###nsys=/usr/local/cuda-10.1/bin/nsys
  ###nsys=/usr/local/cuda-10.2/bin/nsys
  nsys=/cvmfs/sft.cern.ch/lcg/releases/cuda/11.0RC-d9c38/x86_64-centos7-gcc62-opt/bin/nsys
  ncugui="Launch the Nsight Compute GUI from Windows"
  nsysgui="Launch the Nsight System GUI from Windows"
fi

# Set the ncu sampling period (default is auto)
# The value is in the range [0..31], the actual period is 2**(5+value) cycles. 
###ncu="${ncu} --sampling-interval 0"  # MAX sampling frequency
###ncu="${ncu} --sampling-interval 31" # MIN sampling frequency

# See https://developer.nvidia.com/blog/using-nsight-compute-to-inspect-your-kernels/
# These used to be called gld_transactions and global_load_requests
# See also https://docs.nvidia.com/nsight-compute/2019.5/NsightComputeCli/index.html#nvprof-metric-comparison
# See also https://stackoverflow.com/questions/60535867
metrics=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

# Add registers
metrics+=,launch__registers_per_thread

# GUI analysis
if [ "$tag" != "nogui" ]; then

  if [ "$label" == "" ]; then
    echo "ERROR! You must specify a label"
    usage
  fi

  arg1=$(echo $args | cut -d' ' -f1)
  arg2=$(echo $args | cut -d' ' -f2)
  arg3=$(echo $args | cut -d' ' -f3)
  
  if [ "${host%%raplab*}" != "${host}" ]; then
    logs=logs_raplab
  elif [ "${host%%cern.ch}" != "${host}" ] && [ "${host##b}" != "${host}" ]; then
    logs=logs_lxbatch
  else
    logs=logs
  fi
  trace=$logs/eemumuAV_${tag}_`date +%m%d_%H%M`_b${arg1}_t${arg2}_i${arg3}
  if [ "$label" != "" ]; then trace=${trace}_${label}; fi
  
  echo
  echo "PROFILING: ${cmd}"
  echo "OUTPUT: ${trace}.*"
  echo
  
  \rm -f ${trace}.*
  
  hostname > ${trace}.txt
  echo "nproc=$(nproc)" >> ${trace}.txt
  echo >> ${trace}.txt
  ( time ${cmd} ) 2>&1 | tee -a ${trace}.txt
  nvidia-smi -q -d CLOCK >> ${trace}.txt
  
  if [ "$tag" == "cu" ]; then
    echo
    echo "${ncu} --set full --metrics ${metrics} -o ${trace} ${cmd}"
    echo
    ${ncu} --set full --metrics ${metrics} -o ${trace} ${cmd}
  fi
  echo
  echo "${nsys} profile -o ${trace} ${cmd}"
  echo
  ${nsys} profile -o ${trace} ${cmd}
  echo ""
  echo "TO ANALYSE TRACE FILES:"
  echo "  ${ncugui}"
  echo "  ${nsysgui}"
  
# NO-GUI analysis
else

  echo
  echo "PROFILING: ${cmd}"
  echo "${ncu} --metrics ${metrics} ${cmd}"
  echo
  ${ncu} --metrics ${metrics} ${cmd}

fi
