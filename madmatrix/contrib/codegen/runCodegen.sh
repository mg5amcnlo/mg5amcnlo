#! /bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: S. Hageboeck (Sep 2023) for the MG5aMC CUDACPP plugin.
# Further modified by: S. Hageboeck, A. Valassi (2023).
# Integrated with the MadGraph7 project in Feb 2026.

set -e

mg5=../../MG5aMC/mg5amcnlo/bin/mg5_aMC
pluginLink=../../MG5aMC/mg5amcnlo/PLUGIN/CUDACPP_OUTPUT

if [ $# == 0 ]; then
  echo "Usage ${0} <process directory to generate> <...>"
  exit 0
fi

stageChanges=true
checkDirectoryClean=true
removeGeneratedCode=true
dry_run=false

stageChanges () {
  process=$1
  process_tmp=$2
  TOPDIR=$(git rev-parse --show-toplevel)

  # Stage all modified files
  git status --porcelain ${process} | awk "/^.[MTADRU]/{ print(\"${TOPDIR}/\"\$2); }" | xargs git add

  # Stage new files
  git status --porcelain ${process} | awk "/^\?\?/{ print(\$2); }" | while read file; do
    if [ -f ${TOPDIR}/${file/${process}/"${process_tmp}/"} ]; then
      echo "New file from code generator adding to git: ${TOPDIR}/${file}"
      git add ${TOPDIR}/${file}
    fi
  done

  # Clean up files that are no longer generated
  while read file; do
    if [[ ! -f ${process_tmp}/${file#*/} && ! ${file} == *.gitignore && ! "${file}" == *mg5.in ]]; then
      FILES_TO_CLEAN="${FILES_TO_CLEAN} ${file}"
    fi
  done < <(git ls-files ${process})
}

while (( "$#" )); do
  case "$1" in
    --no-stage)
      stageChanges=false
      ;;
    --keep|-k)
      removeGeneratedCode=false
      ;;
    --force|-f)
      checkDirectoryClean=false
      ;;
    --dry-run|-n)
      stageChanges=false
      checkDirectoryClean=false
      dry_run=true
      ;;
    *)
      PROCESSES="${PROCESSES} $1"
      ;;
  esac
  shift
done


if [ ! -x "${mg5}" ]; then
  echo "${mg5} not found. Please change into the cudacpp directory or init the mg5amc submodule."
  exit 1
fi


if [ ! -L ${pluginLink} ]; then
  echo "The cudacpp plugin needs to be linked to ${pluginLink}."
  exit 1
fi

for process in ${PROCESSES}; do
  ProcessFile=${process}/mg5.in
  if [ ! -e "${ProcessFile}" ]; then
    echo "MG5 input in ${ProcessFile} not found. Please create this file with all madgraph commands to generate this process."
    exit 1
  fi

  if ${checkDirectoryClean}; then
    if git status --porcelain ${process} | grep -q "^ [MTADRU]"; then
      git status ${process}
      echo -e "\n${process} has uncommitted changes. Please commit or stash them before running the code generator. Override with --force / -f."
      exit 2
    fi
  fi

  process_tmp=${process%/*}_gen
  processFile_tmp=$(mktemp /tmp/mg5_XXXXXXX)
  echo "Generating ${process} into ${process_tmp} ..."
  sed -E "s/(output [[:graph:]]+) [[:graph:]]+ (.*)/\1 ${process_tmp} \2/" < ${ProcessFile} | tee ${processFile_tmp}
  echo ""
  ${mg5} < ${processFile_tmp} || { echo -e "\n\nError generating ${process}"; exit 1; }

  echo -e "\n\nMadgraph generation completed, syncing files ..."
  if $dry_run; then
    diff -qr ${process} ${process_tmp}
  else
    rsync -ra --exclude HTML/ ${process_tmp}/* ${process}
  fi

  if ${stageChanges}; then
    echo -e "Staging changes to git ...\n"
    stageChanges ${process} ${process_tmp}
  fi

  if ${removeGeneratedCode}; then
    rm -r ${process_tmp}
  fi
done

if [ -n "${FILES_TO_CLEAN}" ]; then
  echo -e "\n\nThe following files were not created by the code generator. Consider removing these from git:"
  echo "${FILES_TO_CLEAN}"
fi
