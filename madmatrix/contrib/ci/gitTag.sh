#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2024) for the MG5aMC CUDACPP plugin (based on earlier work for the HEP_OSlibs project).
# Further modified by: A. Valassi (2024).
# Integrated with the MadGraph7 project in Feb 2026.

set -e # fail on error

cd $(dirname $0)
topdir=$(cd ../..; pwd) # the top directory in this local repo
scr=$(basename $(cd ..; pwd))/$(basename $(pwd))/$(basename $0) # the name of this script in the git repo

skipFetch=0
###skipFetch=1 # FOR DEBUGGING!

# Protect non-prerelease tags from deletion?
PROTECTTAGS=1
###PROTECTTAGS=0 # WARNING! USE WITH CARE...

# Usage
function usage()
{
  echo "Usage (1): $0 [-f] <tagsuffix>"
  echo "Creates a new version tag (from the HEAD of the local branch) and pushes it to the remote repository"
  echo "Valid formats for <tagsuffix> are 'n.nn.nn' or 'n.nn.nn_txt' where txt only contains letters or digits)"
  echo "Version number must match the (n1,n2,n3) specified with single digits in the CUDACPP_OUTPUT/__init__.py file"
  echo "For release tags (no trailing '_txt'), the github CI will then create also a running tag with '_latest' suffix"
  echo "Use the -f option to delete and recreate a version tag that already exists"
  echo ""
  echo "Usage (2): $0 -l"
  echo "Shows all available git tags (these are not necessarily github releases)"
  exit 1
}

# Command line arguments
force=""
if [ "$1" == "-f" ]; then
  force=$1
  shift
fi
if [ "$1" == "" ] || [ "$2" != "" ]; then
  usage
elif [ "$1" == "-l" ]; then
  tagsuffix=""
elif [ "${1/-}" != "${1}" ]; then # exclude immediately tagsuffix beginning with "-" (print usage on '-h')
  usage
else
  tagsuffix="$1"
  shift
fi

# Check that all git submodules have been updated
if ! git submodule status | grep '^ ' > /dev/null; then
  echo "ERROR! There are git submodules that need to be updated"
  git submodule status
  exit 1
fi

# Check that all changes are committed to git (except at most for this script only in the '-l' mode)
if [ "$(git status --porcelain | grep -v ^?? | grep -v ' M MG5aMC/mg5amcnlo')" != "" ]; then
  if [ "$tagsuffix" != "" ] || [ "$(git status --porcelain | grep -v ^??)" != " M ${scr}" ]; then
    echo "ERROR! There are local changes not committed to git yet"
    exit 1
  fi
fi

# Check that the local and remote branch are in sync
# See https://stackoverflow.com/a/3278427, https://stackoverflow.com/a/17938274
if [ "${skipFetch}" == "0" ]; then
  git fetch
fi
if [ "$(git rev-parse @{0})" != "$(git rev-parse @{u})" ]; then
  echo "ERROR! You need to git push or git pull"
  git status
  exit 1
fi

# Determine the local and remote git branches
brnl=`git branch | grep ^* | cut -d ' ' -f2`
brnr=`git branch -vv | grep ^* | awk '{print $4}' | sed 's|^\[||' | sed 's|]$||'`
echo "INFO: git local  branch is '${brnl}'"
echo "INFO: git remote branch is '${brnr}'"
###if [ "${brnl}" != "master" ]; then
###  echo "ERROR! Invalid local branch ${brn}"
###  exit 1
###fi
if [ "${brnr#origin/}" == "${brnr}" ]; then
  echo "ERROR! Invalid remote branch ${brnr} does not track origin"
  exit 1
fi

# Determine the remote git origin and the corresponding repo name
# Stop if the remote git origin is not at github.com
url=`git config --get remote.origin.url`
echo "INFO: git remote is        '${url}'"
url=${url%.git}
prj=${url##*github.com:}
if [ "${url}" == "${prj}" ]; then
  echo "ERROR! git remote does not seem to be at github.com"
  exit 1
fi
echo "INFO: git repo is          '${prj}'"
prjown=${prj%/*}
prjnam=${prj#*/}
echo "INFO: git repo owner is    '${prjown}'"
echo "INFO: git repo name  is    '${prjnam}'"

# Check the repo name
if [ "${prjnam}" != "madgraph4gpu" ]; then # TEMPORARY! this will change eventually...
  echo "ERROR! Invalid repo name '${prjnam}' (expect 'madgraph4gpu')"
  exit 1
fi

# Check the repo owner
# Determine the tag prefix used for all tags handled by this script
if [ "${prjown}" == "madgraph5" ]; then # TEMPORARY! this will change eventually...
  PREFIX=cudacpp_for
else
  PREFIX=${prjown}_cudacpp_for
fi
echo "INFO: tag prefix for repo owner '${prjown}' is '${PREFIX}'"

# Remove local git tags that are no longer on the remote repo
# See https://stackoverflow.com/a/27254532
if [ "${skipFetch}" == "0" ]; then
  git fetch --prune origin +refs/tags/*:refs/tags/*
fi
echo ""

# Retrieve the list of existing tags
existing_tags=$(git tag -l | grep ^${PREFIX} || true)

#
# OPTION 1 - LIST TAGS
#
if [ "$tagsuffix" == "" ]; then

  # See https://stackoverflow.com/questions/13208734
  echo "INFO: list existing tags starting with '${PREFIX}' (detailed list)"
  ###git --no-pager log --oneline --decorate --tags --no-walk
  if [ "${existing_tags}" == "" ]; then echo "[None]"; fi
  for tag in ${existing_tags}; do
    echo "--------------------------------------------------------------------------------"
    echo "$tag"
    echo "--------------------------------------------------------------------------------"
    ###git --no-pager for-each-ref --format="%(refname)" refs/tags/${tag}
    git --no-pager for-each-ref --format="TagDate:       %(creatordate)" refs/tags/${tag}
    git --no-pager log -n1 ${tag} --pretty=format:'commit         %h%nAuthor:        %an%nAuthorDate:    %ad%nMessage:       "%s"%n'
    echo ""
  done

#
# OPTION 2 - CREATE NEW TAGS
#
else

  # Determine cudacpp_version (as in archiver.sh)
  echo "INFO: determine cudacpp and mg5amc versions"
  cudacpp_major=$(cat ${topdir}/epochX/cudacpp/CODEGEN/PLUGIN/CUDACPP_SA_OUTPUT/__init__.py | grep __version__ | sed -r 's/(.*=|\(|\)|,)/ /g' | awk '{print $1}')
  cudacpp_minor=$(cat ${topdir}/epochX/cudacpp/CODEGEN/PLUGIN/CUDACPP_SA_OUTPUT/__init__.py | grep __version__ | sed -r 's/(.*=|\(|\)|,)/ /g' | awk '{print $2}')
  cudacpp_patch=$(cat ${topdir}/epochX/cudacpp/CODEGEN/PLUGIN/CUDACPP_SA_OUTPUT/__init__.py | grep __version__ | sed -r 's/(.*=|\(|\)|,)/ /g' | awk '{print $3}')
  if [ ${cudacpp_major} -lt 0 ] || [ ${cudacpp_major} -gt 99 ]; then echo "ERROR! cudacpp_major is not in the [0,99] range"; exit 1; fi
  if [ ${cudacpp_minor} -lt 0 ] || [ ${cudacpp_minor} -gt 99 ]; then echo "ERROR! cudacpp_minor is not in the [0,99] range"; exit 1; fi
  if [ ${cudacpp_patch} -lt 0 ] || [ ${cudacpp_patch} -gt 99 ]; then echo "ERROR! cudacpp_patch is not in the [0,99] range"; exit 1; fi
  cudacpp_version=$(printf "%1d.%02d.%02d" ${cudacpp_major} ${cudacpp_minor} ${cudacpp_patch})
  echo "> cudacpp_version = $cudacpp_version"

  # Determine mg5_version (as in HEPToolInstaller.py)
  mg5VERSION=${topdir}/MG5aMC/mg5amcnlo/VERSION 
  if [ ! -f ${mg5VERSION} ]; then echo "ERROR! File ${mg5VERSION} not found"; exit 1; fi 
  mg5_version=$(python3 -c "import re; print(re.findall(r'version\s*=\s*([\.\d]*)','$(cat ${mg5VERSION} | grep ^version)')[0])")
  echo "> mg5_version_current = $mg5_version"
  echo ""

  # Validate tagsuffix
  # Expect 1-digit major and 2-digit minor and patch versions with an optional prerelease suffix ('n.nn.nn[_txt]')
  # See https://stackoverflow.com/questions/229551
  # See https://stackoverflow.com/questions/55486225
  echo "INFO: validate tag suffix '${tagsuffix}'"
  if [[ $tagsuffix == *" "* ]]; then
    echo "ERROR! Invalid tag suffix '${tagsuffix}' (no spaces allowed)"; exit 1
  fi
  if [ `python3 -c "import re; print(re.match('^[0-9]\.[0-9]{2}\.[0-9]{2}(|_[0-9a-z]+)$','${tagsuffix}') is not None)"` == "False" ]; then
    echo "ERROR! Invalid tag suffix '${tagsuffix}' (valid formats are 'n.nn.nn' or 'n.nn.nn_txt' where txt only contains letters or digits)"; exit 1
  fi
  if [ "${tagsuffix/${cudacpp_version}}" == "${tagsuffix}" ]; then
    echo "ERROR! Invalid tag suffix '${tagsuffix}' (the version number does not match the cudacpp_version)"; exit 1
  fi
  echo ""

  # List the existing tags
  echo "INFO: list existing tags starting with '${PREFIX}' (short list)"
  if [ "${existing_tags}" == "" ]; then echo "> [None]"; fi
  for tag in ${existing_tags}; do
    echo "> $tag"
  done
  echo ""

  # Build the version tag and running tag names
  versTAG=${PREFIX}${mg5_version}_v${tagsuffix}
  runnTAG=${PREFIX}${mg5_version}_latest
  echo "INFO: will create version tag ${versTAG}"

  # Check if the version tag is a pre-release tag 'n1.n2.n3_txt
  if [ "${tagsuffix%_*}" != "${tagsuffix}" ]; then
    prerelease=1
    echo "INFO: tag ${versTAG} is a pre-release tag: running tag ${runnTAG} will NOT be updated by the CI"
  else
    prerelease=0
    echo "INFO: tag ${versTAG} is a release tag: running tag ${runnTAG} will be updated by the CI"
  fi

  # Check if the version tag already exists
  for tag in ${existing_tags}; do
    if [ "${versTAG}" == "${tag}" ]; then
      if [ "${prerelease}" == "0" ] && [ "${PROTECTTAGS}" == "1" ]; then
        echo "ERROR! Tag ${tag} already exists and is not a pre-release tag: you must unprotect it and delete it from the web interface"
        exit 1
      elif [ "${force}" != "-f" ]; then
        echo "ERROR! Tag ${tag} already exists: use the -f option to recreate it"
        exit 1
      else
        echo "WARNING! Tag ${tag} already exists and the -f option was specified: it will be recreated"
      fi
    fi
  done
  echo ""

  # Create the version (~permanent) tag
  # (optionally use '-f' to replace any previously existing tag with the same name)
  echo "INFO: create version tag ${versTAG}"
  git tag ${force} ${versTAG} -m "Version tag ${versTAG}" -m "Tag created on $(date)"

  # Push the tag to the remote repository
  # (use '-f' to replace any previously existing tag with the same name)
  echo "INFO: push tag to the remote repository"
  git push -f --tags

fi
