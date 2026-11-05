#!/bin/bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2024) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2024).
# Integrated with the MadGraph7 project in Feb 2026.
# Path to the top directory of madgraphgpu
# In the CI this would be simply $(pwd), but allow the script to be run also outside the CI
echo "Executing $0 $*"
topdir=$(cd $(dirname $0)/../..; pwd)

# Check that all git submodules have been updated
cd ${topdir}
if ! git submodule status | grep '^ ' > /dev/null; then
  echo "ERROR! There are git submodules that need to be updated"
  git submodule status
  exit 1
fi
mg5_commit_current=$(git submodule status | awk '/ MG5aMC\/mg5amcnlo /{print substr($1,0,7)}')

# Create a temporary directory and a VERSION.txt file
cd ${topdir}/epochX/cudacpp/CODEGEN/PLUGIN/CUDACPP_SA_OUTPUT
tmpdir=$(mktemp -d)
outdir=${tmpdir}/CUDACPP_OUTPUT
mkdir ${outdir}
outfile=${outdir}/VERSION.txt
touch ${outfile}
dateformat='%Y-%m-%d_%H:%M:%S UTC'
cudacpp_major=$(cat __init__.py | grep __version__ | sed -r 's/(.*=|\(|\)|,)/ /g' | awk '{print $1}')
cudacpp_minor=$(cat __init__.py | grep __version__ | sed -r 's/(.*=|\(|\)|,)/ /g' | awk '{print $2}')
cudacpp_patch=$(cat __init__.py | grep __version__ | sed -r 's/(.*=|\(|\)|,)/ /g' | awk '{print $3}')
###echo "(From CUDACPP_OUTPUT/__init__.py)"
###echo "cudacpp (major, minor, patch) = ( ${cudacpp_major}, ${cudacpp_minor}, ${cudacpp_patch} )"
if [ ${cudacpp_major} -lt 0 ] || [ ${cudacpp_major} -gt 99 ]; then echo "ERROR! cudacpp_major is not in the [0,99] range"; exit 1; fi
if [ ${cudacpp_minor} -lt 0 ] || [ ${cudacpp_minor} -gt 99 ]; then echo "ERROR! cudacpp_minor is not in the [0,99] range"; exit 1; fi
if [ ${cudacpp_patch} -lt 0 ] || [ ${cudacpp_patch} -gt 99 ]; then echo "ERROR! cudacpp_patch is not in the [0,99] range"; exit 1; fi
cudacpp_version=$(printf "%1d.%02d.%02d" ${cudacpp_major} ${cudacpp_minor} ${cudacpp_patch})
echo "(From CUDACPP_OUTPUT/__init__.py)" >> ${outfile}
echo "cudacpp_version              = ${cudacpp_version}" >> ${outfile}
echo "mg5_version_minimal          = $(cat __init__.py | awk '/minimal_mg5amcnlo_version/{print $3}'  | sed 's/(//' | sed 's/)//' | sed 's/,/./g')" >> ${outfile}
echo "mg5_version_latest_validated = $(cat __init__.py | awk '/latest_validated_version/{print $3}'  | sed 's/(//' | sed 's/)//' | sed 's/,/./g')" >> ${outfile}
echo "" >> ${outfile}
echo "(From MG5AMC/mg5amcnlo)" >> ${outfile}
echo "mg5_version_current          = $(cat ../../../../../MG5aMC/mg5amcnlo/VERSION | awk '/version =/{print $3}' | sed -r 's/[^0-9.]//g')" >> ${outfile}
echo "mg5_commit_current           = ${mg5_commit_current}" >> ${outfile}
echo "" >> ${outfile}
echo "TARBALL DATE:  $(date -u +"${dateformat}")" >> ${outfile}
echo "" >> ${outfile}
TZ=UTC git --no-pager log -n1 --date=format-local:"${dateformat}" --pretty=format:'commit         %h%nAuthor:        %an%nAuthorDate:    %ad%nCommitter:     %cn%nCommitterDate: %cd%nMessage:       "%s"%n' >> ${outfile}
python3 -c 'print("="*132)'; cat ${outfile}; python3 -c 'print("="*132)'
cp ${outfile} ${topdir}
echo "VERSION.txt file available on ${topdir}/$(basename ${outfile})"

# Copy all relevant plugin files to the temporary directory
cd ${topdir}/epochX/cudacpp/CODEGEN/PLUGIN/CUDACPP_SA_OUTPUT
for file in $(git ls-tree --name-only HEAD -r); do
  if [ "${file/acceptance_tests}" != "${file}" ]; then continue; fi # acceptance_tests are not needed for code generation
  mkdir -p ${outdir}/$(dirname ${file})
  cp -dp ${file} ${outdir}/${file} # preserve symlinks for AUTHORS, COPYING, COPYING.LESSER and COPYRIGHT
done

# Create the tgz archive
outtgz=cudacpp.tar.gz
cd ${tmpdir}
tar -czf ${outtgz} CUDACPP_OUTPUT
mv ${outtgz} ${topdir}
echo "Archive available on ${topdir}/${outtgz}"
