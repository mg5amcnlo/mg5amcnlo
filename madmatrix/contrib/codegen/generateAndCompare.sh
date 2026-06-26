#!/usr/bin/env bash
# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2021) for the MG5aMC CUDACPP plugin.
# Further modified by: A. Valassi (2021-2025).
# Integrated with the MadGraph7 project in Feb 2026.

set -e # fail on error

#--------------------------------------------------------------------------------------

function codeGenAndDiff()
{
  proc=$1
  cmdext="$2"
  if [ "$3" != "" ]; then echo -e "INTERNAL ERROR!\nUsage: ${FUNCNAME[0]} <proc> [<cmd>]"; exit 1; fi
  # Process-dependent hardcoded configuration
  echo -e "================================================================"
  cmd=
  case "${proc}" in
    ee_mumu)
      cmd="generate e+ e- > mu+ mu-"
      ;;
    gg_tt)
      cmd="generate g g > t t~"
      ;;
    gg_ttg)
      cmd="generate g g > t t~ g"
      ;;
    gg_ttgg)
      cmd="generate g g > t t~ g g"
      ;;
    gg_ttggg)
      cmd="generate g g > t t~ g g g"
      ;;
    gg_ttgggg)
      cmd="generate g g > t t~ g g g g"
      ;;
    gg_tt01g)
      cmd="generate g g > t t~; add process g g > t t~ g"
      ;;
    gq_ttq)
      cmd="define q = u c d s u~ c~ d~ s~; generate g q > t t~ q"
      ;;
    gu_ttu)
      cmd="generate g u > t t~ u"
      ;;
    gg_uu)
      cmd="generate g g > u u~"
      ;;
    gq_ttllq)
      cmd="define q = u c d s u~ c~ d~ s~; generate g q > t t~ l- l+ q"
      ;;
    pp_tt)
      cmd="generate p p > t t~"
      ;;
    pp_tttt)
      cmd="generate p p > t t~ t t~"
      ;;
    pp_tt012j)
      cmd="define j = p
      generate p p > t t~ @0
      add process p p > t t~ j @1
      add process p p > t t~ j j @2"
      ;;
    pp_ttW) # TEMPORARY! until no_b_mass #695 and/or #696 are fixed
      cmd="define p = p b b~
      define j = p
      define w = w+ w- # W case only
      generate p p > t t~ w @0
      add process p p > t t~ w j @1"
      ;;
    pp_ttZ) # TEMPORARY! until no_b_mass #695 and/or #696 are fixed
      cmd="define p = p b b~
      define j = p
      generate p p > t t~ z @0
      add process p p > t t~ z j @1"
      ;;
    uu_dd)
      cmd="generate u u~ > d d~"
      ;;
    uu_tt)
      cmd="generate u u~ > t t~"
      ;;
    uu_ttg)
      cmd="generate u u~ > t t~ g"
      ;;
    bb_tt)
      cmd="generate b b~ > t t~"
      ;;
    nobm_gg_tt)
      cmd="import model sm-no_b_mass
      generate g g > t t~"
      ;;
    nobm_pp_ttW)
      cmd="import model sm-no_b_mass
      define p = p b b~
      define j = p
      define w = w+ w- # W case only
      generate p p > t t~ w @0
      add process p p > t t~ w j @1"
      ;;
    nobm_pp_ttZ)
      cmd="import model sm-no_b_mass
      define p = p b b~
      define j = p
      generate p p > t t~ z @0
      add process p p > t t~ z j @1"
      ;;
    nobm_gu_ttxwpd)
      cmd="import model sm-no_b_mass
      generate g u > t t~ w+ d"
      ;;
    nobm_pp_eejjj)
      cmd="import model sm-no_b_mass
      define p = p b b~
      define j = p
      generate p p > e+ e- j j j @3"
      ;;
    nobm_pp_eejjjj)
      cmd="import model sm-no_b_mass
      define p = p b b~
      define j = p
      generate p p > e+ e- j j j j @4"
      ;;
    gg_eegguu)
      cmd="generate g g > e+ e- g g u u~"
      ;;
    #nobm_gg_eegguu) # essentially the same code as gg_eegguu
    #  cmd="import model sm-no_b_mass
    #  define p = p b b~
    #  define j = p
    #  generate g g > e+ e- g g u u~"
    #  ;;
    loop_nobm_gg_tt)
      cmd="import model loop_sm-no_b_mass
      generate g g > t t~"
      ;;
    loop_nobm_pp_ttW)
      cmd="import model loop_sm-no_b_mass
      define p = p b b~
      define j = p
      define w = w+ w- # W case only
      generate p p > t t~ w @0
      add process p p > t t~ w j @1"
      ;;
    loop_nobm_pp_ttZ)
      cmd="import model loop_sm-no_b_mass
      define p = p b b~
      define j = p
      generate p p > t t~ z @0
      add process p p > t t~ z j @1"
      ;;
    heft_gg_h)
      cmd="set auto_convert_model T; import model heft; generate g g > h"
      ;;
    gg_bb)
      cmd="generate g g > b b~" # for comparison: 3 SM diagrams
      ;;
    heft_gg_bb)
      ###cmd="set auto_convert_model T; import model heft; generate g g > b b~" # 3 SM diagrams (as in SM gg_bb)
      cmd="set auto_convert_model T; import model heft; generate g g > b b~ HIW<=1" # 4 diagrams (3 SM plus 1 HEFT): fix #828
      ;;
    heft_gg_h_bb)
      cmd="set auto_convert_model T; import model heft; generate g g > h > b b~" # 1 HEFT diagram
      ;;
    smeft_gg_tttt)
      cmd="set auto_convert_model T; import model SMEFTsim_topU3l_MwScheme_UFO -massless_4t; generate g g > t t~ t t~"
      ;;
    susy_gg_tt)
      cmd="import model MSSM_SLHA2; generate g g > t t~"
      ;;
    susy_gg_tttt)
      cmd="import model MSSM_SLHA2; generate g g > t t~ t t~"
      ;;
    susy_gq_ttq)
      cmd="import model MSSM_SLHA2; define q = u c d s u~ c~ d~ s~; generate g q > t t~ q"
      ;;
    susy_gq_ttllq)
      cmd="import model MSSM_SLHA2; define q = u c d s u~ c~ d~ s~; generate g q > t t~ l- l+ q"
      ;;
    ###susy_gu_ttllu)
    ###  cmd="import model MSSM_SLHA2; generate g u > t t~ l- l+ u"
    ###  ;;
    ###susy_gux_ttllux)
    ###  cmd="import model MSSM_SLHA2; generate g u~ > t t~ l- l+ u~"
    ###  ;;
    ###susy_gd_ttlld)
    ###  cmd="import model MSSM_SLHA2; generate g d > t t~ l- l+ d"
    ###  ;;
    ###susy_gdx_ttlldx)
    ###  cmd="import model MSSM_SLHA2; generate g d~ > t t~ l- l+ d~"
    ###  ;;
    susy_gg_gogo)
      cmd="import model MSSM_SLHA2; generate g g > go go"
      ;;
    susy_gg_t1t1)
      cmd="import model MSSM_SLHA2; generate g g > t1 t1~"
      ;;
    susy_gg_ulul)
      cmd="import model MSSM_SLHA2; generate g g > ul ul~"
      ;;
    atlas)
      cmd="import model sm-no_b_mass
      define p = g u c d s b u~ c~ d~ s~ b~
      define j = g u c d s b u~ c~ d~ s~ b~
      generate p p > t t~
      add process p p > t t~ j
      add process p p > t t~ j j"
      ;;
    cms)
      cmd="define vl = ve vm vt
      define vl~ = ve~ vm~ vt~
      define ell+ = e+ mu+ ta+
      define ell- = e- mu- ta-
      generate p p > ell+ ell- @0"
      ;;
    pp_ttjjj) # From Sapta for CMS
      cmd="define p = g u c d s u~ c~ d~ s~
      define j = g u c d s u~ c~ d~ s~
      define l+ = e+ mu+
      define l- = e- mu-
      define vl = ve vm vt
      define vl~ = ve~ vm~ vt~
      generate p p > t t~ j j j"
      ;;
    *)
      if [ "$cmdext" == "" ]; then
        echo "ERROR! Unknown process '$proc' and no external process specified with '-c <cmd>'"
        usage
      fi
      ;;
  esac
  if [ "$cmdext" != "" ]; then
    if [ "$cmd" != "" ]; then 
      echo "ERROR! Invalid option '-c <cmd>' to predefined process '$proc'"
      echo "Predefined cmd='$cmd'"
      echo "User-defined cmd='$cmdext'"
      usage
    else
      cmd="$cmdext"
    fi
  fi
  echo -e "\n+++ Generate code for '$proc'\n"
  ###exit 0 # FOR DEBUGGING
  # Vector size for mad/madonly meexporter (VECSIZE_MEMMAX)
  vecsize=32 # NB THIS IS NO LONGER IGNORED (but will eventually be tunable via runcards)
  # Generate code for the specific process
  pushd $MG5AMC_HOME >& /dev/null
  mkdir -p ../TMPOUT
  outproc=../TMPOUT/CODEGEN_${OUTBCK}_${proc}
  if [ "${SCRBCK}" == "gridpack" ] && [ "${UNTARONLY}" == "1" ]; then
    ###echo -e "WARNING! Skip generation of gridpack.tar.gz (--nountaronly was not specified)\n"
    echo "ERROR! gridpack mode is no longer supported by this script!"; exit 1
  else
    \rm -rf ${outproc} ${outproc}.* ${outproc}_*
    if [ "${HELREC}" == "0" ]; then
      helrecopt="--hel_recycling=False"
    else
      helrecopt=
    fi
    echo "set stdout_level DEBUG" >> ${outproc}.mg # does not help (log is essentially identical) but add it anyway
    echo "set zerowidth_tchannel F" >> ${outproc}.mg # workaround for #476: do not use a zero top quark width in fortran (~E-3 effect on physics)
    echo "${cmd}" | sed "s/;/\n/g" | sed "s/ *$//" | sed "s/^ *//" >> ${outproc}.mg
    if [ "${SCRBCK}" == "gridpack" ]; then # $SCRBCK=$OUTBCK=gridpack
      ###echo "output ${outproc} ${helrecopt}" >> ${outproc}.mg
      ###echo "launch" >> ${outproc}.mg
      ###echo "set gridpack True" >> ${outproc}.mg
      ###echo "set ebeam1 750" >> ${outproc}.mg
      ###echo "set ebeam2 750" >> ${outproc}.mg
      echo "ERROR! gridpack mode is no longer supported by this script!"; exit 1
    elif [ "${SCRBCK}" == "alpaka" ]; then # $SCRBCK=$OUTBCK=alpaka
      ###echo "output standalone_${SCRBCK}_cudacpp ${outproc}" >> ${outproc}.mg
      echo "ERROR! alpaka mode is no longer supported by this script!"; exit 1
    elif [ "${OUTBCK}" == "madnovec" ]; then # $SCRBCK=cudacpp and $OUTBCK=madnovec
      echo "output madevent ${outproc} ${helrecopt}" >> ${outproc}.mg
    elif [ "${OUTBCK}" == "madonly" ]; then # $SCRBCK=cudacpp and $OUTBCK=madonly
      echo "output madevent ${outproc} ${helrecopt} --vector_size=${vecsize}" >> ${outproc}.mg
    elif [ "${OUTBCK}" == "mad" ]; then # $SCRBCK=cudacpp and $OUTBCK=mad
      echo "output madevent_simd ${outproc} ${helrecopt} --vector_size=${vecsize} " >> ${outproc}.mg
    elif [ "${OUTBCK}" == "madcpp" ]; then # $SCRBCK=cudacpp and $OUTBCK=madcpp
      echo "output madevent_simd ${outproc} ${helrecopt} --vector_size=32" >> ${outproc}.mg
    elif [ "${OUTBCK}" == "madgpu" ]; then # $SCRBCK=cudacpp and $OUTBCK=madgpu
      echo "output madevent ${outproc} ${helrecopt} --vector_size=32 --me_exporter=standalone_gpu" >> ${outproc}.mg
    else # $SCRBCK=cudacpp and $OUTBCK=cudacpp, cpp or gpu
      echo "output standalone_${OUTBCK} ${outproc}" >> ${outproc}.mg
    fi
    echo "--------------------------------------------------"
    cat ${outproc}.mg
    echo -e "--------------------------------------------------\n"
    PYTHONPATHOLD=$PYTHONPATH
    export PYTHONPATH=..:$PYTHONPATHOLD
    ###{ strace -f -o ${outproc}_strace.txt python3 ./bin/mg5_aMC -m CUDACPP_OUTPUT ${outproc}.mg ; } >& ${outproc}_log.txt
    { time python3 ./bin/mg5_aMC -m CUDACPP_OUTPUT ${outproc}.mg || true ; } >& ${outproc}_log.txt
    export PYTHONPATH=$PYTHONPATHOLD
    cat ${outproc}_log.txt | egrep -v '(Crash Annotation)' > ${outproc}_log.txt.new # remove firefox 'glxtest: libEGL initialize failed' errors
    \mv ${outproc}_log.txt.new ${outproc}_log.txt
  fi
  echo "Code generation completed in $SECONDS seconds" >> ${outproc}_log.txt
  # Check the code generation log for errors 
  if [ -d ${outproc} ] && ! grep -q "Please report this bug" ${outproc}_log.txt; then
    ###cat ${outproc}_log.txt; exit 0 # FOR DEBUGGING
    cat ${MG5AMC_HOME}/${outproc}_log.txt | { egrep 'INFO: (Try|Creat|Organiz|Process)' || true; }
    echo ""
    cat ${MG5AMC_HOME}/${outproc}_log.txt | grep 'Code generation'
  else
    echo "*** ERROR! Code generation failed"
    cat ${MG5AMC_HOME}/${outproc}_log.txt
    echo "*** ERROR! Code generation failed"
    exit 1
  fi
  # Add a workaround for https://github.com/oliviermattelaer/mg5amc_test/issues/2 (THIS IS ONLY NEEDED IN THE MADGRAPH4GPU GIT REPO)
  # (NEW SEP2024: move this _before_ `madevent treatcards param` as otherwise param_card.inc changes during the build of the code)
  if [ "${OUTBCK}" == "madnovec" ] || [ "${OUTBCK}" == "madonly" ] || [ "${OUTBCK}" == "mad" ] || [ "${OUTBCK}" == "madcpp" ] || [ "${OUTBCK}" == "madgpu" ]; then
    cat ${outproc}/Cards/ident_card.dat | head -3 > ${outproc}/Cards/ident_card.dat.new
    cat ${outproc}/Cards/ident_card.dat | tail -n+4 | sort >> ${outproc}/Cards/ident_card.dat.new
    \mv ${outproc}/Cards/ident_card.dat.new ${outproc}/Cards/ident_card.dat
  fi
  # Patches moved here from patchMad.sh after Olivier's PR #764 (THIS IS ONLY NEEDED IN THE MADGRAPH4GPU GIT REPO)  
  if [ "${OUTBCK}" == "mad" ]; then
    # Force the use of strategy SDE=1 in multichannel mode (see #419)
    sed -i 's/2  = sde_strategy/1  = sde_strategy/' ${outproc}/Cards/run_card.dat
    # Force the use of VECSIZE_MEMMAX=16384 (OLD CODE BEFORE JUNE24 WARP_SIZE)
    ###sed -i 's/16 = vector_size/16384 = vector_size/' ${outproc}/Cards/run_card.dat 
    # Force the use of WARP_SIZE=32 and NB_WARP=512 i.e. VECSIZE_MEMMAX=16384 (NEW CODE AFTER JUNE24 WARP_SIZE: NEEDS FIX FOR CRASH #885)
    sed -i 's/16 = vector_size/32 = vector_size/' ${outproc}/Cards/run_card.dat 
    sed -i 's/1 = nb_warp/512 = nb_warp/' ${outproc}/Cards/run_card.dat 
    # Force the use of WARP_SIZE=32 and NB_WARP=2 i.e. VECSIZE_MEMMAX=64 is identical to (TEMPORARY) VECSIZE_USED=64 in tmad tests (workaround for crash #885)
    ###sed -i 's/16 = vector_size/32 = vector_size/' ${outproc}/Cards/run_card.dat 
    ###sed -i 's/1 = nb_warp/2 = nb_warp/' ${outproc}/Cards/run_card.dat 
    # Force the use of fast-math in Fortran builds
    sed -i 's/-O = global_flag.*/-O3 -ffast-math -fbounds-check = global_flag ! build flags for all Fortran code (for a fair comparison to cudacpp; default is -O)/' ${outproc}/Cards/run_card.dat
    # Generate run_card.inc and param_card.inc (include stdout and stderr in the code generation log which is later checked for errors)
    # These two steps are part of "cd Source; make" but they actually are code-generating steps
    # Note: treatcards run also regenerates vector.inc if vector_size has changed in the runcard
    ${outproc}/bin/madevent treatcards run >> ${outproc}_log.txt 2>&1 # AV BUG! THIS MAY SILENTLY FAIL (check if output contains "Please report this bug")
    ${outproc}/bin/madevent treatcards param >> ${outproc}_log.txt 2>&1 # AV BUG! THIS MAY SILENTLY FAIL (check if output contains "Please report this bug")
    # Generate matrix*.pdf files using ps2pdf and remove the original matrix*.ps (NB: this only works if ghostscript is installed)
    if which ps2pdf > /dev/null 2>&1; then
      for matrixps in ${outproc}/SubProcesses/P*/matrix*.ps; do
        matrixpdf=$(dirname $matrixps)/$(basename $matrixps .ps).pdf
        if ps2pdf $matrixps $matrixpdf; then
          ###\rm -f $matrixps # keep also the .ps file as suggested by Olivier #854
          # Strip PDF metadata from ps2pdf to make the file reproducible (use binary 'grep -a' for tests)
          # Alternatively I tried pdftk (https://stackoverflow.com/questions/60738960) but this did not remove PdfID0/PdfID1 or XMP metadata
          # Use 'pdftk <file.pdf> dump_data' to inspect the PDF metadata (but this does not include XMP metadata!)
          # Use 'xxd -c256 <file.pdf>' to view the binary content in text format
          cat $matrixpdf \
            | awk -vdate="D:20240301000000+01'00'" '{print gensub("(^/ModDate\\().*(\\)>>endobj$)","\\1"date"\\2","g")}' \
            | awk -vdate="D:20240301000000+01'00'" '{print gensub("(^/CreationDate\\().*(\\)$)","\\1"date"\\2","g")}' \
            | awk -vid="0123456789abcdef0123456789abcdef" '{print gensub("(^/ID \\[<).*><.*(>\\]$)","\\1"id"><"id"\\2","g")}' \
            | awk -vid="0123456789abcdef0123456789abcdef" '{print gensub("(^/ID \\[\\().*\\)\\(.*(\\)\\]$)","\\1"id")("id"\\2","g")}' \
            | awk -vid="0123456789abcdef0123456789abcdef" '{print gensub("\\("id"\\)","<"id">","g")}' \
            | awk -vdate="2024-03-01T00:00:00+01:00" '{print gensub("(<xmp:ModifyDate>).*(</xmp:ModifyDate>)","\\1"date"\\2","g")}' \
            | awk -vdate="2024-03-01T00:00:00+01:00" '{print gensub("(<xmp:CreateDate>).*(</xmp:CreateDate>)","\\1"date"\\2","g")}' \
            | awk -vuuid="'uuid=01234567-89ab-cdef-0123-456789abcdef'" '{print gensub("(xapMM:DocumentID=).*(/>$)","\\1"uuid"\\2","g")}' \
            > ${matrixpdf}.new
          \mv ${matrixpdf}.new $matrixpdf
        fi
      done
    fi
    # Remove card.jpg/png, diagrams.html and matrix*.jpg/png files (NB: these are only created if ghostscript is installed)
    \rm -f ${outproc}/SubProcesses/P*/card.jpg
    \rm -f ${outproc}/SubProcesses/P*/card.png
    \rm -f ${outproc}/SubProcesses/P*/diagrams.html
    \rm -f ${outproc}/SubProcesses/P*/matrix*jpg
    \rm -f ${outproc}/SubProcesses/P*/matrix*png
    # Cleanup
    \rm -f ${outproc}/crossx.html
    \rm -f ${outproc}/index.html
    \rm -f ${outproc}/madevent.tar.gz
    \rm -f ${outproc}/Cards/delphes_trigger.dat
    \rm -f ${outproc}/Cards/plot_card.dat
    \rm -f ${outproc}/bin/internal/run_plot*
    \rm -f ${outproc}/HTML/*
    \rm -rf ${outproc}/bin/internal/__pycache__
    \rm -rf ${outproc}/bin/internal/ufomodel/py3_model.pkl
    \rm -rf ${outproc}/bin/internal/ufomodel/__pycache__
    touch ${outproc}/HTML/.keep # new file
    if [ "${patchlevel}" != "0" ]; then
      # Add global flag '-O3 -ffast-math -fbounds-check' as in previous gridpacks
      # (FIXME? these flags are already set in the runcards, why are they not propagated to make_opts?)
      echo "GLOBAL_FLAG=-O3 -ffast-math -fbounds-check" > ${outproc}/Source/make_opts.new
      cat ${outproc}/Source/make_opts >> ${outproc}/Source/make_opts.new
      \mv ${outproc}/Source/make_opts.new ${outproc}/Source/make_opts
    fi
    if [ "${patchlevel}" == "2" ]; then
      sed -i 's/DEFAULT_F2PY_COMPILER=f2py.*/DEFAULT_F2PY_COMPILER=f2py3/' ${outproc}/Source/make_opts
      cat ${outproc}/Source/make_opts | sed '/#end/q' | head --lines=-1 | sort > ${outproc}/Source/make_opts.new
      cat ${outproc}/Source/make_opts | sed -n -e '/#end/,$p' >> ${outproc}/Source/make_opts.new
      \mv ${outproc}/Source/make_opts.new ${outproc}/Source/make_opts
    fi
  fi
  popd >& /dev/null
  # Choose which directory must be copied (for gridpack generation: untar and modify the gridpack)
  if [ "${SCRBCK}" == "gridpack" ]; then
    ###outprocauto=${MG5AMC_HOME}/${outproc}/run_01_gridpack
    ###if ! $SCRDIR/untarGridpack.sh ${outprocauto}.tar.gz; then echo "ERROR! untarGridpack.sh failed"; exit 1; fi
    echo "ERROR! gridpack mode is no longer supported by this script!"; exit 1
  else
    outprocauto=${MG5AMC_HOME}/${outproc}
  fi
  cp -dpr ${MG5AMC_HOME}/${outproc}_log.txt ${outprocauto}/
  # Output directories: examples ee_mumu.sa for cudacpp, eemumu.auto for alpaka and gridpacks, eemumu.cpp or eemumu.gpu for cpp and gpu
  autosuffix=sa
  if [ "${SCRBCK}" == "gridpack" ]; then
    ###autosuffix=auto
    echo "ERROR! gridpack mode is no longer supported by this script!"; exit 1
  elif [ "${SCRBCK}" == "alpaka" ]; then
    ###autosuffix=auto
    echo "ERROR! alpaka mode is no longer supported by this script!"; exit 1
  elif [ "${OUTBCK}" == "cpp" ]; then
    autosuffix=cpp # no special suffix for the 311 branch any longer
  elif [ "${OUTBCK}" == "gpu" ]; then
    autosuffix=gpu # no special suffix for the 311 branch any longer
  elif [ "${OUTBCK}" == "madnovec" ] || [ "${OUTBCK}" == "madonly" ] || [ "${OUTBCK}" == "mad" ] || [ "${OUTBCK}" == "madcpp" ] || [ "${OUTBCK}" == "madgpu" ]; then
    autosuffix=${OUTBCK}
  fi
  # Replace the existing generated code in the output source code directory by the newly generated code and create a .BKP
  rm -rf ${OUTDIR}/${proc}.${autosuffix}.BKP
  if [ -d ${OUTDIR}/${proc}.${autosuffix} ]; then mv ${OUTDIR}/${proc}.${autosuffix} ${OUTDIR}/${proc}.${autosuffix}.BKP; fi
  cp -dpr ${outprocauto} ${OUTDIR}/${proc}.${autosuffix}
  echo -e "\nOutput source code has been copied to ${OUTDIR}/${proc}.${autosuffix}"
  # Add file mg5.in as in Stephan's runCodegen.sh script
  cat ${MG5AMC_HOME}/${outproc}.mg | sed "s|${outproc}|${proc}.${autosuffix}|" > ${OUTDIR}/${proc}.${autosuffix}/mg5.in
  # Fix build errors which arise because the autogenerated directories are not relocatable (see #400)
  if [ "${OUTBCK}" == "madnovec" ] || [ "${OUTBCK}" == "madonly" ] || [ "${OUTBCK}" == "mad" ] || [ "${OUTBCK}" == "madcpp" ] || [ "${OUTBCK}" == "madgpu" ]; then
    cat ${OUTDIR}/${proc}.${autosuffix}/Cards/me5_configuration.txt | sed 's/mg5_path/#mg5_path/' > ${OUTDIR}/${proc}.${autosuffix}/Cards/me5_configuration.txt.new
    \mv ${OUTDIR}/${proc}.${autosuffix}/Cards/me5_configuration.txt.new ${OUTDIR}/${proc}.${autosuffix}/Cards/me5_configuration.txt
  fi
  # Additional patches for mad directory (integration of Fortran and cudacpp)
  # [NB: these patches are not applied to madnovec/madonly, which are meant as out-of-the-box references]
  ###if [ "${OUTBCK}" == "mad" ]; then
  ###  dir_patches=PROD
  ###  $SCRDIR/patchMad.sh ${OUTDIR}/${proc}.${autosuffix} ${vecsize} ${dir_patches} ${PATCHLEVEL}
  ###fi
  # Additional patches that are ONLY NEEDED IN THE MADGRAPH4GPU GIT REPO
  cat << EOF > ${OUTDIR}/${proc}.${autosuffix}/.gitignore
crossx.html
index.html
results.dat*
results.pkl
run_[0-9]*
events.lhe*
EOF
  if [ -d ${OUTDIR}/${proc}.${autosuffix}/bin/internal/ufomodel ]; then # see PR #762
    cat << EOF > ${OUTDIR}/${proc}.${autosuffix}/bin/internal/ufomodel/.gitignore
py3_model.pkl
EOF
  fi
  if [ -f ${OUTDIR}/${proc}.${autosuffix}/SubProcesses/proc_characteristics ]; then 
    sed -i 's/bias_module = None/bias_module = dummy/' ${OUTDIR}/${proc}.${autosuffix}/SubProcesses/proc_characteristics
  fi
  for p1dir in ${OUTDIR}/${proc}.${autosuffix}/SubProcesses/P*; do
    cat << EOF > ${p1dir}/.gitignore
.libs
.cudacpplibs
madevent
madevent_fortran
madevent_cpp
madevent_cuda

G[0-9]*
ajob[0-9]*
input_app.txt
symfact.dat
gensym
EOF
  done
  # Compare the existing generated code to the newly generated code for the specific process
  if [ "$QUIET" != "1" ]; then
    pushd ${OUTDIR} >& /dev/null
    echo -e "\n+++ Compare old and new code generation log for $proc\n"
    ###if diff -c ${proc}.${autosuffix}.BKP/${outproc}_log.txt ${proc}.${autosuffix}; then echo "Old and new code generation logs are identical"; fi # context diff
    if diff ${proc}.${autosuffix}.BKP/$(basename ${outproc})_log.txt ${proc}.${autosuffix}; then echo "Old and new code generation logs are identical"; fi # context diff
    echo -e "\n+++ Compare old and new generated code for $proc\n"
    if $SCRDIR/diffCode.sh ${BRIEF} -r -c ${proc}.${autosuffix}.BKP ${proc}.${autosuffix}; then echo "Old and new generated codes are identical"; else echo -e "\nWARNING! Old and new generated codes differ"; fi
    popd >& /dev/null
  fi
  # Print a summary of the available code
  if [ "$QUIET" != "1" ]; then
    echo
    echo -e "Manually developed code is\n  ${OUTDIR}/${proc}"
    echo -e "Old generated code moved to\n  ${OUTDIR}/${proc}.${autosuffix}.BKP"
    echo -e "New generated code moved to\n  ${OUTDIR}/${proc}.${autosuffix}"
  fi
}

#--------------------------------------------------------------------------------------

function usage()
{
  # NB: Generate only one process at a time
  if [ "${SCRBCK}" == "gridpack" ]; then
    # NB: gridpack generation uses the 311 branch by default
    ###echo "Usage: $0 [--nobrief] [--nountaronly] [--nohelrec] <proc>"
    echo "ERROR! gridpack mode is no longer supported by this script!"; exit 1
  elif [ "${SCRBCK}" == "alpaka" ]; then
    # NB: alpaka generation uses the 311 branch by default
    ###echo "Usage: $0 [--nobrief] <proc>"
    echo "ERROR! alpaka mode is no longer supported by this script!"; exit 1
  else
    # NB: all options with $SCRBCK=cudacpp use the 311 branch by default and always disable helicity recycling
    echo "Usage:   $0 [-q|--nobrief] [--cpp|--gpu|--madnovec|--madonly|--mad|--madcpp*|--madgpu] [--nopatch|--upstream] [-c '<cmd>'] <proc>"
    echo "         (*Note: the --madcpp option exists but code generation fails for it)"
    echo "         (**Note: <proc> will be used as a relative path in ${OUTDIR} and should not contain '/' characters"
    echo "Example: $0 gg_tt --mad"
    echo "Example: $0 gg_bb --mad -c 'generate g g > b b~'"
  fi
  exit 1
}

#--------------------------------------------------------------------------------------

# Clean up MG5AMC/mg5amcnlo
function cleanup_MG5AMC_HOME()
{
  # Remove MG5aMC fragments from previous runs
  rm -f ${MG5AMC_HOME}/py.py
  rm -f ${MG5AMC_HOME}/Template/LO/Source/make_opts
  rm -f ${MG5AMC_HOME}/input/mg5_configuration.txt
  rm -f ${MG5AMC_HOME}/models/sm/py3_model.pkl
  # Remove any *~ files in MG5AMC_HOME
  rm -rf $(find ${MG5AMC_HOME} -name '*~')
}

# Clean up MG5AMC/mg5amcnlo/PLUGIN
# (NB: as of PR #1008, this is needed before code generation, to ensure that the only plugin is MG5AMC/MG5AMC_PLUGIN)
function cleanup_MG5AMC_PLUGIN()
{
  # Remove and recreate MG5AMC_HOME/PLUGIN
  rm -rf ${MG5AMC_HOME}/PLUGIN
  mkdir ${MG5AMC_HOME}/PLUGIN
  touch ${MG5AMC_HOME}/PLUGIN/__init__.py
}

#--------------------------------------------------------------------------------------

# Script directory
SCRDIR=$(cd $(dirname $0); pwd)

# Output source code directory for the chosen backend (generated code will be created as a subdirectory of $OUTDIR)
OUTDIR=$(dirname $SCRDIR) # e.g. epochX/cudacpp if $SCRDIR=epochX/cudacpp/CODEGEN

# Script directory backend (cudacpp, gridpack or alpaka)
SCRBCK=$(basename $OUTDIR) # e.g. cudacpp if $OUTDIR=epochX/cudacpp

# Default output backend (in the cudacpp directory this can be changed using commad line options like --cpp, --gpu or --mad)
OUTBCK=$SCRBCK

# Default: brief diffs (use --nobrief to use full diffs, use -q to be much quieter)
QUIET=
BRIEF=--brief

# Default for gridpacks: untar gridpack.tar.gz but do not regenerate it (use --nountaronly to regenerate it)
UNTARONLY=1

# Default: apply all patches in patchMad.sh (--nopatch is ignored unless --mad is also specified)
PATCHLEVEL=
patchlevel=2 # [DEFAULT] complete generation of cudacpp .sa/.mad (copy templates and apply patch commands)

# Default for gridpacks: use helicity recycling (use --nohelrec to disable it)
# (export the value to the untarGridpack.sh script)
# Hardcoded for cudacpp and alpaka: disable helicity recycling (#400, #279) for the moment
if [ "${SCRBCK}" == "gridpack" ]; then export HELREC=1; else export HELREC=0; fi

# Process command line arguments (https://unix.stackexchange.com/a/258514)
cmd=
proc=
while [ "$1" != "" ]; do
  if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
  elif [ "$1" == "--nobrief" ] && [ "$QUIET" != "1" ]; then
    BRIEF=
  elif [ "$1" == "-q" ] && [ "$BRIEF" != "" ]; then
    QUIET=1
  elif [ "$1" == "--nopatch" ] && [ "${PATCHLEVEL}" == "" ]; then
    PATCHLEVEL=--nopatch
    patchlevel=1 # [--nopatch] modify upstream MG5AMC but do not apply patch commands (reference to prepare new patches)
  elif [ "$1" == "--upstream" ] && [ "${PATCHLEVEL}" == "" ]; then
    PATCHLEVEL=--upstream
    patchlevel=0 # [--upstream] out of the box codegen from upstream MG5AMC (do not even copy templates)
  elif [ "$1" == "--nountaronly" ] && [ "${SCRBCK}" == "gridpack" ]; then
    ###UNTARONLY=0
    echo "ERROR! gridpack mode is no longer supported by this script!"; exit 1
  elif [ "$1" == "--nohelrec" ] && [ "${SCRBCK}" == "gridpack" ]; then
    ###export HELREC=0
    echo "ERROR! gridpack mode is no longer supported by this script!"; exit 1
  elif [ "$1" == "--cpp" ] && [ "${SCRBCK}" == "cudacpp" ]; then
    export OUTBCK=${1#--}
  elif [ "$1" == "--gpu" ] && [ "${SCRBCK}" == "cudacpp" ]; then
    export OUTBCK=${1#--}
  elif [ "$1" == "--madnovec" ] && [ "${SCRBCK}" == "cudacpp" ]; then
    export OUTBCK=${1#--}
  elif [ "$1" == "--madonly" ] && [ "${SCRBCK}" == "cudacpp" ]; then
    export OUTBCK=${1#--}
  elif [ "$1" == "--mad" ] && [ "${SCRBCK}" == "cudacpp" ]; then
    export OUTBCK=${1#--}
  elif [ "$1" == "--madcpp" ] && [ "${SCRBCK}" == "cudacpp" ]; then
    export OUTBCK=${1#--}
  elif [ "$1" == "--madgpu" ] && [ "${SCRBCK}" == "cudacpp" ]; then
    export OUTBCK=${1#--}
  elif [ "$1" == "-c" ] && [ "$2" != "" ]; then
    cmd="$2"
    shift
  elif [ "$proc" == "" ]; then
    proc="$1"
  else
    usage
  fi
  shift
done
if [ "$proc" == "" ]; then usage; fi
if [ "${proc/\/}" != "${proc}" ]; then echo "ERROR! <proc> '${proc}' should not contain '/' characters"; usage; fi

echo "SCRDIR=${SCRDIR}"
echo "OUTDIR=${OUTDIR}"
echo "SCRBCK=${SCRBCK} (uppercase=${SCRBCK^^})"
echo "OUTBCK=${OUTBCK}"

echo "BRIEF=${BRIEF}"
echo "QUIET=${QUIET}"
echo "proc=${proc}"

# Make sure that python3 is installed
if ! python3 --version >& /dev/null; then echo "ERROR! python3 is not installed"; exit 1; fi

# Define MG5AMC_HOME as a path to the mg5amcnlo git submodule in this repo (NEW IMPLEMENTATION BASED ON SUBMODULES)
# Make sure that $MG5AMC_HOME exists
if [ "$MG5AMC_HOME" == "" ]; then
  # (FIXME: in a future implementation, this will not be ignored - e.g. to share a single git submodule in different madgraph4gpu directories)
  echo "WARNING! Environment variable MG5AMC_HOME is already defined as '$MG5AMC_HOME' but it will be redefined"
fi
MG5AMC_HOME=${SCRDIR}/../../../MG5aMC/mg5amcnlo
if [ ! -d $MG5AMC_HOME ]; then
  echo "ERROR! Directory $MG5AMC_HOME does not exist"
  exit 1
fi
export MG5AMC_HOME=$(cd $MG5AMC_HOME; pwd)
echo -e "\nDefault MG5AMC_HOME=$MG5AMC_HOME on $(hostname)\n"

# Make sure that $ALPAKA_ROOT and $CUPLA_ROOT exist if alpaka is used
###if [ "${SCRBCK}" == "alpaka" ]; then
###  if [ "$ALPAKA_ROOT" == "" ]; then
###    echo "ERROR! ALPAKA_ROOT is not defined"
###    echo "To download ALPAKA please run 'git clone -b 0.8.0 https://github.com/alpaka-group/alpaka.git'"
###    exit 1
###  fi
###  echo -e "Using ALPAKA_ROOT=$ALPAKA_ROOT on $(hostname)\n"
###  if [ ! -d $ALPAKA_ROOT ]; then echo "ERROR! Directory $ALPAKA_ROOT does not exist"; exit 1; fi
###  if [ "$CUPLA_ROOT" == "" ]; then
###    echo "ERROR! CUPLA_ROOT is not defined"
###    echo "To download CUPLA please run 'git clone -b 0.3.0 https://github.com/alpaka-group/cupla.git'"
###    exit 1
###  fi
###  echo -e "Using CUPLA_ROOT=$CUPLA_ROOT on $(hostname)\n"
###  if [ ! -d $CUPLA_ROOT ]; then echo "ERROR! Directory $CUPLA_ROOT does not exist"; exit 1; fi
###fi

# Show the git branch and commit of MG5aMC
if ! git --version >& /dev/null; then
  echo -e "ERROR! git is not installed: cannot retrieve git properties of MG5aMC_HOME\n"; exit 1
fi
cd ${MG5AMC_HOME}
if [ "$QUIET" != "1" ]; then
  echo -e "Using $(git --version)"
  echo -e "Retrieving git information about MG5AMC_HOME"
fi
if ! git log -n1 >& /dev/null; then
  echo -e "ERROR! MG5AMC_HOME is not a git clone\n"; exit 1
fi
branch_mg5amc=$(git branch --no-color | \grep ^* | awk '{print $2}')
commit_mg5amc=$(git log --oneline -n1 | awk '{print $1}')
if [ "$QUIET" != "1" ]; then
  echo -e "Current git branch of MG5AMC_HOME is '${branch_mg5amc}'"
  echo -e "Current git commit of MG5AMC_HOME is '${commit_mg5amc}'"
fi
cd - > /dev/null

# Copy MG5AMC ad-hoc patches if any (unless --upstream is specified)
###if [ "${PATCHLEVEL}" != "--upstream" ] && [ "${SCRBCK}" == "cudacpp" ]; then
###  patches=$(cd $SCRDIR/MG5aMC_patches/${dir_patches}; find . -mindepth 2 -type f)
###  echo -e "Copy MG5aMC_patches/${dir_patches} patches..."
###  for patch in $patches; do
###    ###echo "DEBUG: $patch"
###    patch=${patch#./} # strip leading "./"
###    if [ "${patch}" != "${patch%~}" ]; then continue; fi # skip *~ files
###    ###if [ "${patch}" != "${patch%.GIT}" ]; then continue; fi # skip *.GIT files
###    ###if [ "${patch}" != "${patch%.py}" ]; then continue; fi # skip *.py files
###    ###if [ "${patch}" != "${patch#patch.}" ]; then continue; fi # skip patch.* files
###    echo cp -dpr $SCRDIR/MG5aMC_patches/${dir_patches}/$patch $MG5AMC_HOME/$patch
###    cp -dpr $SCRDIR/MG5aMC_patches/${dir_patches}/$patch $MG5AMC_HOME/$patch
###  done
###  echo -e "Copy MG5aMC_patches/${dir_patches} patches... done\n"
###fi

# Clean up MG5AMC/mg5amcnlo before code generation
cleanup_MG5AMC_HOME

# Clean up MG5AMC/mg5amcnlo/PLUGIN before code generation
# (NB: between PR #766 and PR #1008, this was removed because cudacpp was a symlink in MG5AMC/mg5amcnlo/PLUGIN)
# (NB: as of PR #1008, this is needed again to ensure that the only plugin is MG5AMC/MG5AMC_PLUGIN)
cleanup_MG5AMC_PLUGIN

# Print differences in MG5AMC with respect to git after copying ad-hoc patches
if [ "$QUIET" != "1" ]; then
  cd ${MG5AMC_HOME}
  echo -e "\n***************** Differences to the current git commit ${commit_patches} [START]"
  ###if [ "$(git diff)" == "" ]; then echo -e "[No differences]"; else git diff; fi
  if [ "$(git diff)" == "" ]; then echo -e "[No differences]"; else git diff --name-status; fi
  echo -e "***************** Differences to the current git commit ${commit_patches} [END]\n"
  cd - > /dev/null
fi

# Copy the new plugin to MG5AMC_HOME (if the script directory backend is cudacpp or alpaka)
#if [ "${SCRBCK}" == "cudacpp" ]; then
#  if [ "${OUTBCK}" == "no-path-to-this-statement" ]; then
#    echo -e "\nWARNING! '${OUTBCK}' mode selected: do not copy the cudacpp plugin (workaround for #341)"
#  else # currently succeeds also for madcpp and madgpu (#341 has been fixed)
#    echo -e "\nINFO! '${OUTBCK}' mode selected: copy the cudacpp plugin\n"
#    cp -dpr ${SCRDIR}/PLUGIN/${SCRBCK^^}_SA_OUTPUT ${MG5AMC_HOME}/PLUGIN/${SCRBCK^^}_OUTPUT
#    ls -l ${MG5AMC_HOME}/PLUGIN
#  fi
###elif [ "${SCRBCK}" == "alpaka" ]; then
###  cp -dpr ${SCRDIR}/PLUGIN/${SCRBCK^^}_CUDACPP_SA_OUTPUT ${MG5AMC_HOME}/PLUGIN/
###  ls -l ${MG5AMC_HOME}/PLUGIN
#fi

# For gridpacks, use separate output directories for MG 29x and MG 3xx
###if [ "${SCRBCK}" == "gridpack" ]; then
###  if [ "${HELREC}" == "0" ]; then
###    OUTDIR=${OUTDIR}/3xx_nohelrec
###  else
###    OUTDIR=${OUTDIR}/3xx
###  fi
###  echo "OUTDIR=${OUTDIR} (redefined)"
###fi

# Generate the chosen process (this will always replace the existing code directory and create a .BKP)
SECONDS=0 # bash built-in
export CUDACPP_CODEGEN_PATCHLEVEL=${PATCHLEVEL}
codeGenAndDiff $proc "$cmd"

# Clean up MG5AMC/mg5amcnlo after code generation
cleanup_MG5AMC_HOME

# Clean up MG5AMC/mg5amcnlo/PLUGIN after code generation
# (NB: between PR #766 and PR #1008, this was removed because cudacpp was a symlink in MG5AMC/mg5amcnlo/PLUGIN)
# (NB: as of PR #1008, this remains unnecessary because MG5AMC/mg5amcnlo/PLUGIN is not modified by generateAndCompare.sh)
###cleanup_MG5AMC_PLUGIN

# Check formatting in the auto-generated code
if [ "${OUTBCK}" == "cudacpp" ]; then
  echo -e "\n+++ Check code formatting in newly generated code ${proc}.sa\n"
  if ! $SCRDIR/checkFormatting.sh -q -q ${proc}.sa; then
    echo "ERROR! Auto-generated code does not respect formatting policies"
    exit 1
  fi
elif [ "${OUTBCK}" == "mad" ]; then
  echo -e "\n+++ Check code formatting in newly generated code ${proc}.mad\n"
  if ! $SCRDIR/checkFormatting.sh -q -q ${proc}.mad; then
    echo "ERROR! Auto-generated code does not respect formatting policies"
    exit 1
  fi
fi

echo
echo "********************************************************************************"
echo
echo "Code generation and additional checks completed in $SECONDS seconds"
echo
