#!/usr/bin/env bash
set -e

SHOWER=$1
#HEP-> hepmc/stdhep
#TOP-> top
OUTPUT=$2
RUN_NAME=$3
NFILE=$4

export %(ld_library_path)s=$%(ld_library_path)s:%(extralibs)s
# PDF sets copied for this process (lib/PDFsets) and the LHAPDF default ones
LHAPDF_EXTRA_PATH="%(lhapdf_data_path)s"
if [ -n "$LHAPDF_EXTRA_PATH" ]; then
    export LHAPDF_DATA_PATH="$LHAPDF_EXTRA_PATH${LHAPDF_DATA_PATH:+:$LHAPDF_DATA_PATH}"
fi

if [ $SHOWER == "HERWIG7" ] ; then
    export PYTHIA8DATA=""
else
    export PYTHIA8DATA=`pwd`/xmldoc
fi

# if one is splitting file cd to a new dir and link all files here
if [[ "$NFILE" != "" ]]; then
    mkdir -p "run_$NFILE"
    cd "run_$NFILE"
    ln -sf "../events_$NFILE.lhe" events.lhe
    # Deliver the log even when the shower or the analysis fails. Successful
    # jobs no longer need their input link; the caller removes the split files
    # after all jobs have finished.
    finish_split_job() {
        status=$?
        if [ -f mcatnlo_run.log ]; then
            mv mcatnlo_run.log "../mcatnlo_run_$NFILE.log"
        fi
        if [ "$status" -eq 0 ]; then rm -f events.lhe; fi
        exit "$status"
    }
    trap finish_split_job EXIT
    if [ $SHOWER == "PYTHIA8" ] ; then
        cp ../Pythia8.exe ../Pythia8.cmd .
        if [ -f ../mcatnlo_locked.dat ] ; then cp ../mcatnlo_locked.dat . ; fi
        # Each split job needs a different random sequence. Keep the first
        # job's fixed seed, and wrap later seeds within Pythia's valid range.
        python3 - "$NFILE" <<'PYTHIA_SEED' || exit 1
import pathlib
import re
import sys

path = pathlib.Path('Pythia8.cmd')
text = path.read_text()
matches = list(re.finditer(r'^\s*Random:seed\s*=\s*([+-]?\d+)(?=\s|$)',
                           text, re.MULTILINE | re.IGNORECASE))
if matches and int(matches[-1][1]) > 0:
    match = matches[-1]
    seed = (int(match[1]) + int(sys.argv[1]) - 2) %% 900000000 + 1
    path.write_text(text[:match.start(1)] + str(seed) + text[match.end(1):])
PYTHIA_SEED
        if [ -f ../config.sh ] ; then cp ../config.sh . ; fi
    else
        if [ $SHOWER == "HERWIG7" ] ; then
            cp ../Herwig* .
            if [ -f ../HepMCFortran.so ] ; then cp ../HepMCFortran.so . ; fi
        fi
        cp ../MCATNLO_$SHOWER\_EXE ../MCATNLO_$SHOWER\_input .
        if [ $SHOWER == "HERWIG7" ] ; then
            # Same per-job seed offset as for Pythia8.
            python3 - "$NFILE" <<'HERWIG_SEED' || exit 1
import pathlib
import re
import sys

path = pathlib.Path('MCATNLO_HERWIG7_input')
text = path.read_text()
matches = list(re.finditer(r'(RandomNumberGenerator:Seed\s+)([+-]?\d+)', text))
if matches and int(matches[-1][2]) > 0:
    match = matches[-1]
    seed = (int(match[2]) + int(sys.argv[1]) - 2) %% 900000000 + 1
    path.write_text(text[:match.start(2)] + str(seed) + text[match.end(2):])
HERWIG_SEED
        fi
    fi
fi

if [ -e events.lhe.gz ] ; then
    gunzip events.lhe.gz
fi

if [ "$SHOWER" == "HERWIG6" ] || [ "$SHOWER" == "PYTHIA6Q" ] || [ "$SHOWER" == "PYTHIA6PT" ] || [ "$SHOWER" == "HERWIG7" ] ; then
    ./MCATNLO_$SHOWER\_EXE < MCATNLO_$SHOWER\_input > mcatnlo_run.log 2>&1

elif [ "$SHOWER" == "PYTHIA8" ] ; then
    # The driver verifies the settings listed in mcatnlo_locked.dat, which
    # the MC@NLO subtraction requires, after initialising Pythia8.
    if [ -f mcatnlo_locked.dat ] ; then
        ./Pythia8.exe Pythia8.cmd mcatnlo_locked.dat > mcatnlo_run.log 2>&1
    else
        ./Pythia8.exe Pythia8.cmd > mcatnlo_run.log 2>&1
    fi
fi

if [ "$OUTPUT" == "HEP" ] ; then
    # hep or hepmc output
    # this is for the final filename
    if [[ "$NFILE" != "" ]]; then
        NAME="../events_$NFILE"
    else
        NAME="events"
    fi
    # at the end a file called events.hep.gz or events.hepmc.gz will be delivered
    if [ "$SHOWER" == "HERWIG6" ] || [ "$SHOWER" == "PYTHIA6Q" ] || [ "$SHOWER" == "PYTHIA6PT" ] ; then
        mv events.lhe.hep $NAME.hep
        gzip $NAME.hep
    elif [ "$SHOWER" == "HERWIG7" ] ; then
        mv MCATNLO_HERWIG7.hepmc $NAME.hepmc
        gzip $NAME.hepmc
    elif [ "$SHOWER" == "PYTHIA8" ] ; then
        mv Pythia8.hep $NAME.hepmc
        gzip $NAME.hepmc
    fi

elif [ "$OUTPUT" == "TOP" ] || [ "$OUTPUT" == "HWU" ]; then
    #top output 
    # this is for the final filename
    if [[ "$NFILE" != "" ]]; then
        NAME="../histfile_$NFILE"
    else
        NAME="histfile"
    fi
    # Only pass existing files to tar. Unmatched globs used to make every
    # analysis job report a tar error, even when its output was present.
    shopt -s nullglob
    histograms=( *.top *.TOP *.HwU *.hwu *.HWU )
    tar -cf "$NAME.tar" -T /dev/null "${histograms[@]}" > tarlog.txt 2>&1
fi
