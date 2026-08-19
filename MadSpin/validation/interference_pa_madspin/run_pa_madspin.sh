#!/bin/bash
# Pure-interference mode under spinmode = PA and spinmode = madspin, for both
# values of pure_interference_output.  Driven from the repository root:
#
#   bash MadSpin/validation/interference_pa_madspin/run_pa_madspin.sh <workdir> [nevents]
#
# PR #363 shipped two output paths (`pure_interference_output = weighted`, the
# default, and `= unweighted`) validated end to end only under
# `spinmode = onshell`.  This sweep runs the SAME physics block -- (I,I) of
# p p > t t~, i.e.
#
#     set pure_interference t  = + -
#     set pure_interference t~ = + -
#
# -- under every (spinmode x output) combination, on ONE shared production
# sample, so the only thing that changes between runs is MadSpin.
#
#   tag         spinmode   pure_interference_output
#   onshell_w   onshell    weighted     (control: must reproduce doc 13.16)
#   onshell_u   onshell    unweighted   (control for the unweighted path)
#   pa_w        PA         weighted
#   pa_u        PA         unweighted
#   ms_w        madspin    weighted
#   ms_u        madspin    unweighted
#
# The production sample is generated ONCE (no madspin card) and every
# combination decays that same file with the standalone MadSpin front end, so
# the six results are statistically correlated event by event and any
# difference between them is MadSpin's alone.  `run_integration.sh` separately
# checks the generate_events path, where the decayed file has to land on top of
# Events/<run>/unweighted_events.lhe.gz.
#
# Reshuffle instrumentation: the card is run through drive_madspin.py, which is
# `MadSpin/madspin` plus a wrapper on Event.reshuffle_production that counts
# every call and every return that is not a finite positive number.  That is
# the only route by which the pure-interference weight W can come out <= 0 for
# a reason that is not physics, so the rate is measured rather than argued
# about.  Nothing in the shipped source is modified.
set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MG5=${MG5:-$ROOT/bin/mg5_aMC}
DRIVER=${DRIVER:-$(dirname "${BASH_SOURCE[0]}")/drive_madspin.py}
PYTHON=${PYTHON:-python3}
WORK=${1:?usage: run_pa_madspin.sh <workdir> [nevents]}
N_EVENTS=${2:-50000}
NB_CORE=${NB_CORE:-8}
ME_SEED=${ME_SEED:-4321}

mkdir -p "$WORK"

# ------------------------------------------------------- production, once
if [ ! -f "$WORK/.generated" ]; then
  {
    echo "set acknowledged_v3.1_syntax True --no_save"
    echo "generate p p > t t~"
    echo "output $WORK/prod -f"
  } > "$WORK/gen.mg5"
  "$MG5" "$WORK/gen.mg5" > "$WORK/gen.log" 2>&1
  touch "$WORK/.generated"
fi

d="$WORK/prod"
perl -pi -e "s/^(\s*)\S+(\s*=\s*nevents\b)/\${1}$N_EVENTS\${2}/"   "$d/Cards/run_card.dat"
perl -pi -e "s/^(\s*)\S+(\s*=\s*iseed\b)/\${1}$ME_SEED\${2}/"      "$d/Cards/run_card.dat"
perl -pi -e "s/^(\s*)\S+(\s*=\s*use_syst\b)/\${1}False\${2}/"      "$d/Cards/run_card.dat"
grep -q '^run_mode' "$d/Cards/me5_configuration.txt" || \
  printf 'run_mode = 2\nnb_core = %s\nautomatic_html_opening = False\n' "$NB_CORE" \
      >> "$d/Cards/me5_configuration.txt"

if [ ! -f "$WORK/prod/Events/prod/unweighted_events.lhe.gz" ]; then
  ( cd "$WORK" && "$d/bin/generate_events" -f prod > "$WORK/log_prod.txt" 2>&1 )
fi
EVT="$WORK/prod/Events/prod/unweighted_events.lhe.gz"
ls -l "$EVT"

# ------------------------------------------------------------ combinations
# tag | spinmode | output | ms_seed
COMBOS=(
  "onshell_w|onshell|weighted|7786"
  "onshell_u|onshell|unweighted|7787"
  "pa_w|PA|weighted|7788"
  "pa_u|PA|unweighted|7789"
  "ms_w|madspin|weighted|7790"
  "ms_u|madspin|unweighted|7791"
)

for c in "${COMBOS[@]}"; do
  IFS='|' read -r tag mode out msseed <<< "$c"
  o="$WORK/$tag"
  if [ -f "$o/events_decayed.lhe.gz" ]; then echo "skip $tag"; continue; fi
  mkdir -p "$o"
  cp "$EVT" "$o/events.lhe.gz"
  {
    echo "import $o/events.lhe.gz"
    echo "set seed $msseed"
    echo "set spinmode $mode"
    echo "set max_weight_ps_point 400"
    echo "set BW_cut 15"
    echo "set nb_core $NB_CORE"
    echo "set pure_interference t  = + -"
    echo "set pure_interference t~ = + -"
    echo "set pure_interference_output $out"
    echo "define lp = e+ mu+"
    echo "define lm = e- mu-"
    echo "define vl = ve vm"
    echo "define vlx = ve~ vm~"
    echo "decay t > b w+, w+ > lp vl"
    echo "decay t~ > b~ w-, w- > lm vlx"
    echo "launch"
  } > "$o/madspin_card.dat"
  echo "=== $tag (spinmode=$mode, output=$out)"
  start=$(date +%s)
  set +e
  MS_PA_JACLOG="$o/jac" \
      "$PYTHON" -O "$DRIVER" "$o/madspin_card.dat" > "$o/madspin.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$o/exit_code"
  echo "$(( $(date +%s) - start ))" > "$o/wallclock_s"
  echo "$tag exit=$rc  $(cat "$o/wallclock_s")s"
done

echo "all combinations done"
