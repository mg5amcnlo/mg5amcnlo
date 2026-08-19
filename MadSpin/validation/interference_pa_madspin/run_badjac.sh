#!/bin/bash
# Fault injection: what do the two pure-interference output paths DO with a
# failed reshuffle?
#
#   bash MadSpin/validation/interference_pa_madspin/run_badjac.sh <workdir> [nevents] [prob]
#
# `<workdir>` must already hold the production sample run_pa_madspin.sh made
# (`$WORK/prod/Events/prod/unweighted_events.lhe.gz`).
#
# Why this exists.  Under `spinmode = PA` the pure-interference convolution is
# W = wgt * jac with jac the return of Event.reshuffle_production, and a
# jac <= 0 is the one way W can be <= 0 for a reason that is NOT physics.  In a
# normal p p > t t~ run that never happens (measured: 0 in 50 000 events -- the
# virtuality draw is capped at the remaining sqrt(shat) budget by
# _draw_mass_value, so RAMBO's `sum(new_mass) > sqrts` branch is unreachable),
# which leaves the handling untested.  So it is forced.
#
# Two injections, because they are two different questions:
#
#   jac = 0   -- "this trial contributes nothing".  W = 0 exactly.
#   jac = -1  -- RAMBO's own sentinel for an impossible mass set.  W = -|wgt|:
#                a full-magnitude NEGATIVE weight, indistinguishable at the
#                weight level from a legitimate interference sign.  This is the
#                conflation to check for: an invalid trial must not be able to
#                enter the sample carrying a sign.
#
# Both are run against both output shapes, and against `decay_output = weighted`
# (which shares the same code path but DOES treat signed <= 0 as dead) as the
# contrast case.
set -e

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DRIVER=${DRIVER:-$(dirname "${BASH_SOURCE[0]}")/drive_madspin.py}
PYTHON=${PYTHON:-python3}
WORK=${1:?usage: run_badjac.sh <workdir> [nevents] [prob]}
N=${2:-2000}
PROB=${3:-0.10}
NB_CORE=${NB_CORE:-4}

EVT="$WORK/prod/Events/prod/unweighted_events.lhe.gz"
[ -f "$EVT" ] || { echo "no production sample at $EVT -- run run_pa_madspin.sh first"; exit 1; }

# tag | pure_interference_output | forced jac | decay_output
CASES=(
  "bad0_w|weighted|0.0|"
  "bad0_u|unweighted|0.0|"
  "badm1_w|weighted|-1.0|"
  "badm1_u|unweighted|-1.0|"
  "bad0_dw||0.0|weighted"
  "badm1_dw||-1.0|weighted"
)

for c in "${CASES[@]}"; do
  IFS='|' read -r tag out jac dout <<< "$c"
  o="$WORK/badjac_$tag"
  if [ -f "$o/events_decayed.lhe.gz" ]; then echo "skip $tag"; continue; fi
  mkdir -p "$o"
  # a short prefix of the production sample: this is a mechanism test, not a
  # measurement, so it does not need the full statistics
  "$PYTHON" - "$EVT" "$o/events.lhe.gz" "$N" <<'PY'
import gzip, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = gzip.open(dst, 'wt')
kept = 0
for line in gzip.open(src, 'rt'):
    if line.startswith('<event') and kept >= n:
        out.write('</LesHouchesEvents>\n'); break
    if line.startswith('</event'):
        kept += 1
    out.write(line)
out.close()
PY
  {
    echo "import $o/events.lhe.gz"
    echo "set seed 9001"
    echo "set spinmode PA"
    echo "set max_weight_ps_point 100"
    echo "set BW_cut 15"
    echo "set nb_core $NB_CORE"
    if [ -n "$out" ]; then
      echo "set pure_interference t  = + -"
      echo "set pure_interference t~ = + -"
      echo "set pure_interference_output $out"
    fi
    [ -n "$dout" ] && echo "set decay_output $dout"
    echo "define lp = e+ mu+"
    echo "define lm = e- mu-"
    echo "define vl = ve vm"
    echo "define vlx = ve~ vm~"
    echo "decay t > b w+, w+ > lp vl"
    echo "decay t~ > b~ w-, w- > lm vlx"
    echo "launch"
  } > "$o/madspin_card.dat"
  echo "=== $tag (output=${out:-<none>} decay_output=${dout:-<none>} forced jac=$jac p=$PROB)"
  set +e
  MS_PA_JACLOG="$o/jac" MS_PA_FORCE_BADJAC="$PROB:$jac" \
      "$PYTHON" -O "$DRIVER" "$o/madspin_card.dat" > "$o/madspin.log" 2>&1
  rc=$?
  set -e
  echo "$rc" > "$o/exit_code"
  echo "$tag exit=$rc"
done

echo "all fault-injection cases done"
