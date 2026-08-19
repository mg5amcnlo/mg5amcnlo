#!/bin/bash
# Two checks that do not belong in the main sweep.
#
#   bash MadSpin/validation/interference_pa_madspin/run_extras.sh <workdir>
#
# `<workdir>` must already hold what run_pa_madspin.sh made.
#
# 1. `integration` -- the generate_events path.  The sweep runs the standalone
#    MadSpin front end, where the decayed file is written next to its input as
#    <input>_decayed.lhe.gz.  That is not where a `launch`-driven run puts it:
#    there the decayed events have to end up ON TOP OF
#    Events/<run>/unweighted_events.lhe.gz, because that is the file every
#    downstream step reads.  Same card, same mode, through generate_events.
#
# 2. `fixedorder` -- `set fixed_order True`, the other path PR #363 shipped
#    unvalidated.  A real fixed-order sample means an aMC@NLO NLO generation
#    and compile, which this does not do.  Instead the production sample is
#    rewrapped into <eventgroup> blocks -- one born event plus one
#    counter-event per group -- which is the format `fixed_order` actually
#    reads (EventFile.next_eventgroup).  That exercises the plumbing: the
#    per-group unpacking, `full_evt = [full_evt] + [counter-events]`, the
#    pure-interference weight applied to every member, and the fact that the
#    zero-cross-section sum takes member 0 only.  It does NOT exercise the
#    physics of NLO subtraction, and the counter-events here carry the born
#    kinematics, so read the result as "the code path runs and the weights land
#    where they should", not as a validation of MadSpin on fixed-order events.
set -e

DRIVER=${DRIVER:-$(dirname "${BASH_SOURCE[0]}")/drive_madspin.py}
PYTHON=${PYTHON:-python3}
WORK=${1:?usage: run_extras.sh <workdir>}
N=${2:-2000}
NB_CORE=${NB_CORE:-6}

EVT="$WORK/prod/Events/prod/unweighted_events.lhe.gz"
[ -f "$EVT" ] || { echo "no production sample at $EVT"; exit 1; }

card_body() {
  echo "set seed 9101"
  echo "set spinmode PA"
  echo "set max_weight_ps_point 100"
  echo "set BW_cut 15"
  echo "set nb_core $NB_CORE"
  echo "set pure_interference t  = + -"
  echo "set pure_interference t~ = + -"
  echo "set pure_interference_output weighted"
  echo "define lp = e+ mu+"
  echo "define lm = e- mu-"
  echo "define vl = ve vm"
  echo "define vlx = ve~ vm~"
  echo "decay t > b w+, w+ > lp vl"
  echo "decay t~ > b~ w-, w- > lm vlx"
  echo "launch"
}

# ------------------------------------------------------- 1. generate_events
d="$WORK/prod"
if [ ! -f "$WORK/integration_done" ]; then
  perl -pi -e "s/^(\s*)\S+(\s*=\s*nevents\b)/\${1}$N\${2}/" "$d/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*iseed\b)/\${1}4321\${2}/" "$d/Cards/run_card.dat"
  card_body > "$d/Cards/madspin_card.dat"
  set +e
  ( cd "$WORK" && "$d/bin/generate_events" -f pa_integration \
        > "$WORK/log_pa_integration.txt" 2>&1 )
  echo "$?" > "$WORK/integration_exit_code"
  set -e
  touch "$WORK/integration_done"
fi
echo "--- generate_events run: exit $(cat "$WORK/integration_exit_code")"
ls -l "$d/Events/pa_integration/" || true

# ------------------------------------------------------------ 2. fixed_order
o="$WORK/fixedorder"
if [ ! -f "$o/events_decayed.lhe.gz" ]; then
  mkdir -p "$o"
  "$PYTHON" - "$EVT" "$o/events.lhe.gz" "$N" <<'PY'
"""Rewrap an ordinary LHE into the <eventgroup> format fixed_order reads.

Each group is the born event plus one counter-event: a verbatim copy whose
XWGTUP is scaled by -0.3.  That is not a physical subtraction term -- it is a
second member with a different weight, which is all the MadSpin code path
needs to be exercised (it decays every member with the SAME decay and applies
the same branching-ratio/interference factor to each).
"""
import gzip, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = gzip.open(dst, 'wt')
buf, in_ev, kept, done = [], False, 0, False
for line in gzip.open(src, 'rt'):
    if done:
        continue
    if line.startswith('<event'):
        if kept >= n:
            out.write('</LesHouchesEvents>\n'); done = True; continue
        in_ev, buf = True, [line]
        continue
    if line.startswith('</event'):
        in_ev = False
        kept += 1
        body = buf[1:]
        head = body[0].split()
        counter = list(body)
        head2 = list(head)
        head2[2] = '%.10e' % (-0.3 * float(head[2]))
        counter[0] = ' ' + ' '.join(head2) + '\n'
        out.write('<eventgroup>\n')
        for ev in (body, counter):
            out.write('<event>\n')
            out.writelines(ev)
            out.write('</event>\n')
        out.write('</eventgroup>\n')
        continue
    if in_ev:
        buf.append(line)
    else:
        out.write(line)
out.close()
print('wrote %d event groups to %s' % (kept, dst))
PY
  {
    echo "import $o/events.lhe.gz"
    echo "set fixed_order True"
    card_body
  } > "$o/madspin_card.dat"
  set +e
  MS_PA_JACLOG="$o/jac" "$PYTHON" -O "$DRIVER" "$o/madspin_card.dat" \
      > "$o/madspin.log" 2>&1
  echo "$?" > "$o/exit_code"
  set -e
fi
echo "--- fixed_order run: exit $(cat "$o/exit_code")"
ls -l "$o" | head

# 2b. the same fixed-order file with pure_interference OFF and
# pure_interference_output at its default, one run per spinmode.  This is the
# control that decides whether anything wrong under `fixed_order` belongs to
# PR #363 at all: with the mode off, none of #363's code runs.
for mode in onshell PA madspin; do
  o="$WORK/fo_ctrl_$mode"
  [ -f "$o/events_decayed.lhe.gz" ] && { echo "skip fo_ctrl_$mode"; continue; }
  mkdir -p "$o"
  cp "$WORK/fixedorder/events.lhe.gz" "$o/events.lhe.gz"
  {
    echo "import $o/events.lhe.gz"
    echo "set fixed_order True"
    echo "set seed 9202"
    echo "set spinmode $mode"
    echo "set max_weight_ps_point 100"
    echo "set BW_cut 15"
    echo "set nb_core $NB_CORE"
    echo "define lp = e+ mu+"
    echo "define lm = e- mu-"
    echo "define vl = ve vm"
    echo "define vlx = ve~ vm~"
    echo "decay t > b w+, w+ > lp vl"
    echo "decay t~ > b~ w-, w- > lm vlx"
    echo "launch"
  } > "$o/madspin_card.dat"
  set +e
  "$PYTHON" -O "$DRIVER" "$o/madspin_card.dat" > "$o/madspin.log" 2>&1
  echo "$?" > "$o/exit_code"
  set -e
  echo "--- fo_ctrl_$mode: exit $(cat "$o/exit_code")"
done

echo "extras done"
