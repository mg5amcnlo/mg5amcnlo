#!/bin/bash
# Cross-check that the normalisation constant
#
#     c = <W>_decay-phase-space = eff * max_weight
#
# is the same for a braced production process as for the unpolarised one.  The
# closure test needs c to be a decay-side constant (the production density
# matrix cancels between the restricted contraction and its normalising trace),
# and the parents of the (I,D+-) / (D+-,I) blocks DO carry a brace on one leg,
# so it is worth measuring rather than arguing.
#
#   bash MadSpin/validation/interference_closure/run_c_check.sh <workdir> [nevents]
#
# Both samples run in the ORDINARY (redraw) mode with an explicit
# 'set unweighting joint', which is what makes the unweighting efficiency equal
# c / max_weight.
set -e

MG5=${MG5:-./bin/mg5_aMC}
WORK=${1:?usage: run_c_check.sh <workdir> [nevents]}
N_EVENTS=${2:-5000}
NB_CORE=${NB_CORE:-4}

mkdir -p "$WORK"

SAMPLES=(
  "c_tbp|p p > t t~{+}|4401|8801"
  "c_tp|p p > t{+} t~|4402|8802"
)

{
  echo "set acknowledged_v3.1_syntax True --no_save"
  for s in "${SAMPLES[@]}"; do
    IFS='|' read -r tag proc seed msseed <<< "$s"
    echo "generate $proc"
    echo "output $WORK/$tag -f"
  done
} > "$WORK/gen.mg5"

if [ ! -f "$WORK/.generated" ]; then
  $MG5 "$WORK/gen.mg5" > "$WORK/gen.log" 2>&1
  touch "$WORK/.generated"
fi

for s in "${SAMPLES[@]}"; do
  IFS='|' read -r tag proc seed msseed <<< "$s"
  d="$WORK/$tag"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*nevents\b)/\${1}$N_EVENTS\${2}/"  "$d/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*iseed\b)/\${1}$seed\${2}/"        "$d/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*use_syst\b)/\${1}False\${2}/"     "$d/Cards/run_card.dat"
  {
    echo "set seed $msseed"
    echo "set spinmode onshell"
    echo "set max_weight_ps_point 400"
    echo "set BW_cut 15"
    echo "set unweighting joint"
    echo "define lp = e+ mu+"
    echo "define lm = e- mu-"
    echo "define vl = ve vm"
    echo "define vlx = ve~ vm~"
    echo "decay t > b w+, w+ > lp vl"
    echo "decay t~ > b~ w-, w- > lm vlx"
    echo "launch"
  } > "$d/Cards/madspin_card.dat"
  printf 'run_mode = 2\nnb_core = %s\nautomatic_html_opening = False\n' "$NB_CORE" \
      >> "$d/Cards/me5_configuration.txt"
  ( "$d/bin/generate_events" -f cchk > "$WORK/log_$tag.txt" 2>&1 ) &
done
wait

for s in "${SAMPLES[@]}"; do
  IFS='|' read -r tag proc seed msseed <<< "$s"
  echo "== $tag ($proc)"
  grep -h "joint maximum weight\|unweight efficiency" "$WORK/log_$tag.txt"
done
