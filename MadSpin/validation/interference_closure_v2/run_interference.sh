#!/bin/bash
# Generate the ten samples of the MadSpin *interference* closure test on
# p p > t t~, second run -- on the reworked pure-interference interface.
# Driven from the repository root:
#
#   bash MadSpin/validation/interference_closure_v2/run_interference.sh <workdir> [nevents] [n_int]
#
# The nine blocks of the (h_t, h_tbar) production density matrix, and the
# samples that carry them.  Everything with an I index is now named DIRECTLY
# from the card -- including (I,I), which the first run of this test had to
# obtain by subtraction because the ';' spelling of a two-particle request was
# silently truncated.  The four diagonal-diagonal blocks are not interference
# at all (_validate_pure_interference requires at least one disjoint pair), so
# they still come from production braces, exactly as in the polarisation
# closure test.
#
#   block        production           madspin card
#   (D+,D+)      p p > t{+} t~{+}     --
#   (D+,D-)      p p > t{+} t~{-}     --
#   (D-,D+)      p p > t{-} t~{+}     --
#   (D-,D-)      p p > t{-} t~{-}     --
#   (I ,D+)      p p > t t~           pure_interference t = + - ; t~ = + +
#   (I ,D-)      p p > t t~           pure_interference t = + - ; t~ = - -
#   (D+,I )      p p > t t~           pure_interference t = + + ; t~ = + -
#   (D-,I )      p p > t t~           pure_interference t = - - ; t~ = + -
#   (I ,I )      p p > t t~           pure_interference t = + - ; t~ = + -
#
# (written as one 'set' line per particle -- they accumulate; ';' on one line
# raises.)  All five interference samples therefore run on the SAME unpolarised
# production process, each from its own independent MadEvent run, so there is
# no relative normalisation between them to establish and no braced parent
# whose cross-section has to cancel.
#
# The output is fully weighted: w = sigma_ref * BR * W / c, every trial kept, so
# sum_bin(w)/N is the interference contribution to that bin in pb.  Nothing has
# to be reconstructed from a log.
set -e

MG5=${MG5:-./bin/mg5_aMC}
WORK=${1:?usage: run_interference.sh <workdir> [nevents] [n_int]}
N_EVENTS=${2:-50000}          # reference + the four diagonal samples
N_INT=${3:-$N_EVENTS}         # the five interference samples
NB_CORE=${NB_CORE:-2}
BATCH=${BATCH:-4}             # samples generated concurrently

mkdir -p "$WORK"

# tag | process | madspin-extra-settings (\n separated) | me_seed | ms_seed | nevents
SAMPLES=(
  "unpol|p p > t t~||4321|7777|$N_EVENTS"
  "pp|p p > t{+} t~{+}||4322|7778|$N_EVENTS"
  "pm|p p > t{+} t~{-}||4323|7779|$N_EVENTS"
  "mp|p p > t{-} t~{+}||4324|7780|$N_EVENTS"
  "mm|p p > t{-} t~{-}||4325|7781|$N_EVENTS"
  "i_dp|p p > t t~|set pure_interference t = + -\nset pure_interference t~ = + +|4326|7782|$N_INT"
  "i_dm|p p > t t~|set pure_interference t = + -\nset pure_interference t~ = - -|4327|7783|$N_INT"
  "dp_i|p p > t t~|set pure_interference t = + +\nset pure_interference t~ = + -|4328|7784|$N_INT"
  "dm_i|p p > t t~|set pure_interference t = - -\nset pure_interference t~ = + -|4329|7785|$N_INT"
  "ii|p p > t t~|set pure_interference t = + -\nset pure_interference t~ = + -|4330|7786|$N_INT"
)

# ---------------------------------------------------------------- generation
{
  echo "set acknowledged_v3.1_syntax True --no_save"
  for s in "${SAMPLES[@]}"; do
    IFS='|' read -r tag proc extra seed msseed nev <<< "$s"
    echo "generate $proc"
    echo "output $WORK/$tag -f"
  done
} > "$WORK/gen.mg5"

if [ ! -f "$WORK/.generated" ]; then
  $MG5 "$WORK/gen.mg5" > "$WORK/gen.log" 2>&1
  touch "$WORK/.generated"
fi

# --------------------------------------------------------------------- cards
for s in "${SAMPLES[@]}"; do
  IFS='|' read -r tag proc extra seed msseed nev <<< "$s"
  d="$WORK/$tag"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*nevents\b)/\${1}$nev\${2}/"     "$d/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*iseed\b)/\${1}$seed\${2}/"      "$d/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*use_syst\b)/\${1}False\${2}/"   "$d/Cards/run_card.dat"
  {
    echo "set seed $msseed"
    echo "set spinmode onshell"
    echo "set max_weight_ps_point 400"
    echo "set BW_cut 15"
    [ -n "$extra" ] && printf '%b\n' "$extra"
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
done

# ------------------------------------------------------------------ generate
i=0
for s in "${SAMPLES[@]}"; do
  IFS='|' read -r tag proc extra seed msseed nev <<< "$s"
  if [ -f "$WORK/$tag/Events/closure/unweighted_events.lhe.gz" ]; then
      echo "skip $tag (already done)"; continue
  fi
  ( "$WORK/$tag/bin/generate_events" -f closure > "$WORK/log_$tag.txt" 2>&1 ) &
  i=$((i+1))
  if [ $((i % BATCH)) -eq 0 ]; then wait; fi
done
wait

echo "all samples done"
