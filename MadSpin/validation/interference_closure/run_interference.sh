#!/bin/bash
# Generate the eleven samples of the MadSpin *interference* closure test on
# p p > t t~ (see README.md).  Driven from the repository root:
#
#   bash MadSpin/validation/interference_closure/run_interference.sh <workdir> [nevents]
#
# The nine blocks of the (h_t, h_tbar) production density matrix, and the
# samples that carry them:
#
#   block                       process                 madspin card
#   (D+ , D+)   t{+} t~{+}      p p > t{+} t~{+}        --
#   (D+ , D-)   t{+} t~{-}      p p > t{+} t~{-}        --
#   (D- , D+)   t{-} t~{+}      p p > t{-} t~{+}        --
#   (D- , D-)   t{-} t~{-}      p p > t{-} t~{-}        --
#   (I  , D+)                   p p > t   t~{+}         pure_interference t  = + -
#   (I  , D-)                   p p > t   t~{-}         pure_interference t  = + -
#   (D+ , I )                   p p > t{+} t~           pure_interference t~ = + -
#   (D- , I )                   p p > t{-} t~           pure_interference t~ = + -
#   (I  , I )                   -- by subtraction, see below
#
# The last block would need TWO pure_interference entries in one card, i.e.
#   set pure_interference t = + - ; t~ = + -
# which cannot be written: extended_cmd.Cmd.precmd splits every card line on
# ';' and executes the pieces as separate commands, so only the first entry
# ever reaches the option (silently).  It is obtained instead as
#
#   (I,I) = (I , anything) - (I , D+) - (I , D-)
#
# where (I, anything) is a pure_interference run on the *unpolarised*
# production with no brace on the antitop: the per-particle restriction of the
# other leg is then None, i.e. the full 2x2 block.  The two routes
#
#   x_t  = (I,D+) + (I,D-) + (I,I)          p p > t t~,  pure_interference t
#   x_tb = (D+,I) + (D-,I) + (I,I)          p p > t t~,  pure_interference t~
#
# are both generated, which gives two independent determinations of the
# nine-term total and a direct cross-check on (I,I).
set -e

MG5=${MG5:-./bin/mg5_aMC}
WORK=${1:?usage: run_interference.sh <workdir> [nevents]}
N_EVENTS=${2:-50000}
NB_CORE=${NB_CORE:-2}
BATCH=${BATCH:-4}            # samples generated concurrently

mkdir -p "$WORK"

# tag  process  madspin-extra-settings  me_seed  ms_seed
SAMPLES=(
  "unpol|p p > t t~|set unweighting joint|4321|7777"
  "pp|p p > t{+} t~{+}||4322|7778"
  "pm|p p > t{+} t~{-}||4323|7779"
  "mp|p p > t{-} t~{+}||4324|7780"
  "mm|p p > t{-} t~{-}||4325|7781"
  "i_tbp|p p > t t~{+}|set pure_interference t = + -|4326|7782"
  "i_tbm|p p > t t~{-}|set pure_interference t = + -|4327|7783"
  "i_tp|p p > t{+} t~|set pure_interference t~ = + -|4328|7784"
  "i_tm|p p > t{-} t~|set pure_interference t~ = + -|4329|7785"
  "x_t|p p > t t~|set pure_interference t = + -|4330|7786"
  "x_tb|p p > t t~|set pure_interference t~ = + -|4331|7787"
)

# ---------------------------------------------------------------- generation
{
  echo "set acknowledged_v3.1_syntax True --no_save"
  for s in "${SAMPLES[@]}"; do
    IFS='|' read -r tag proc extra seed msseed <<< "$s"
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
  IFS='|' read -r tag proc extra seed msseed <<< "$s"
  d="$WORK/$tag"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*nevents\b)/\${1}$N_EVENTS\${2}/"  "$d/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*iseed\b)/\${1}$seed\${2}/"        "$d/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*use_syst\b)/\${1}False\${2}/"     "$d/Cards/run_card.dat"
  {
    echo "set seed $msseed"
    echo "set spinmode onshell"
    echo "set max_weight_ps_point 400"
    echo "set BW_cut 15"
    [ -n "$extra" ] && echo "$extra"
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
  IFS='|' read -r tag proc extra seed msseed <<< "$s"
  if [ -f "$WORK/$tag/Events/closure/unweighted_events.lhe.gz" ]; then
      echo "skip $tag (already done)"; continue
  fi
  ( "$WORK/$tag/bin/generate_events" -f closure > "$WORK/log_$tag.txt" 2>&1 ) &
  i=$((i+1))
  if [ $((i % BATCH)) -eq 0 ]; then wait; fi
done
wait

echo "all samples done"
