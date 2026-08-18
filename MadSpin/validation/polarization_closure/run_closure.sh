#!/bin/bash
# Generate the five samples of the MadSpin production-polarisation closure test
# on p p > t t~ (see README.md).  Everything is driven from the repository root.
#
#   bash MadSpin/validation/polarization_closure/run_closure.sh <workdir>
#
# The polarised processes have to be five *separate* MadEvent outputs: each
# polarisation is its own amplitude and cannot be extracted from a single run.
set -e

MG5=${MG5:-./bin/mg5_aMC}
WORK=${1:?usage: run_closure.sh <workdir>}
SEED=${SEED:-4321}          # MadEvent random seed, identical for all samples
MSSEED=${MSSEED:-7777}      # MadSpin random seed, identical for all samples

# N proportional to sigma: the like-helicity samples carry twice the cross
# section of the opposite-helicity ones (and decay ~5x faster in the MadSpin
# accept/reject), so this keeps the per-event weight nearly uniform across the
# four samples that get summed.
N_UNPOL=50000
N_LIKE=50000                # t{+}t~{+} and t{-}t~{-}
N_OPP=20000                 # t{+}t~{-} and t{-}t~{+}

mkdir -p "$WORK"

cat > "$WORK/gen.mg5" <<EOF
set acknowledged_v3.1_syntax True --no_save
generate p p > t t~
output $WORK/unpol -f
generate p p > t{+} t~{+}
output $WORK/pp -f
generate p p > t{+} t~{-}
output $WORK/pm -f
generate p p > t{-} t~{+}
output $WORK/mp -f
generate p p > t{-} t~{-}
output $WORK/mm -f
EOF

$MG5 "$WORK/gen.mg5"

write_cards () {                       # $1 = dir, $2 = nevents
  perl -pi -e "s/^(\s*)\S+(\s*=\s*nevents\b)/\${1}$2\${2}/"      "$1/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*iseed\b)/\${1}$SEED\${2}/"     "$1/Cards/run_card.dat"
  perl -pi -e "s/^(\s*)\S+(\s*=\s*use_syst\b)/\${1}False\${2}/"  "$1/Cards/run_card.dat"
  cat > "$1/Cards/madspin_card.dat" <<EOF
set seed $MSSEED
set spinmode madspin
set max_weight_ps_point 400
set BW_cut 15
define lp = e+ mu+
define lm = e- mu-
define vl = ve vm
define vlx = ve~ vm~
decay t > b w+, w+ > lp vl
decay t~ > b~ w-, w- > lm vlx
launch
EOF
  printf 'run_mode = 2\nnb_core = 4\nautomatic_html_opening = False\n' \
      >> "$1/Cards/me5_configuration.txt"
}

write_cards "$WORK/unpol" $N_UNPOL
write_cards "$WORK/pp"    $N_LIKE
write_cards "$WORK/mm"    $N_LIKE
write_cards "$WORK/pm"    $N_OPP
write_cards "$WORK/mp"    $N_OPP

for d in unpol pp mm pm mp; do
  ( "$WORK/$d/bin/generate_events" -f closure > "$WORK/log_$d.txt" 2>&1 ) &
done
wait

python3 MadSpin/validation/polarization_closure/analyse_closure.py \
        "$WORK" MadSpin/validation/polarization_closure/plots closure
