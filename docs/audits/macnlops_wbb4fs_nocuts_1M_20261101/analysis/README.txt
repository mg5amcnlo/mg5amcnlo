No-cut W+ b bbar four-flavour campaign analysis

Core files are shared between first-emission LHE and full shower:
  wbb_observables.cc, jet_tools.py, observables.json, py8an_HwU_wbb.f.
They require one stable W+, b and bbar and retain every signed event.
No event acceptance, jet-pT or rapidity cuts are applied. Anti-kT R=0.5
jets are diagnostics. Radiation jets omit the stable b/bbar; all-parton jets
include them. Absent jets have pT=0. Undefined rapidity coordinates do not
reject events. Primary radiation variable is pT(W+b+bbar).

LHE stage:
  python analysis/diag_weights.py INPUT.lhe.gz BACKUP.lhe.gz ARRAYS.npz SUMMARY.json Wbb
The matching-weight input is retained under BACKUP only after output validation.
Two extra weights are added: S veto only and H removal only. All events retain
their nonzero signed nominal weight, including those with zero matched weight.
The saved arrays include source indices, all observables, S/H type, original
and matched weights, decision rule, and keep probability.

Shared-library example:
  g++ -O2 -std=c++17 -shared -fPIC -I ROOT/Template/NLO/SubProcesses \
      wbb_observables.cc ROOT/Template/NLO/SubProcesses/fjcore.cc \
      -o wbb_observables.so

Full-shower analysis objects:
  HwU.o py8an_HwU_wbb.o wbb_observables.o fjcore.o
The HwU hook writes wbb_observables.bin and wbb_weight_names.txt in each
shower job directory. Preserve those files together with MADatNLO.HwU,
Pythia8.cmd and the run logs before subsequent stages reuse directories.
Binary format: two native int32 values (19, number_of_weights), followed by
float64 rows [19 observables, all weights] for every original input event.
Weight labels in the text file are ordered exactly as the binary columns.

First/full postprocessing and three-page plot book:
  python analysis/analyse.py --first ARRAYS.npz --full-root RUN_PYTHIA8_N \
      --out OUTPUT_DIRECTORY
Job subdirectories run_1, run_2, ... are ordered numerically. The reader
checks full weights against first-emission records event by event, then
checks every HwU bin against binary-stream sums using explicit printed
rounding bounds. OUTPUT_DIRECTORY retains full-event NPZ, comparison.json
and wbb_nocuts_validation.pdf. Statistics use paired shower/source chunks;
all original signed weights are used without survivor/range normalization.
All missing, underflow, overflow and zero-observable weights are accounted
separately. Conditional expected transfers use recorded keep probabilities
and are distinct from the realized matching comparison.

Independent kinematic and Fortran ABI checks:
  python analysis/check_analysis.py

The invalid initial continuation-configuration preflight is archived under
results/pilot_full_configuration_preflight. The corrected, complete 500-event
pilot is archived under results/pilot_full and results/pilot_validation.
Neither pilot is the 1M-event production validation.

Full-shower diagnostic lifecycle correction (main 1M sample):
Native Pythia83.cc calls PYAEND at intermediate 5000-event checkpoints as well
as at completion. PYAEND must flush unit 97, not close it. The original main
RUN_PYTHIA8_3 retained only the initial 5001 stream rows and final 4999 rows
in implicitly reopened sequential fort.97; native shower and HwU accumulation
still processed all 20000 events/job. Original outputs/source are archived at
results/full_1M_incomplete_streams/. A diagnostic-only isolated replay lives at
results/full_1M_replay/, using identical ordered input events and original job
cards. See replay_proof.json for all-job HwU byte identity and stream counts.
This change affects output lifecycle only, with no physics/observable changes.
