# Global-recoil gluon Sudakov denominator fix

Implemented the gluon contribution to the competing-shower denominator using
stock Pythia 8.318's global-recoil first-step rate. The source damping,
generation weights, original events and installed Pythia are unchanged.
The separate local signed-S/channel-beta issue remains.

## Implementation

`MacNLOPSMatch.cc::gluonFsrRate` integrates both g -> gg and g -> qqbar over
the native finite z support. It includes each colour end's coefficient and
starting cap, the massive-global-recoiler dead-cone weight, allowed daughter
flavours/masses and pair thresholds. A massive pair does not acquire an
extra evolution-pT cutoff absent from the native brancher.

The coupling follows the actual one-loop shower evolution: initialized
Lambda3/4/5 and the native max(1.2,m_c), max(4,m_b) switches. This differs
slightly at low scales from calling the public AlphaStrong::alphaS helper.
Quark and ISR rates retain their existing definitions. The z integration
is adaptive, with a threshold map for massive daughters; log-pT integration
splits at the physical ceiling, colour-end caps and native coupling switches.

`macnlops_c_gluon_profile` supplies each Born gluon's radiator mass, summed
hard-recoiler mass, total hard mass and both intrinsic colour-end caps. It
is a separate getter, preserving the old quark-profile ABI. Gluon table
factors are removed only from the competing-shower denominator; the source
continues to use its original tables. The unused extrapolation helper was
removed.

A native startup condition needs explicit treatment for soft gluons. If
Pythia's initial flavour slab has an empty overestimated z range, it aborts
that colour end before descending. Thus a valid soft gluon can radiate at
Q=3 GeV but be inactive at Q=8 GeV. The denominator reproduces this condition
per Q node and per colour end. Cached single-end primitives retain unequal
caps without applying a global clipping that would change the shower.

When such a startup transition lies inside the starting-Q distribution,
a shower-only quadrature splits at the native flavour floors and global
ceiling. Source averaging keeps its original grid, evaluation order and
normalization. Its mean remains bit-identical; the physical starting-Q law
and its CDF do not change. Nongating Wj profiles retain their old Q nodes.
The 512-node limit fails closed.

## Validation

The final builds use six isolated copies of existing provider objects with
new source files compiled separately. No production executable, old event,
matching weight, original provider or Pythia library was overwritten.

- All **90 targeted tests pass** on the final source. These include compiled
  rate checks against a closed massless-recoiler primitive, independent
  massive-recoiler reference values, both end caps, native coupling switches,
  pair thresholds, startup gates, source/profile reset and source invariance.
- **95 verified saved contexts** reuse 825,000 native trials with independent
  context/provider/component streams. The same pre-existing density-replay
  mismatch remains excluded. All 60 quark-Born denominator values are
  unchanged, and all 96 source means/Q weights/CDFs agree exactly.
- The 35 gluon contexts now have pure-FSR native/predicted survival
  **0.999213 +/- 0.001187**, range 0.98007–1.01479, maximum pull 2.45.
  Their full ISR+FSR ratio is 0.996800 +/- 0.002655, maximum pull 2.43.
  No corrected context has a pull above 3. These selected contexts do not
  estimate a full-population transfer residual.

For the formerly discrepant W+gluon example:

| Probability | Old denominator | Corrected denominator | Native measurement |
|---|---:|---:|---:|
| Full survival | 0.406006 | 0.486385 | 0.475400 +/- 0.007063 |
| Gluon FSR survival | 0.717280 | 0.872585 | 0.879600 +/- 0.004602 |

Additional controls use **1,510,000 new native trials**:

| Control | Trials | Result |
|---|---:|---|
| Three Born contexts, eight low stops 0.51–4.8 GeV, full/FSR | 960,000 | Maximum pull 1.98 |
| Soft-gluon fixed-Q startup gate | 300,000 | Maximum pull 1.20; inactive Q gives survival 1 |
| Soft-gluon continuous starting-Q distribution | 250,000 | All tested stops agree within sampling precision |

The continuous control uses a physical soft W+g Born with gluon energy 7 GeV
and the exact starting distribution on [1,10] GeV. At stop=1 GeV the native
survival is **0.88942 +/- 0.00140**. The corrected split quadrature gives
**0.88940522**; an unsplit 12-node grid would give 0.85389663. A 48-node
reference agrees with the split result. The compiled Fortran fixture
independently checks the PS-only grid and bit-identical source mean/CDF.
Sparse zero-survival observations retain Wilson upper limits rather than
claiming zero relative uncertainty.

The **1,200-event integrated matching pilot** processes 200 events from
each provider: 887 S and 313 H. Native momenta, continuation scales and source
Sudakovs are bit-identical to their original results. Acceptance probabilities
change for 177 events, all in the two gluon-Born providers. The four
quark-Born providers retain their probabilities. This is not a statistical
measurement of the resulting production cross-section shift.

The validation comprises **2,335,000 native trials**, including 825,000
previously saved trials reused as fixed targets. Source/PDF/parameter and
compiled-object bindings are in validation/provenance.json. The 90-test log
is logs/targeted_tests_final.txt. Native sampling errors do not include hard
integration, beta, context-selection or unmeasured ISR-envelope effects.
The normal matching endpoint-identity guard remains intact; synthetic
low-scale CDF requests use an explicitly separate diagnostic sidecar.

## Scope and reproduction

This fixes the gluon no-emission denominator. It does not replace the local
signed-S density approximation, certify all remaining matching-kernel
approximations, or establish full-shower transfer closure. No new 1M corrected
matching/full-shower production is included. The historical multijet results
remain bound to their original source and matcher epochs.

Full scripts, source snapshots, build commands, raw empirical trials,
provider bindings and diagnostic outputs remain at:
`/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/gluon_fix`.

The compact repository artifact includes numerical comparisons, provenance,
compiled-test output and plots. snapshot.json pins the retained and external
files. The implementation patch is relative to the pre-fix fixed-beam/
compact-Born source epoch; it does not include the earlier unrelated changes.
