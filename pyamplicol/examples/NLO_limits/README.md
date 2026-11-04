# Local NLO limit stability

`nlo_limit_stability_tests.py` is a standalone, serial example using the public
batched correlation API. It compares the full-colour `g g > g g g g` real
matrix element with correlated `g g > g g g` Born counterterms at 1 TeV and
`alpha_s = 1/8`. It does not import the paper's study scripts.

From a checkout with pyAmpliCol, Symbolica and Matplotlib available:

```sh
python examples/NLO_limits/nlo_limit_stability_tests.py --self-test
python examples/NLO_limits/nlo_limit_stability_tests.py --prepare-only
python examples/NLO_limits/nlo_limit_stability_tests.py --smoke \
  --formats double double-double --oracle-precision 160 \
  --output .artifacts/nlo-limits/smoke.json
python examples/NLO_limits/nlo_limit_stability_tests.py
python examples/NLO_limits/nlo_limit_stability_tests.py --render
```

Run the smoke test before the full scan. Evaluation and generation run under
the repository's process-tree memory watchdog, with a **10 GiB** default limit;
generation uses one worker and one optimization core. There are no native
builds. Existing compatible process artifacts and completed result batches
are reused. `--render` only reads retained results and creates the two-panel
PDF/PNG. Defaults are under `.artifacts/nlo-limits`; `--artifact-dir`,
`--output`, and `--plot` select different locations.

The ordinary real process retains default certified current reuse and its
normal compiled colour/helicity replay. Correlated Born generation disables
those reductions to preserve the complete source basis. The isolated precision
adapter covers the real process's exact colour-replay arithmetic as well as
the correlated sources and contractions.

`--formats`, `--limits`, and `--exponents` select a smaller scan. The reference
defaults to 1200 decimal digits (`--oracle-precision`); a smaller smoke oracle
is allowed when it exceeds every selected target precision. A checkpoint is saved
after each completed format/limit batch. A failed real-emission batch is
bisected diagnostically, retaining its valid points; this exceptional fallback
is recorded in `real_batch`. Measured failures are preserved on resume unless
`--retry-failed` is explicit. Ordinary successful evaluation is batched.
Completing a scan requires successful oracle measurements for every requested
point; missing or failed references cause a nonzero exit after checkpointing.
Target-precision failures remain stability data, and `--render` remains
available for partial results.
Results record the Git revision, artifact IDs, batch call counts, and elapsed
times. `--prepare-only` generates/reuses the two artifacts without starting a scan.

## Named results, not scalar correlation loops

For each approach-point batch the script makes one Born call containing all
requested dipoles, the ordinary Born, and the literal spin projection:

```python
from pyamplicol import CorrelatedRequest

requests = {
    "born": CorrelatedRequest("born", spin_vectors={}),
    "spin": CorrelatedRequest("born", spin_vectors={3: spin_vectors_by_point}),
    "T13": CorrelatedRequest("T13", spin_vectors={}),
    # The study also includes the other nine off-diagonal Born dipoles.
}
results = born.evaluate_correlated_many(born_points, requests, precision=1000)
real_part = results["T13"][point_index].real
imaginary_part = results["T13"][point_index].imag
point_components = {name: values[point_index] for name, values in results.items()}
```

Every tuple follows the input phase-space-point order. Request labels such as
`"spin"` are user-defined output names; `"born"` and `"T13"` inside each request
are generation-time colour IDs. Empty `spin_vectors={}` explicitly requests
ordinary physical helicities, independently of any setter state. The same
fixed Born is repeated across the trajectory; the real process is evaluated
as its own batch. Checkpointed components retain both decimal parts.

## Counterterms and mappings

The metric is `(+,-,-,-)`. The collinear trajectory uses the inverse final-final
Catani--Seymour map at fixed `z=3/5`, with `kT=delta*1000 GeV`. One generic,
rational transverse direction is used, **without azimuthal averaging**. With
`e=k_perp/kT`, `a=z/(1-z)+(1-z)/z`, and `CA=3`,

```text
C_collinear = (8*pi*alpha_s/s_ij) * 2*CA * [a*B + 2*z*(1-z)*B(e)]
```

`B(e)` is the literal vector replacement on Born leg 3. A five-gluon Born is
used so this test resolves nontrivial azimuthal spin interference. The smoke
test checks that the selected spin projection is not simply `B/2`.

The separate soft trajectory has `q=delta*r` along a fixed wide-angle null
direction. An inverse-map recoil keeps the mapped Born fixed and all real
momenta on shell. Using mapped Born momenta `P_a` consistently,

```text
C_soft = -8*pi*alpha_s * sum_(a<b) [(P_a.P_b)/((P_a.q)*(P_b.q)) * B_ab]
```

All five Born gluons enter the ten dipoles, including incoming legs. Their
colour charges are already crossed; there is no additional incoming sign.
Massless diagonal eikonal terms vanish. Public final-state symmetry factors
are removed (`R=24*R_public`, every Born object is `6*B_public`), while the
common initial-state spin/colour averages remain. No flux or phase-space
Jacobian is included in this pointwise comparison.

These are two isolated leading-power limit tests, **not a complete NLO
subtraction scheme**. Do not add the two counterterms together over arbitrary
phase space. Generically `R,C~delta^-2`, `C-R~delta^-1`, and `(C-R)/R~delta`;
the unaveraged remainder need not approach a finite value. The divergent
remainder is integrable with the unresolved phase-space measure.

The formulae follow [Catani--Seymour, equations 5.3--5.6](https://arxiv.org/pdf/hep-ph/9605323)
and [Catani--Grazzini, equations 7, 12 and 87--89](https://arxiv.org/pdf/hep-ph/9908523).
See also [the correlation conventions](../../docs/correlator-conventions.md).

## Arithmetic and plotted quantity

The three curves use actual binary64, Symbolica DoubleFloat with 31 decimal
digits retained through Python transfers, and guarded arbitrary precision
with 1000 output digits. This is **not** a relabelling of public
`precision=16`/`precision=32`: those correlated calls normally use guarded
multiprecision. The study's isolated context temporarily substitutes scalar
operations and stage precision in explicitly listed exact-runtime modules,
then restores them. Real binary64 uses the ordinary native evaluator; its
correlated Born uses the explicit `evaluate_complex` binary64 entry point for
retained Symbolica kernels, not `evaluate_complex_with_prec(..., 16)`.
DoubleFloat is used in the sources, kernels, normalization, colour reduction,
counterterm and final subtraction. No public runtime defaults are changed.
The constant pi is constructed accurately and rounded once into the target
arithmetic. These adapters intentionally support the fixed built-in-SM study:
an unexpected model-parameter expression derivation fails explicitly instead
of silently performing part of a double-double calculation in multiprecision.

Kinematics are constructed with guard digits before target-arithmetic
rounding; no high-precision input passes through a Python float. The source
`s_ij` is constructed before rounding, rather than recovered from a cancelled
binary64 dot product. The reference uses the same unrounded source family,
so the reported error includes input rounding, evaluation and subtraction.

For `F=C-R`, the plots show
`min(1, max(0, -log10(abs((F-F_reference)/F_reference))) / tracked_digits)`.
Zero means no retained digits, not a zero matrix element. Failed targets get
zero only when a valid reference exists; absent or zero-remainder references
are not turned into false stability measurements. Each panel spans
`-log10(delta)=1,...,15`; lines only connect evaluated points.
