# Interference in madmatrix and madspace: status and plan

Status checked on 2026-10-09 against `main` at 44b8e196cc, with madspace built
from that tree. PR #243 (`[treextree]` / `[LIxtree=QCD]`, still open) was
checked out on its own for the treextree probe. Line numbers drift.

"Interference" here means three different things:

1. **Squared-order selection**: `QCD^2==2`, `QED^2==2`, `^2<=`, `^2>`. When the
   constraint cannot be met by dropping diagrams, it needs the jamps split by
   amplitude order.
2. **`left [treextree] right`** (PR #243): two tree amplitudes. A hidden `INTERF`
   order tags the right-hand diagrams and `INTERF^2==1` is selected, so it is a
   case of (1).
3. **`left [LIxtree=QCD] right`** (PR #243): a loop-induced amplitude times a
   tree. This needs MadLoop.

## 1. Status

| | madevent | madmatrix CPU (`standalone`, `mg7`) | madmatrix GPU | madspace (mg7 run) |
|---|---|---|---|---|
| squared-order selection | yes | **yes** (PR #125) | refused when compiling (`static_assert nampso == 1`) | **works**: signed weights handled |
| `[treextree]` | yes (PR #243) | **works once allowed** (PR #243 refuses the format) | no | **works once allowed** |
| `[LIxtree]` | yes (PR #243) | no (madmatrix has no loops) | no | n/a |

madspace was already designed for signed integrands:
- every channel keeps a signed accumulator and an `|w|` accumulator (`channel_generator.cpp:364-371`);
- VEGAS fills its grid with `w²` (`cpu/runtime.cpp:926`);
- the survey stop, batch sizing and VEGAS patience all use the `|w|` accumulator;
- unweighting accepts on `|w|` and restores the sign with `copysign` (`cpu/runtime.cpp:789-794`, GPU the same);
- the MadNIS loss uses `|f|`;
- events go out at `±σ_abs`, and their mean is σ;
- systematics ratios are multiplicative and keep the sign.

No code path crashes or produces NaN on a negative weight.

## 2. Measurements

All at 13 TeV with lhaid 331900 and the default cuts unless stated. Same seed
per pair of runs.

| process | mg7 (madmatrix + madspace) | madevent |
|---|---|---|
| `p p > u u~ QCD^2==2` (5k ev.) | -9481(84) pb, 85% negative events | -9532(63) pb |
| `p p > j j QCD^2==2` (10k ev.) | 5.283(87)e4 pb | 5.263(53)e4 pb |
| `p p > z > e+ e- [treextree] p p > a > e+ e-`, m_ll > 200 | 0.0600(36) pb (20k), 0.0659(26) (50k) | 0.0596(61) pb |

**treextree matrix element, fixed point.** `u u~ > z > e+ e- [treextree] u u~ > a > e+ e-`,
classic RAMBO point at 1000 GeV, `FPTYPE=d`. To run it, `standalone` was added to
`TREE_INTERFERENCE_FORMATS`; no other change.
- madmatrix 1.2397455147009457e-03
- Fortran (`standalone_fortran`) 1.2397455147009446e-03
- madmatrix |Z+γ|² − |Z|² − |γ|² = 1.2397455147009459e-03

**Helicity column of the treextree events: wrong in mg7.** The table gives the
signed σ_i in pb per (incoming-quark helicity, e⁻ helicity), as sum(w)/N:

| class | madevent (5k) | mg7 (50k) |
|---|---|---|
| (−,−) | +0.476(10) | +0.335(2) |
| (−,+) | −0.381(9) | −0.227(2) |
| (+,−) | −0.151(7) | −0.119(1) |
| (+,+) | +0.124(5) | +0.076(1) |

The totals agree. The per-helicity contributions differ by up to 16σ. madevent
is correct by construction (see D1).

## 3. Defects, most severe first

**D1. Helicity selection uses a signed running sum** (madmatrix).
- Code: `backend/simd/SigmaKin.cc:603,618`, `backend/cpu/SigmaKin.cc` (same lines −1), `backend/gpu/SigmaKin.cc:487-501`.
- It tests `rnd < S_i / S_total`, where S_i is the partial sum of the signed per-helicity |M|².
- With mixed signs the CDF is not monotonic, and a negative total inverts it. It cannot crash (the last ratio is exactly 1), but the helicity written to the event is wrong, as measured above. This matters for anything that reads LHE helicities: τ decays in the shower, polarisation analyses.
- madevent (`matrix_madevent_group_v4.inc:150-222`) does this instead:
  - `ANS = Σ|T_i|`;
  - pick i with probability `|T_i|/Σ|T|`;
  - `ANS = sign(T_i)·Σ|T|`.
- This is exact per helicity. The price is a larger σ_abs: 1.126 pb against mg7's 0.757 pb on the treextree sample, so about 2.2× more events for the same precision on σ.
- For a matrix element that is never negative (everything except interference) the two schemes give identical results.

**D2. The LHE `<init>` line declares positive unit weights** (`launch.py:1724-1726`, and `gridpack.py:76`).
- `IDWTUP` is hard-coded to `+3`, and `XMAXUP = σ`, which is negative for `u u~`.
- madevent writes `-4` and `max|w|` (14270).
- Under the Les Houches Accord only −3/−4 declare negative weights. Whether Pythia8 keeps the sign under `+3` was not checked; it is checked in phase 1.

**D3. The systematics percentages divide by the signed nominal** (`mg7/systematics_summary.py:63-64,83`).
- The run printed `Scale variation: +-24.1% --30%`.
- This is the mg7 twin of the madevent `systematics.py` fix in PR #243 and mg5 #442.

**D4. `qcd_power = -1` for every mixed-order process** (`export_mg7.py:472-484`).
- The power is taken from the diagrams, not from the squared orders that are kept. `QCD^2==2` has a definite α_s¹.
- The result is correct, because systematics re-evaluates the matrix element (`launch.py:860-898`). But the combine stage then runs single-threaded and the matrix element is evaluated twice.
- PR #243 has `export_v4.interference_alpha_s_power` for madevent's `config_nqcd.inc`. One helper should serve both.

**D5. PR #243 refuses `standalone` and `mg7` for `[treextree]`** (`TREE_INTERFERENCE_FORMATS = ['madevent', 'standalone_fortran']`).
- The reason given ("C++/mg7 have no squared split orders") is out of date since PR #125. Section 2 shows the format works.
- `LIxtree` must stay refused there.

**D6. The GPU refusal only comes at compile time.**
- `check_split_orders` (`madmatrix/output.py:248`) only logs an info message.
- The `static_assert` in `backend/gpu/SigmaKin.cc:46-51` fires when `launch` compiles for `device = cuda/hip`.
- The device jamp buffers are sized `ncolor` (`gpu/SigmaKin.cc:224,276`).

**D7. Channel pruning ranks by `|signed mean|`** (`launch.py:1276-1278` → `simplify_phasespace`, `launch.py:2478-2500`).
- A channel with a large |f| that cancels internally is folded into the flat channel.
- The total stays unbiased but the variance grows. It should rank by `status.mean_abs`.
- `tot_cs == 0` divides by zero.

**D8. The MadNIS `BufferUnweighter` takes the quantile of the signed weights** (`madspace/src/phasespace/unweighter.cpp:34`).
- With mixed signs the max weight comes out too small, or ≤ 0, in which case every event is accepted.
- The result stays consistent (`acc_factor`), but the buffer efficiency drops. It should take the quantile of `|w|`.

**D9. Cosmetic.**
- The relative error prints negative (`event_generator.cpp:1445,1530`).
- The weight histogram is in units of the signed σ (`event_histograms.cpp:29-31,156`), while the card says to expect a spike at 1 (`banner.py`, `histogram_weight_max`).
- `plots.py:111-120,163` picks the log axis from positive bins only and hides negative bins. The ratio bands flip when σ < 0 (`:170`).

**D10. Run-card defaults.**
- For `^2` processes, madevent sets `dynamical_scale_choice = 3`, `sde_strategy = 2` and `use_syst = False` (`banner.py:5266-5284`). `RunCardMG7` sets none of these.
- mg7 systematics are correct here, thanks to the matrix-element re-evaluation, so madevent's reason for switching them off does not apply.
- The mg7 defaults reproduce madevent's σ (section 2).

**D11. Tests.**
- Only `test_cmd.py:1336` covers this: standalone `u u~ > u u~ QED^2==2`.
- Nothing tests the mg7 run, signed unweighting, the helicity column, the `<init>` line or the GPU refusal.

Latent and outside this plan: `DiscreteOptimizer` is never enabled. The local
variable at `channel_generator.cpp:170` shadows the member. If it is
re-enabled, it accumulates signed weights and copies an unfilled tensor
(`discrete_optimizer.cpp:38-57`).

## 4. Plan

### Phase 1: signed-weight fixes on main (one PR, CPU)

1. **D1 helicity selection**, in the CPU, SIMD (including mixed precision) and GPU kernels. Recommended: the madevent convention.
   - Keep the per-helicity `|T_i|`.
   - Select on `Σ|T|`.
   - Return `sign(T_i)·Σ|T|` as the matrix element.
   - Only when `nampso > 1`, since otherwise every T_i ≥ 0 and nothing changes. A compile-time branch keeps every other process byte-identical.
   - The systematics re-evaluation (`build_systematics_matrix_elements`) must then use the same helicity. Either pass `selhel` back, or form the ratio from T_i of that helicity. A fresh draw would mix signs. With phase 2 in place, a kept order with a single α_s power no longer re-evaluates at all.
2. **D2**: `weight_mode = -4` when σ < 0 or any event weight is negative, otherwise `3`. `XMAXUP = max|w|` (σ_abs for unit-weight runs). The same in `gridpack.py`. Shower `p p > u u~ QCD^2==2` with Pythia8 before and after, and compare the signed `sigmaGen`.
3. **D3**: percentages against `|nominal|` and `|central|`. Unit test with a negative-total summary.
4. **D7**: `status.mean_abs` in `simplify_phasespace`, and guard `tot_cs == 0`.
5. **D8**: quantile of `|w|` in `BufferUnweighter`, plus a madspace test with mixed-sign weights. This rebuilds madspace, so the source hash changes.
6. **D9**: `|mean|` in the relative-error print; weight histogram in units of σ_abs; linear axis when any bin is negative.
7. **Tests** (acceptance, gated on madspace like the other mg7 tests):
   - mg7 `p p > u u~ QCD^2==2`, 2k events: σ < 0 within 5σ of a pinned reference; `<init>` reads `-4` and `XMAXUP > 0`; the events carry both signs.
   - madmatrix standalone: helicity selection at a mixed-sign point. Run `check_sa.exe perf -v` on many random draws and compare the signed per-helicity sums with the per-helicity matrix elements.

#### Phase 1 status (2026-10-09, uncommitted)

All done; decided convention: (a), madevent's.

- **D1.**
  - Code: `select_helicity_signed` in `backend/{cpu,simd}/SigmaKin.cc`, under `if constexpr( nampso > 1 )`, so nothing changes elsewhere.
  - It is switched on by a new trailing `sigmaKin` argument, `sampleSignedHelicity = false`. umami passes `random_helicity_in != nullptr`, so event generation gets the convention, while systematics re-evaluation, `check_sa` and fbridge keep ΣT.
  - The GPU kernel is left alone: phase 4 needs the same rule.
  - heft `g g > t t~ HIG^2==1` (m_H = 500 GeV, Γ_H = 30 GeV):
    - per-helicity values at 5 mixed-sign lab points equal Fortran `SMATRIXHEL`/4 (the gluon helicity average), with the same labels, to the grid step;
    - σ unchanged: −0.405(16) against −0.402(12) pb before;
    - σ_abs goes from 4.16 to 5.57 pb.
  - The d, f, m (mixed) and scalar builds all compile. The mixed second page is checked too.
- **D1, opt-out (option (c)).** The run card `[generation] interference_helicity = "summed"` turns the convention off. It defaults to `"exact"`.
  - mg7 sends the umami parameter `interference_helicity_summed`, per instance and only when asked for, since older libraries refuse unknown parameters. The matrix element is then the helicity sum, with lower variance.
  - The LHE helicities of the interference subprocesses (flagged by `export_mg7` as `interference`) are written as 9 on every line, through `launch.lhe_helicities`.
- **D2.** `IDWTUP = -4`, not −3, for a sample with negative weights. Pythia8 8.317 takes the cross section as |XSECUP| × ⟨sign⟩ under −3: −6405 pb for a −9572 pb sample. With −4 the showered σ is −9678(75) pb against −9635(40). `XMAXUP = σ_abs`. Positive samples keep +3 and XMAXUP = σ, so nothing changes for them. Whether a positive sample with kept overweights should declare 4 is a separate question.
- **D8, measured.** The fix is checked by `madspace/tests/test_buffer_unweighter.py`, which needed a new Python binding for `BufferUnweighter`. It has synthetic positive, all-negative and mixed batches. With the signed quantile the all-negative batch kept 100% (no unweighting) and the mixed one 94%, against 28.5% for the positive batch; the two signed tests fail on the old code. The rescaled `adaptive_prob` kept the buffer unbiased in all cases.
  - The effect on training is not measurable on our samples. MadNIS was forced on with 4000 batches and the same seed:

    | process | quantile | RSD | unw. eff. | final loss |
    |---|---|---|---|---|
    | `p p > u u~ QCD^2==2` | signed | 0.1081 | 0.658 | 0.502 |
    | `p p > u u~ QCD^2==2` | \|w\| | 0.1085 | 0.649 | 0.501 |
    | heft `g g > t t~ HIG^2==1`, m_H = 500 GeV | signed | 0.0886 | 0.694 | 0.993 |
    | heft `g g > t t~ HIG^2==1`, m_H = 500 GeV | \|w\| | 0.0880 | 0.714 | 0.990 |

    The buffers reach capacity (120k) within about 1200 batches either way.
  - With the default card the buffer is never used: `buffer_skip_batches = 1000` equals `train_batches = 1000`, so the old code had no effect on default runs.
- **D3, D7, D9** as planned. In D9 the weight histogram is in units of σ_abs, with a symmetric range when the process has a squared-order constraint. The xsec acceptance test also divides by |reference| now.
- **Tests:**
  - `test_cmd.py::test_standalone_interference_helicity_choice`: umami via ctypes, `u u~ > t t~ g QED^2==2`. It fails against the old selection.
  - `test_mg7_interference.py`: `p p > u u~ QCD^2==2` against −12253(10) pb, from three mg7 seeds of 200k events. madevent with the same settings gives −12340(42) pb, 0.7% (2σ) away; not chased.
  - The unit additions are in the systematics-summary, plot and banner tests.
- **Also seen.** mg7 hands the matrix element lab-frame momenta (no me_frame boost on main). The helicities of *massive* legs are therefore lab-frame helicities, while madevent's are partonic-CM ones. Per-helicity comparisons with madevent only hold for massless legs. This is pre-existing and not about interference.

### Phase 2: α_s power of the kept squared orders (D4)

- In `export_mg7.py`, compute `qcd_power` from `split_order_tables` (`amp_so`, `sqsoindex`, `chosen`) together with each amplitude's QCD order. The power of a kept pair (m,n) is `(QCD_m+QCD_n)/2`; it is a single value only if all kept pairs share it, otherwise keep −1.
- Share the helper with PR #243's `interference_alpha_s_power` once that is merged, or land it first and rebase #243 on it.
- Check that the systematics weights agree to 1e-12 between the re-evaluation path and the power path on `p p > u u~ QCD^2==2`, and that the combine stage is multi-threaded again.

### Phase 3: `[treextree]` on madmatrix and mg7 (D5)

Depends on PR #243, either as a commit on it or as a follow-up.
- Add `standalone` and `mg7` to `TREE_INTERFERENCE_FORMATS`. Refuse a GPU device at launch (phase 4).
- Decide which of #243's madevent run-card policy (`interference_mode` → scale 3, no systematics) carries over to `RunCardMG7` (D10).
- Tests:
  - the sum rule at a fixed point (measured to 2e-16);
  - madmatrix against `standalone_fortran`;
  - mg7 σ against madevent for the Z/γ case;
  - the per-helicity signed sums against madevent (this catches D1).

### Phase 4: GPU

1. **Now**: refuse early with a clear message. Check in `launch.py` `compile_matrix_elements` when `device ∈ {cuda, hip}` and a subprocess has `nampso > 1`. Have `output` say so too.
2. **Port**:
   - device jamp buffers sized `njampso`;
   - the colour-sum kernel loops over **all ordered** amplitude-order pairs with `sqSoIndex`/`chosenSqso` (the triangle trap of PR #125: a triangle gives about 2×);
   - `add_and_select_hel` uses the phase-1 convention;
   - the BLAS colour sum stays off.
3. Validate on CECI lemaitre4 against the CPU at the same momenta.

#### Phase 4 status (2026-10-09): written, waiting for the GPU CI

No CUDA toolchain was available locally, so the GPU code is not compiled yet.

What was changed:
- **Jamps.** `calculate_jamps` keeps `njampso` jamps. `DeviceAccessJamp`, all the device jamp buffers (`MatrixElementKernels.cc`, `umami.cc`) and the memset use `njampso` as the plane length.
- **Colour sum.** `color_sum_kernel` has the ordered-pair split-order branch, in colour-algebra precision.
- **Tables.** `sqSoIndex` and `chosenSqso` are generated `__device__ constexpr`, as `icolamp` already is (an empty macro on CPU).
- **cuBLAS.** It is refused at runtime for `nampso > 1`, with a warning and an assert.
- **Helicity choice.** `add_and_select_hel` uses the madevent convention behind the same `sampleSignedHelicity` flag, which GPU umami sets from the random-helicity input.
- **Helicity filtering.** GPU umami's filtering now uses the per-helicity jamp buffer as scratch, which removes a limit of 8 orders.
- **Message.** The `static_assert` and the "GPU will not compile" message are gone.

**CI.** `.github/gpu_ci/interference_gpu.sh` and `interference_checks.py` add an opt-in `interference` job to `gpu_runner_ci.yml`. It is on by default for *Run workflow*, and off on pushes to main until it is validated. It checks:
1. standalone `u u~ > u u~` against the Fortran interference, the QED=0+2+4 = `QED^2<=4` sum rule, and FPTYPE=m;
2. the umami helicity choice on `u u~ > t t~ g QED^2==2`, with device memory through `cudaMalloc`/`hipMalloc` via ctypes;
3. mg7 `p p > u u~ QCD^2==2` on the GPU against −12253(10) pb, the `<init>` line and the signed events.

The script passes on CPU (`--backend simd_128`): the interference agrees to 3.7e−16, the sum rule to 3.2e−16, 11 of 16 points have mixed signs, and mg7 gives −12202(136) pb.

### Not planned: `[LIxtree]` in mg7

madmatrix has no loop matrix elements. Supporting it would need either a
umami shim around the Fortran MadLoop library, or loops in madmatrix. Both
are separate projects. Keep `LIxtree` madevent-only, refused for `mg7` as
#243 already does.

## 5. Decision needed

**The helicity convention (phase 1.1).**
- (a) The madevent convention: exact helicities, about 2× more events for interference samples, and identical for every other process. Recommended.
- (b) Keep ΣT as the weight and only select on `|T_i|`. Cheaper, but the helicity column stays statistically wrong whenever the signs are mixed.
- (c) (b) behind a run-card switch, with (a) as the default.
