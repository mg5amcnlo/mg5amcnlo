# Drell–Yan generation-only restart and reweighting checks

Tested the new provisional-pool workflow on `p p > e+ e- [QCD]` at 13 TeV,
with folding `(2,2,2)`, PYTHIA8 matching, internal `nn23nlo` PDFs, scale
reweighting, and `store_rwgt_info=True`. These are LHE events, without showering.
The old export `/tmp/mg5-amplicol-backend-sq5iorrd/drell_yan` was copied to
`/tmp/mg5-ampli-pools-dy-_vkgdhs1/drell_yan`; its original files were unchanged.
The copy received the current `simple_integrator.f90`,
`ampli_mint_adapter.f90`, `driver_mintMC.f`, run interface, and `ampli_pool.py`.
Exact tested source hashes are in `source_hashes.json`.

Both runs used `--only_generation` from the saved folded survey. They completed
production, finalization, built-in scale reweighting, and collection.

| Check | `pool_restart_unity` | `pool_restart_sum` |
|---|---:|---:|
| Requested / generated events | 40 / 40 | 20 / 20 |
| Seed | 610271 | 610272 |
| `nevt_job` | 12 | 2500 |
| Production signed rate [pb] | 1839.1406767 ± 8.5083737 | 1847.5904 ± 8.3318715 |
| Event weight magnitude range | 0.99614229–1 | 96.40661–97.460776 |
| Sum of absolute event weights | 39.95045197 | 1946.722657 |
| LHE `IDWTUP` | -4 | -4 |
| Events retaining `rwgt` / `mgrwgt` | 40 / 40 | 20 / 20 |
| Finite scale variation weights | 1080 / 1080 | 540 / 540 |

The unity run split eight parent channels into 672 small fixed-budget workers.
The sum run used eight workers. All selected events happened to have positive
signs in these small samples. Signed negative-weight coverage comes from the
larger ttbar benchmark and unit tests, not these small samples.

Each final `<init>` signed rate agrees with its production manifest. Native
residual corrections remain in `XWGTUP`; `unity` denotes a nominal scale and
does not force all final magnitudes to one. The compact worker `<header>` event
count is rewritten to the final parent-channel count before Fortran reweighting,
which reads and processes exactly that many events. The exact event counts and
complete scale weights verify this end-to-end boundary.

`validation.json` contains the full production manifests, selection diagnostics,
rate moments, batch budgets, event checks and summaries. The two small compressed
LHE samples and run banners are retained for inspection. PDF uncertainty
reweighting was not exercised because this configuration uses MG5's internal
central PDF implementation.
