# Correcting `A_e`, the mass stage's per-event normalisation

`doc/madspin_pa_mass_stage/bound_design.md` section 5 established that MadSpin's
sequential mass stage draws virtualities from a proposal `q_e`, weights them by
`w`, and redraws until it accepts -- writing **exactly one** event whatever

    A_e = E_{q_e}[w]

is.  `A_e` is therefore divided out, and it is not a constant.  That assessment
declined to recommend implementing the fix on its own authority, because doing
it changes what MadSpin's output *is*.

This directory is the measurement that decision needs: what the correction
costs, and what it changes, **for every affected mode**.  No recommendation is
made here either beyond the last section.

## Files

| path | what |
|---|---|
| `ae_kernel.py` | the integral, three evaluators for `J`, and the cost of each |
| `measure_ae.py` | `A_e` and `E_q[w^2]` over a whole production sample, exactly, per mode |
| `ratio_R.py` | the offshell `Tr(rho_off)/\|M\|^2_on` factor, sliced in `sqrt(shat)` |
| `analyse_impact.py` | what correcting it changes: spectrum, lineshapes, cross section, weights |
| `assessment.md` | the numbers and what they say |
| `data/` | raw measurement, and each run's `madspin.log` |

The prototype itself is in `MadSpin/interface_madspin.py` under the option
`mass_normalisation`, which is **off by default and is not a supported option**.
See `assessment.md` section 6 for what is prototype scaffolding and what is not.

## Setup

`p p > t t~` at 6.5+6.5 TeV, LO, 200 000 unweighted production events, both tops
fully leptonic, `BW_cut = 15`, `unweighting = sequential`, `nb_core = 8`, seed 42
-- the same process, statistics and observable as
`MadSpin/validation/mt_lineshape/`, so the lineshape numbers here are directly
comparable with that campaign's **measured** replica noise floor.

## Reproducing

```bash
export PATH="$HOME/.pyenv/versions/mg-3.14/bin:$PATH"

# one baseline run per mode, with ms_dir set so the Zhat tables land on disk
python3 MadSpin/madspin card_pa.dat        # spinmode PA
python3 MadSpin/madspin card_panojac.dat   # spinmode PA, density_keep_jacobian False
python3 MadSpin/madspin card_madspin.dat   # spinmode madspin

python3 doc/madspin_ae_normalisation/ae_kernel.py \
    --events <production .lhe.gz> --json data/cost.json --check
for m in pa panojac madspin; do
  python3 doc/madspin_ae_normalisation/measure_ae.py \
      --events <production .lhe.gz> --mode $m \
      --ztables <ms_dir_$m>/max_wgt_sequential_* --out data/ae_$m.npz
done

# the offshell R factor: one more madspin run with the estimator on
python3 MadSpin/madspin card_madspin_ae.dat   # + mass_normalisation_draws 24
python3 doc/madspin_ae_normalisation/ratio_R.py \
    --baseline <b_madspin>/events_decayed.lhe.gz \
    --corrected <p_madspin>/events_decayed.lhe.gz \
    --ae data/ae_madspin_R1.npz --ref 0.956832 --out data/ratio_R.json

for m in pa panojac madspin; do
  python3 doc/madspin_ae_normalisation/analyse_impact.py \
      --ae data/ae_$m.npz --lhe <b_$m>/events_decayed.lhe.gz \
      --label $m --out data/impact_$m.json
done
```

The `.npz` files are stored as float32 (7 digits, against a worst-case
quadrature error of 1.2e-4); the production and decayed LHE files are not
committed.
