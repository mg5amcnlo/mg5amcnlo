# `pure_interference` under `PA` and `madspin`

PR #363 shipped two output shapes for the pure-interference mode --
`pure_interference_output = weighted` (the default) and `= unweighted` -- and
validated both end to end only under `spinmode = onshell`.  This directory
validates them under `spinmode = PA` and `spinmode = madspin`, and tests the
one code path that only `PA` can reach: a reshuffling that fails and hands back
`W <= 0`.

Read **`RESULTS.md`**.  The short version:

* both output shapes reproduce the committed closure numbers under both
  spinmodes -- `<C_nn>` pulls between -0.59 and +1.15, `<C_rr>` between +0.31
  and +1.66, on 50 000 production events each;
* the `unweighted` output carries exactly two signed weight values and one
  magnitude in all three spinmodes;
* `c` is 0.955 of its analytic candidate under `PA` and `madspin` against
  0.997 under `onshell`, i.e. the `<jac> != 1` the plan predicted, measured;
* the failed-reshuffle path **never fires** in 376 000 reshuffles, so it was
  forced.  Forcing it shows that a `W = 0` is handled correctly by both output
  shapes, that a `W < 0` from a failed reshuffle would be indistinguishable
  from physics and invisible to every monitor the mode has, and that only a
  retry policy in `lhe_parser` keeps that from happening.  Recorded as a
  fragility, not a bug;
* `fixed_order` has two defects that are present with `pure_interference`
  switched off, and are therefore not #363's.

Everything here is driven from the repository root; nothing modifies shipped
source.  `drive_madspin.py` is `MadSpin/madspin` with the instrumentation
monkey-patched on at start-up.

    bash MadSpin/validation/interference_pa_madspin/run_pa_madspin.sh <work> 50000
    bash MadSpin/validation/interference_pa_madspin/run_badjac.sh     <work> 2000 0.10
    bash MadSpin/validation/interference_pa_madspin/run_extras.sh     <work> 2000
    python3 MadSpin/validation/interference_pa_madspin/analyse_pa_madspin.py <work> <out>
    python3 MadSpin/validation/interference_pa_madspin/analyse_badjac.py     <work> <out>
    python3 MadSpin/validation/interference_pa_madspin/analyse_extras.py     <work> <out>

`analyse_pa_madspin.py`'s LHE reader, spin basis and observables are copied
verbatim from `MadSpin/validation/interference_closure_v2/analyse_interference.py`
(branch `claude/ms-interference-closure-v2`) so the numbers are directly
comparable to that test's.  They are copied rather than imported only because
the two validation directories sit on sibling branches.
