# Auxiliary native-Born normalization audit

The positive LO control integrates to20681.3 ±8.99 pb. Deduplicating104
first-step worker records into the12 actual native GF channels gives a
Born-record sum20845.699 ±12.527 pb. Deduplication, schema interpretation and
linked-file provenance are verified by `audit_born_normalisation.py` and
`Born_normalisation_audit.json`.

The auxiliary integral is `int_born_xsec=28`, filled from actual weighted
`itype=2` Born records in `fill_mint_function_NLOPS`, accumulated by ordinary
MINT sums/squares. It is distinct from the virtual-sampling diagnostic f6.
All726385 S events have source Born contexts matching their registry's
primary process directory, so unrecognized mixed Born channels do not
explain this comparison.

The discrepancy is concentrated in P0_dxg_wpux:

- LO physical-channel fraction:2369.043 ±15.652 pb (conditional partition
  error from200k unweighted LO events).
- Native B:2539.139 ±4.398 pb, of which GF2.0 contributes2054.584 ±3.740.
- Reflected P0_gdx_wpux: LO2378.763 ±15.684, native2380.537 ±5.119;
  GF2.0 is1893.037 ±4.438.
- The other four physical Born channels agree within1.3 conditional sigma.

The dxg/gdx GF2 Born discrepancy persists across several MINT iterations,
so it is not solely a final-iteration outlier. However no deterministic
PDF-carrier, integrator, or source-code fault was established in this audit.
A Born-only integration of the same native GF integrands would be needed to
separate a measure/context effect from a stable auxiliary-integration bias.

The corresponding S integrals differ similarly:1879.689 versus1727.715 pb.
Their beta ratios are .9148758 versus .9126686, only0.242% apart. This
particular mirrored beta difference cannot explain the15–30% transfer shape
error by itself. Nor does this open finding weaken the independently
measured wrong-sign S-density interval near Born pT20 GeV.

Per-directory full NLO H+S rates are **not** an independent physical mirror
check after H-history owner repartition: their buckets need not coincide
with physical incoming-parton subprocesses. Their asymmetry is therefore
not used as evidence for an upstream physics bug.

This remains an explicit open validation finding. No source or production
result was changed or relabelled.
