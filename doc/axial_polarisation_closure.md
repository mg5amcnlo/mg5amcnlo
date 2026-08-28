# The axial polarisation and the density-matrix closure, measured with tau decays

Covered by `test_standalone_density_axial_tau` and
`test_standalone_density_axial_massless_daughters` in
`tests/acceptance_tests/test_cmd.py`, next to the existing
`test_standalone_density_uu` / `_dd` closure tests.

## What is being tested

`test_standalone_density_uu` checks that the full matrix element of
`u u~ > z z, z > e+ e-` is reproduced by contracting the production density
matrix with the two decay density matrices. It runs over the **three physical
polarisations** of each `Z` and it closes.

The question here is whether three is always enough. In the tetrad

    eps_-1, eps_0, eps_+1,   eps_A^mu = k^mu/sqrt(k.k)

the unitary-gauge numerator of a massive vector propagator decomposes as

    -g^{mu nu} + k^mu k^nu / M^2
       = sum_{i=-1,0,1} eps_i^mu eps_i^{*nu}  +  c_A  eps_A^mu eps_A^nu ,
      c_A = (Q^2 - M^2)/M^2 ,        Q^2 = k.k

so the factorisation into production times decay runs over **four** directions,
not three, and the fourth carries a virtuality-dependent coefficient that
vanishes on shell. The three-state sum is exact only when the fourth term is
zero, and that term is a product of two amplitudes:

* decay side. `k_mu M_dec^mu = -2 m_f g_A ubar(p1) gamma5 v(p2)`, proportional
  to the **mass of the decay fermions**. Zero for `e+ e-`, non-zero for
  `ta+ ta-`. This is the reason for taus.
* production side. `k_mu M_prod^mu`, which is zero by current conservation
  whenever the `Z` is emitted from a **massless** fermion line.

Both factors must be non-zero. That second condition is not optional and it is
what makes the existing `u u~ > z z` / `d d~ > z z` tests unable to see the
effect at all: they close to 1e-14 with taus, because the production axial
column of a massless-quark amplitude is identically zero (measured below at
1e-15 of the largest entry). The process used for the new tests is therefore
`t t~ > z z`, where the `Z` comes off a massive quark line.

## The measurement

Everything is standalone Fortran at a fixed phase-space point -- no Monte Carlo,
no MadSpin run. Three directories:

    generate t t~ > z{T0A}* z{T0A}* / a
    output standalone <prod> --density=3,4 --allow_axial

    generate z{T0A}* > ta+ ta-
    output standalone <dec>  --density=1   --allow_axial

    generate t t~ > z z / a, z > ta+ ta-
    output standalone <full>

`{T0A}` is the three physical states plus the axial one; the `*` makes every
state, including `eps_A`, be built with the leg's own virtuality rather than the
pole mass. Momenta go in through `PS.input` at full double precision (the
`edit_p_in_standalone()` helper used by the older tests writes `%e`, seven
significant digits, which is not enough for a closure meant to hold at 1e-13).

The reconstruction is

    ME_direct  =?  16 * sum_{a b c d} c_a c_b c_c c_d
                          rho_prod[(a,c),(b,d)] rho_dec1[a,b] rho_dec2[c,d]
                     / [ (Q1^2-M^2)^2 + M^2 G^2 ] / [ (Q2^2-M^2)^2 + M^2 G^2 ]

with `c = 1` on the three physical directions and `c = (Q^2-M^2)/M^2` on the
axial one -- or `c = 0` on it, which is the three-polarisation reconstruction.
The 16 is `IDEN` of the two braced decay directories (4 each), which `GET_INTER`
has already divided out.

Model masses as run: **`MTA = 1.777` GeV, `MT = 173` GeV, `MZ = 91.188` GeV**,
read back from the generated `Cards/param_card.dat` and asserted by the test --
a model with `m_tau = 0` would make the whole thing vacuous.

## Results

`t t~ > (z > ta+ ta-)(z > ta+ ta-)`, `sqrt(s)` and both virtualities in GeV:

| sqrt(s) | Q1 | Q2 | ME direct | 3 polarisations | 4 polarisations | rel. 3 | rel. 4 |
|--------:|---:|---:|----------:|----------------:|----------------:|-------:|-------:|
| 600  | 120.000 | 80.000  | 1.34421005e-11 | 1.34299914e-11 | 1.34421005e-11 | 9.01e-04 | 1.4e-14 |
| 600  | 60.000  | 150.000 | 2.92673524e-13 | 2.91558737e-13 | 2.92673524e-13 | 3.81e-03 | 8.3e-15 |
| 800  | 200.000 | 91.188  | 2.21310985e-10 | 2.20494145e-10 | 2.21310985e-10 | 3.69e-03 | 7.3e-15 |
| 500  | 91.188  | 91.188  | 2.89322359e-07 | 2.89322359e-07 | 2.89322359e-07 | 6.7e-16  | 6.7e-16 |
| 1000 | 300.000 | 45.000  | 3.39594864e-13 | 3.36333252e-13 | 3.39594864e-13 | 9.60e-03 | 1.1e-14 |

**Three polarisations fail at the 1e-3 to 1e-2 level; four close at 1e-14.**

The fourth row is both `Z` exactly on shell. There `c_A = (Q^2-M^2)/M^2 = 0` and
three close as well as four -- the axial direction is a piece of the propagator,
not a state the resonance can be produced in, so it must and does drop out on
shell. The failure grows with off-shellness, as `c_A^2` requires.

Same production, same phase-space points, `e+ e-` instead of `ta+ ta-`:

| sqrt(s) | Q1 | Q2 | ME direct | rel. 3 | rel. 4 |
|--------:|---:|---:|----------:|-------:|-------:|
| 600  | 120.000 | 80.000  | 1.34673123e-11 | 2.0e-14 | 2.1e-14 |
| 600  | 60.000  | 150.000 | 2.92677035e-13 | 1.2e-14 | 1.2e-14 |
| 800  | 200.000 | 91.188  | 2.20897897e-10 | 4.2e-15 | 5.3e-15 |
| 500  | 91.188  | 91.188  | 2.90194999e-07 | 1.6e-15 | 1.6e-15 |
| 1000 | 300.000 | 45.000  | 3.38445305e-13 | 8.0e-15 | 6.0e-15 |

The axial column of the decay density matrix is `< 1e-31` of the trace, three
close, and adding the fourth changes nothing. The per-mille gap above is a
daughter-mass effect and nothing else.

### The production side has to be non-conserved too

Axial column of the production density matrix, as a fraction of its largest
entry, and the resulting three-polarisation failure with tau decays:

| production | quark mass | axial column | rel. 3 (worst point) | rel. 4 |
|------------|-----------:|-------------:|---------------------:|-------:|
| `u u~ > z z` | 0     | 1e-15 | 2.4e-14 | 2.4e-14 |
| `b b~ > z z` | 4.7   | 2e-3  | 5.2e-05 | 2.6e-14 |
| `t t~ > z z` | 173   | 1     | 9.6e-03 | 1.1e-14 |

`u u~ > z z` -- the process of the existing closure test -- closes with three
polarisations **even with taus**. That is the one part of the naive framing
("massive daughters, therefore three must fail") that is wrong: massive
daughters are necessary and not sufficient.

### Sign and magnitude of the axial term

Both are as the propagator decomposition predicts.

*Sign.* The missing piece is `c_A^2` times diagonal entries of two
positive-semidefinite density matrices, `rho_prod[(A,.),(A,.)] = |A|^2` and
`rho_dec[A,A] = |A|^2`. It is therefore manifestly **positive**, and the
three-polarisation reconstruction must always **undershoot**. It does, at every
point measured.

*Magnitude.* The decay-side axial entry is analytic:

    rho_dec[A,A] = sum_spins |eps_A . M_dec|^2 / IDEN
                 = 8 m_ta^2 g_A^2 / 4
                 = 2 m_ta^2 g_A^2 ,      g_A = e/(4 s_w c_w)

i.e. proportional to `m_tau^2` and **independent of the virtuality**. Measured
against `2 m_ta^2 g_A^2 = 0.21656039079792713` built from the run's own
`ee^2`, `sw^2`, `cw^2`:

    Q =  30 GeV   0.2165603907979273   ratio 1.000000000000
    Q =  60 GeV   0.2165603907979272   ratio 1.000000000000
    Q =  91.188   0.2165603907979271   ratio 1.000000000000
    Q = 120 GeV   0.2165603907979271   ratio 1.000000000000
    Q = 200 GeV   0.2165603907979272   ratio 1.000000000000
    Q = 400 GeV   0.2165603907979272   ratio 1.000000000000

to twelve digits, with the axial-to-physical off-diagonals at 1e-14. This fixes
the normalisation of the `{A}` external wavefunction, `eps_A = k/sqrt(k.k)`,
independently of the closure itself: had it been built with the pole mass the
entries would carry a spurious `Q^2/M_Z^2` and would not be flat in `Q`.

## Why this is `t t~ > Z Z` and not `g g > Z Z`

The natural home for this measurement is the loop-induced `g g > Z Z`, where the
top box supplies the non-conserved production current. It is not reachable:

1. MG5 refuses any polarisation brace on a massive particle in a loop-induced or
   NLO process (`madgraph_interface.py`, "Polarization restriction can not be
   used for massive particles"). This predates the `{A}` work -- it is on `3.x`
   -- and it makes `g g > z{T0A}* z{T0A}* [sqrvirt=QCD]` impossible to generate.
2. Bypassing the brace by widening `ALLOW_HEL` at the driver level does not work
   either. The loop-induced `GET_ALL_INTER` (`compute_color_flows.f`) resolves
   each helicity through `FIND_H` into a **precomputed** `JAMPL_ALL` table
   indexed over the `NCOMB` rows of the `NHEL` table. `nhel = 4` is not in that
   table, the lookup falls off the end, and every axial entry comes back `NaN`.
   The tree-level path does not have this problem: it overwrites `NHEL` and
   recomputes the amplitude, which is exactly why appending `4` to MadSpin's
   `hel_dict` is enough there.
3. `mp_vxxxxx` in `aloha/template_files/aloha_functions_loop.f` -- the
   quadruple-precision wavefunction MadLoop escalates to on unstable phase-space
   points -- has no `nhel = 4` branch, unlike `vxxxxx` beside it. It would fall
   through to the physical-polarisation code with `hel = 4`.

So the axial polarisation is a tree-level capability today. The physics
statement it establishes is not tree-specific -- what the closure needs is a
non-conserved current at the `Z` leg, and the top box of `g g > Z Z` supplies
one exactly as the massive top line of `t t~ > Z Z` does -- but the measurement
cannot presently be repeated there.
