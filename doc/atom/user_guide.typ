// User guide: bound atomic electrons in MG5_aMC (atom branch).
// Build: typst compile user_guide.typ user_guide.pdf

#let navy = rgb("#1f3b63")
#let pale = rgb("#eef3f9")
#let warm = rgb("#fdf3e7")
#let ink2 = luma(90)
#let hair = 0.4pt + luma(222)
#let sans = "Helvetica Neue"

#set document(title: "Bound atomic electrons in MadGraph5_aMC@NLO: user guide", author: "MG5_aMC atom branch", date: none)
#set page(paper: "a4", margin: (x: 1.8cm, top: 1.8cm, bottom: 1.7cm), numbering: "1 / 1",
  header: context if counter(page).get().first() > 1 {
    set text(8pt, fill: luma(110))
    [Bound atomic electrons in MG5_aMC — user guide #h(1fr) `atom` branch]
  })
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => { v(0.4em); block(text(13.5pt, fill: navy, it)); v(0.05em) }
#show heading.where(level: 2): it => block(above: 0.9em, below: 0.5em, text(11pt, fill: navy, it))
#show raw: set text(font: "DejaVu Sans Mono", size: 8.4pt)
#show raw.where(block: true): it => block(width: 100%, fill: rgb("#f6f6f4"), inset: (x: 8pt, y: 6pt), radius: 3pt, stroke: 0.4pt + luma(225), it)
#show link: set text(fill: rgb("#1f5fa8"))
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: set text(9pt)
#show figure.caption: set align(left)
#set figure(gap: 0.5em)

#let tstyle(body) = {
  set text(7.8pt, font: sans)
  set par(justify: false, leading: 0.4em)
  show table.cell.where(y: 0): set text(fill: white, weight: "bold")
  body
}
#let tstroke(x, y) = (top: if y <= 1 { none } else { hair })
#let thead-fill(x, y) = if y == 0 { navy } else { none }
#let box-note(fill: pale, body) = block(width: 100%, fill: fill, inset: 9pt, radius: 3pt, text(9.5pt, body))

// ================================================================= TITLE ==
#align(left)[
  #text(18pt, fill: navy, weight: "bold")[Bound atomic electrons in MadGraph5_aMC\@NLO]
  #v(-0.6em)
  #text(11.5pt, fill: ink2)[User guide for the `atom` branch: leading-order cross sections and events for a beam scattering on an electron bound in an atom]
]
#v(0.4em)

#box-note[
  *In one paragraph.* A positron, electron, muon or photon beam hits a target atom at rest; the hard process takes place on _one_ of the atom's electrons, which is bound and moves with momentum *p*#sub[B] distributed according to relativistic (Dirac–Hartree–Fock) wavefunctions. With the `atom` branch of MG5_aMC you write such a process like any other, e.g. `generate e+ eC12 > mu+ mu-`, and MadEvent returns the leading-order cross section _per atom_ (summed over all electrons) and unweighted events in the Les Houches format. The electron motion is what makes, for example, μ⁺μ⁻ production possible below the free-electron threshold, and what smears resonances and kinematic edges.
]

= What is computed

The cross section is that of the impulse approximation in the rest frame of the atom (the lab frame):

$ sigma = 1/(4 m_e |arrow(p)_1|) integral (d^3 arrow(p)_B)/((2 pi)^3) d Phi_n (p_1 + p_B; p_f) sum |cal(M)|^2, quad p_B = (m_e, arrow(p)_B), $

where p₁ is the beam momentum, the sum runs over all occupied shells of the atom with their occupation numbers, and in |$cal(M)$|² the spinor of the incoming electron is replaced by the bound Dirac wavefunction in momentum space. The bound electron carries the energy m#sub[e] and an arbitrary three-momentum, so it is off its mass shell; binding energies are neglected. Phase-space points are drawn according to the electron momentum density of the atom, so that also rare high-momentum electrons are sampled efficiently. Everything is leading order.

= Installation

You need Python ≥ 3.7, a Fortran compiler (gfortran ≥ 4.6) and `make`, as for any MG5_aMC installation; nothing else is required. Get the `atom` branch:

```bash
git clone -b atom https://github.com/mg5amcnlo/mg5amcnlo.git MG5_atom
cd MG5_atom
./bin/mg5_aMC
```

The last line starts the MG5_aMC prompt (`exit` leaves it). All examples below are scripts: save the lines in a file, e.g. `example.mg5`, and run `./bin/mg5_aMC example.mg5` from the `MG5_atom` directory. The output directories are created in the directory from which you run.

#box-note(fill: warm)[
  *Tip.* By default MG5_aMC opens result pages in a web browser. The first line of the scripts below, `set automatic_html_opening False`, switches this off (the choice is remembered).
]

= Available atoms

The bound electron is written as `e` followed by the isotope. The model is `sm_atom` (the Standard Model with the bound electrons added); use it with lepton masses, `import model sm_atom-lepton_masses`, since the bound electron needs the electron mass.

#figure(tstyle(table(columns: (auto, auto, auto, auto, auto, auto), stroke: tstroke, fill: thead-fill, inset: (x: 4pt, y: 3pt), align: (left, left, right, left, right, right),
  table.header[Label][Atom][Z][Shells][Bound-electron code][Ion code (LHE)],
  [`eC12`], [carbon-12], [6], [4: 1s 2s 2p], [1000601211], [1000060120],
  [`eW186`], [tungsten-186], [74], [22: 1s … 6s], [1007418611], [1000741860],
  [`ePb208`], [lead-208], [82], [24: 1s … 6p], [1008220811], [1000822080],
  [`eU238`], [uranium-238], [92], [29: 1s … 7s, 5f, 6d], [1009223811], [1000922380],
)), caption: [Atoms with wavefunction tables. The bound-electron code is 10ZZZAAA11, the ion code of the atom in the event file 100ZZZAAA0. Open shells (e.g. the 2p of carbon, the 5f and 6d of uranium) use the mean occupation of the average configuration.]) <tab-atoms>

Each label stands for all electrons of the atom: the shells are summed automatically and the cross section is per atom. For uranium the momentum tables end at |*p*#sub[B]| = 1.69 MeV and are continued by the asymptotic power law above; results that depend mainly on electrons faster than about 2 MeV (deep below a production threshold) carry an uncertainty of tens of per cent for uranium.

= Quick start

== Muon pairs from positrons on carbon

```
set automatic_html_opening False
import model sm_atom-lepton_masses
generate e+ eC12 > mu+ mu-
output madevent epeC12_mumu
launch epeC12_mumu
set ebeam1 100
set nevents 10000
```

- `generate e+ eC12 > mu+ mu-`: the positron is beam 1, the carbon atom beam 2.
- `output madevent epeC12_mumu` writes the process directory `epeC12_mumu`; `launch` runs it.
- `set ebeam1 100`: positron energy 100 GeV. The atom is at rest, so the energy of beam 2 is not used. No PDFs, no default cuts and fixed scales are set automatically for atom processes.

The run takes well under a minute on a laptop, most of it compilation. The result is printed at the end, σ = 4.989×10⁶ ± 1.0×10³ pb per atom, and the events are written to `epeC12_mumu/Events/run_01/unweighted_events.lhe.gz`. For comparison, six free electrons at rest would give 4.987×10⁶ pb: at this energy the electron motion only changes the total at the 10#super[−3] level, but it spreads √s around its free value by a few per cent.

== Muon–electron scattering on carbon

```
set automatic_html_opening False
import model sm_atom-lepton_masses
generate mu+ eC12 > mu+ e-
output madevent mupeC12_mupem
launch mupeC12_mupem
set ebeam1 150
set el 1.0
set nevents 10000
```

Photon exchange in the t-channel makes the cross section diverge for soft electron recoils, so a cut is needed: `set el 1.0` keeps events with an outgoing electron energy above 1 GeV (in the lab, i.e. the atom rest frame). Result: σ = 1.573×10⁹ ± 1.9×10⁶ pb per atom (six free electrons at rest: 1.571×10⁹ pb).

== Matrix elements only

`output standalone` writes Fortran code for |$cal(M)$|² with a bound electron; `make check` builds a small test program that evaluates it at a random phase-space point in the atom rest frame:

```
set automatic_html_opening False
import model sm_atom-lepton_masses
generate e+ ePb208 > mu+ mu-
output standalone epePb208_mumu_SA
```
```bash
cd epePb208_mumu_SA/SubProcesses/P1_*/
make check
./check
```

The program prints the momenta (the bound electron is particle 2, with energy m#sub[e]) and the matrix element; with the bound wavefunction √(2m#sub[e]) $cal(U)$ in place of a free spinor it has dimension GeV#super[−3] for a 2 → 2 process.

== Interactive use

The same commands can be typed at the `./bin/mg5_aMC` prompt. After `launch`, MG5_aMC first asks which programs to run (type `0` or press Enter for none), then offers to edit the cards: type the `set` commands, then `0` to start the run.

= Writing processes

- *One bound electron per process*, as either incoming particle. `e+ eC12 > …` puts the atom on beam 2 (the beam particle moves along +z); `eC12 e+ > …` puts it on beam 1, and then the energy is set with `set ebeam2`.
- *Any final state* of the model, e.g. `e+ eC12 > a a`, `e+ eU238 > mu+ mu-`, `a eC12 > a e-` (Compton scattering), and also *2 → 1* resonances such as `e+ eC12 > z` (the Z mass and width can be changed in the param card to model a light resonance).
- *Outputs*: `madevent` (cross sections and events) and `standalone` (matrix elements). NLO (`[QCD]`, `[QED]`) and the aMC\@NLO output are not available with bound electrons.

= Run-card settings

Atom processes get a dedicated block in `Cards/run_card.dat`:

#figure(tstyle(table(columns: (auto, auto, 1fr), stroke: tstroke, fill: thead-fill, inset: (x: 4pt, y: 3pt), align: (left, left, left),
  table.header[Parameter][Default][Meaning],
  [`eatom_shell`], [0], [0: all shells. Otherwise only one shell, coded sign(κ)(100n + |κ|): −101 = 1s, −201 = 2s, 201 = 2p#sub[1/2], −202 = 2p#sub[3/2], 302 = 3d#sub[3/2], −303 = 3d#sub[5/2], 403 = 4f#sub[5/2], −404 = 4f#sub[7/2].],
  [`eatom_pmax`], [−1], [Largest |*p*#sub[B]| of the bound electron in GeV (−1: no limit beyond kinematics). Use it to cut away very fast electrons, e.g. `set eatom_pmax 0.1`.],
  [`eatom_ps_mode`], [1], [Phase-space sampler of the electron momentum; keep 1 (the other values are for tests).],
  [`eatom_iatom`, `eatom_a`, `eatom_z`], [from the process], [Beam carrying the atom, mass number, atomic number. Set by MG5_aMC; do not edit.],
)), caption: [The atom block of the run card. Change values with `set <name> <value>` after `launch`.])

*Set automatically and kept*: no PDFs and no beam remnants (`lpp1 = lpp2 = 0`), the atom at rest, no default cuts, fixed renormalisation and factorisation scales, and an explicit sum over all shells and spins (`nhel = 0`).

*Cuts* act on the final state in the lab frame, which is the atom rest frame. Processes with a massless particle exchanged in the t-channel need one:
- μ e → μ e, e e → e e: a minimum energy of the outgoing electron, `set el …` (GeV).
- Compton scattering `a eC12 > a e-`: a minimum energy of the outgoing photon, e.g. `set ea 1e-4` (0.1 MeV). Without it the cross section diverges because the off-shell bound electron can absorb the photon.

*Below threshold.* A beam energy below the free-electron threshold is allowed: e.g. `e+ eC12 > mu+ mu-` at 40 GeV (the threshold for an electron at rest is 43.7 GeV) still has a cross section, from the fast electrons of the atom. The sampler handles this without further settings.

= Output

*Cross section.* Per atom, in pb, summed over all electrons. For a thin target of thickness ℓ and density ρ, the number of events is N = N#sub[beam] × (ρ ℓ N#sub[A]/A) × σ.

*Events.* The event file is written in the atom rest frame. The bound electron appears as an incoming particle with PDG id 11, status −1, momentum (m#sub[e], *p*#sub[B]) and helicity 9 (summed over shells and spins); it is off-shell and space-like, and the mass column holds sign(p²)√|p²|. In the `<init>` block the atom is the beam, with its ion code (@tab-atoms) and energy A × 0.9315 GeV; the other beam has the energy you set. The momenta conserve energy and momentum with the bound electron included.

= Physics content and limitations

- Impulse approximation: the hard process happens on one electron; the rest of the atom is a spectator. Binding energies are neglected (the bound electron has energy m#sub[e]); interactions of the final state with the atom (Coulomb distortion, screening) are not included.
- The matrix element is defined in the atom rest frame only; the code evaluates it there in all cases.
- Leading order; no parton shower or QED radiation for the bound electron; one atom per process.
- Open-shell atoms use the mean shell occupations of the average configuration.
- Compton scattering on bound electrons needs the photon-energy cut above. The bound-state treatment is not explicitly gauge invariant; effects are expected at the level of the electron momenta over the hard scale.

= Troubleshooting

- *`ERROR: Do not known (A,Z)`*: the atom has no wavefunction tables (only those of @tab-atoms exist).
- *Very large or unstable cross section* for a process with photon exchange: a cut on the outgoing electron or photon energy is missing.
- *`ERROR: a bound atomic electron requires lpp1 = lpp2 = 0`*: leave `lpp1` and `lpp2` at 0 (no PDFs, no ion beam; the atom is described by its wavefunctions). A non-zero `nhel` is set back to 0 with a warning.
- *Matrix-element program stops with a frame error*: the bound electron must have energy m#sub[e] (atom rest frame); in `standalone` code use momenta in that frame.

= References

- Wavefunctions: Dirac–Hartree–Fock tables computed with DBSR-HF, O. Zatsarinny and C. Froese Fischer, Comput. Phys. Commun. 202 (2016) 287, provided by L. Darmé.
- MG5_aMC: J. Alwall et al., JHEP 07 (2014) 079, #link("https://arxiv.org/abs/1405.0301")[arXiv:1405.0301].
- Applications of positron annihilation on atomic electrons: F. Arias-Aragón, L. Darmé, G. Grilli di Cortona, E. Nardi, PRL 132 (2024) 261801, #link("https://arxiv.org/abs/2403.15387")[arXiv:2403.15387] and #link("https://arxiv.org/abs/2407.15941")[arXiv:2407.15941].
