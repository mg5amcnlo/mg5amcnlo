MAcNLOPS from ordinary MC@NLO events
===================================

This is a scientific-paper-format implementation study of the completed
common_200k_v2 PATCHED calculation, dated 20 October 2026. Authorship is
intentionally unset. The seven vector panels are unchanged copies of those
frozen results. The runtime-status text was updated on 21 October; no events
or plots were regenerated.

Current runtime versus the frozen study
--------------------------------------
Production now uses a multistep workflow: the first-emission driver and
Python removal/veto write the accepted events.lhe.gz, which is then consumed
by the ordinary Pythia shower. The dedicated uninterrupted full-shower
driver and its generation/audit tools have been removed. The frozen full
panels below describe that retired workflow; they have not been regenerated
with the current LHE continuation.

The active code now uses Pythia without the MAcNLOPS trial-headroom,
certified-PDF-envelope/provider and nf4 coupling patches. It has no custom
trial-headroom argument or MAcNLOPS patch-builder prerequisite. The unrelated
existing global QED patch is outside this removal. The incoming/splitting
flavour counts remain four or five, but the effective stock shower coupling
has alpha_s_nfmax=5 in both cases. Native trial-bound warnings are recorded
without rejecting, dropping or retrying source events; no universal native
envelope certification is claimed.

The figures, tables, setup_details.tex, paper_data.json and original evidence
manifest describe the frozen PATCHED configuration, including its genuine
nf4 shower-coupling extension. They are not evidence of stock-runtime results
or of nf4 coupling evolution in current stock Pythia. The removed extension
sources/builds are identified by the external frozen production archives.
Rebuilding this paper does not build Pythia or run the simulation. A new
stock calculation needs a freshly measured source/runtime profile and cannot
reuse patched checkpoints as stock qualifications.

Build the paper (portable; no event files or Python environment needed):

  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

or run `make`. The result is build/main.pdf. A standard TeX Live installation
with pdflatex, BibTeX, natbib, subcaption and the packages named in main.tex
is sufficient. Both bibliography files and all included tables are supplied.

Contents
--------
main.tex                  Main article
setup_details.tex         Checked generation/shower setup
observable_details.tex    Actual analysis definitions
rate_table.tex            Extracted inclusive signed accounting
residual_table.tex        Required and realized negative-H residuals
references.bib            Primary generator/subtraction/PDF/jet references
method_references.bib     Primary matching-method references
figures/                  Seven portable vector PDFs and their source hashes
paper_data.json           Exact accounting rows and checked setup metadata
evidence_manifest.json    Original source and supporting-record checksums
make_tables.py            Optional extraction from existing production records
CHECKS.txt                Scope of manuscript verification and build checks

Re-extracting the tables is optional and requires the large original record
tree. The source package already contains the extracted tables and numbers:

  python3 make_tables.py --production-root /path/to/macnlops_common_20261020

The default root is /export/tmp/rikkert/macnlops_common_20261020. Event files,
the 72-page full plot collection and production executables are not copied
into this small manuscript package. figure_manifest.json and
evidence_manifest.json locate and pin the relevant original artifacts.

Interpretation
--------------
In the frozen study, reference and treatment share one ordinary source and one native evolution
per original event. Pre-veto equality is an identity of this common cohort.
The final prescription keeps the signed H residual required by its chosen
denominator/support and preserves original negative S weights. It does not
assert universally positive weights, exact finite-shower compensation or an
independently proven NLO/logarithmic accuracy for arbitrary processes.

Remaining whole-production particle and final independent histogram/package
audits were skipped at the user's request. The paper states that limitation.
Manuscript checks did not restart them or launch new simulation jobs.
