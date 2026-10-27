MAcNLOPS method and implementation paper — revised 25 October 2026

The main exposition now describes the committed sampled native-component
implementation (baseline dd9475234), including subsequent cleanup and the
numerical fixes documented below. It separates
ordinary matching with the source bogus P, sampled Delta matching with
an undamped full-G numerator and a Delta-weighted H residual, and the
global-source compatibility path. The FKS settings, diagram/flow transport,
identical-particle measures, first-step metadata, protocol, preserved
weights and shower-continuation scales are documented explicitly.

Every million-event rate table and all 696 campaign panels still describe
the completed 23–24 October global-source-v1 campaign. Its Delta veto
included source damping. These are not distributions of the new sampled
veto. Captions and atlas overlays state this distinction. The seven
20 October patched-runtime panels retain their separate historical labels.
No source events, showers or histograms were regenerated for this revision.

Outputs:
  macnlops_paper.pdf    Method, implementation, tables and 15 comparisons.
  macnlops_complete.pdf Same narrative followed by the complete figure atlas.
  macnlops_figures.pdf  Standalone bookmarked atlas at original panel sizes.
  macnlops_latex_source.zip Portable source bundle with every figure asset.

Build with a normal TeX Live installation:
  make paper
  make complete
  make atlas

main.tex is the complete edition; paper.tex selects the narrative edition.
All figures, tables and bibliography entries are included. Rebuilding
requires neither the event samples nor MadGraph/Pythia. The optional
prepare_paper.py refresh reads the completed local campaign, regenerates
the frozen data/figure copies and captures the checkout used at that time;
it is not part of a normal rebuild and does not produce new physics data.

Provenance:
  paper_data.json and figure_manifest.json freeze the numerical tables and
  all 703 figure hashes. evidence_manifest.json and implementation_changes.patch
  retain the original campaign record. post_campaign_review.json and
  sampled_components_review.json retain the earlier implementation checks.
  current_implementation_manifest.json records the code checked for this
  document revision and its verification. These records have distinct
  scopes; none relabels the global campaign as a sampled-mode production.

The paper distinguishes algebraic transfer identities, observed finite
results and accuracy conditions. It does not assert exact finite-shower
cancellation, universal positive weights, anonymous-LHE import, or a fully
nf4 stock shower coupling. The small sampled-source and native-point
checks test implementation paths rather than new precision distributions.
CHECKS.txt and document_checks.json record the document verification.

Matching fixes checked on 25 October:
  The signed-residual rule now uses max(K,0) as transferable support, so
  negative PDF luminosities retain S and any required H residual. Born
  fraction checks use the native Born snapshot when massive-FSR second
  solutions have no counterevent. Single-particle recoil invariants use
  the on-shell model mass. Zero FKS partitions create no unit-Delta replay
  row; annotated hard-input preflight allows accumulated boost roundoff.
  numerical_fix_review.json records all 133 saved density failures passing
  with temporary current objects and a numerical-Delta abort trap.
  The two additional W+jet hard preflight points pass standalone C++ checks.
  These bounded checks do not certify completed rematched million-event
  samples. No source events, shower histories or histograms were produced.
  numerical_implementation_changes.patch records the runtime fix separately
  from the original campaign patch. The frozen figure/table assets remain
  unchanged; updated paper editions still have 62/766/704 pages.
