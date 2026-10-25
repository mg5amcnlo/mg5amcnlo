MAcNLOPS implementation paper — completed 23–24 October 2026 campaign

25 October revision: the Delta section explicitly compares the original
undamped MAcNLOPS veto with the source-damped prescription used in the
campaign. The implementation section records subsequent checksum/routing
optimizations. Figures and numerical results still describe the completed
campaign. The original evidence manifest and implementation patch remain
its frozen provenance; post_campaign_review.json records the later review.

Further 25 October revision: qualified sampled native components now
support KL=False, fixed flow=True and either outer FKS setting. The paper
documents the full-G, Delta-free local veto, matching H residual,
diagram/flow transport and focused checks. Those checks include tiny
fresh ordinary/Delta source runs; they do not update any campaign figure
or validate new large-sample distributions. sampled_components_review.json
records this subsequent implementation separately from the first review.

Outputs:
  macnlops_paper.pdf    Narrative paper, tables and 15 selected comparisons.
  macnlops_complete.pdf Same paper followed by every figure in the atlas.
  macnlops_figures.pdf  Standalone bookmarked atlas, at original figure sizes.

The atlas contains all 696 current validation PDFs, including ordinary and
Delta sources, first/full showers, all observables, ratio zoom companions
and all separate mechanism panels. Seven older figure PDFs from the
previous paper are included separately and visibly marked historical.
The overview books are selections of these same panels, not extra data.

Build with a normal TeX Live installation:
  make paper
  make complete
  make atlas

The main source is main.tex; paper.tex selects the narrative-only edition.
All figures, tables and bibliography entries are included. Rebuilding does
not require the original event samples, MadGraph or Pythia. The optional
prepare_paper.py script refreshes the frozen inputs from the original
completed local campaign; it is not needed for a portable LaTeX rebuild.

paper_data.json freezes the numerical evidence used by the tables.
figure_manifest.json records every included PDF with its source and hash.
evidence_manifest.json records the base Git revision, relevant modified
source hashes, and all 20 completed dataset/level checks.
implementation_changes.patch preserves the tracked implementation changes.
CHECKS.txt describes the document and factual checks performed.

This paper preserves the distinction between algebraic transfer identities,
the actually measured finite-shower results, and conditions not proved by
the existing histograms. It does not claim universal positive weights,
exact finite cancellation, generic anonymous-LHE support, or a fully nf4
stock shower coupling in Wbb. The tiny sampled-mode implementation checks
are distinguished from the frozen campaign evidence. The earlier frozen
20 October paper is preserved separately.
