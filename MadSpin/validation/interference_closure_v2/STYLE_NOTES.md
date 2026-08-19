# Style conventions extracted from the user's `plot_matplotlib.py`

Reference: the user's personal script (854 lines, not part of this repository
and not imported by anything here).  The conventions below were read off
`plot_hist_with_ratio` (l.201), `plot_hist_with_ratio_multi` (l.271),
`plot_hist_with_ratio_weighted` (l.342) and `plot_wb_mass` (l.416), and are what
`plot_interference_userstyle.py` implements.

This file exists so the inverse task -- rendering the user's own plots in the
MG7 paper style -- can be checked against the same list.

## Global

| convention | value |
|---|---|
| rcParams | **none set at all** -- stock matplotlib (DejaVu Sans, `font.size` 10, default mathtext) |
| backend | `matplotlib.use('Agg')` at import |
| no `usetex` | math is written as plain mathtext, `r'$m_{Wb}$ [GeV]'` |
| colour cycle | stock `C0`..`C3`; sample dicts carry an explicit `"color"` key |
| accent colours | `#74c476` / `#238b45` for the uncertainty bands in `plot_wb_mass` |

Note the contrast with `plot_interference.py`, which sets
`text.usetex`, `font.family: serif`, `font.size: 14`, `lines.markersize: 8`
and a recoloured Tableau cycle.  The user's style is the plain sans-serif one.

## Figure and panel geometry

| convention | value |
|---|---|
| figsize | `(6, 6)` in `plot_hist_with_ratio*`; `(7, 7)` in `plot_wb_mass` |
| gridspec | `fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=...)` |
| `hspace` at gridspec time | `0.05` (`0.0` in `plot_wb_mass`) |
| ratio axis | `fig.add_subplot(gs[1], sharex=ax_main)` |
| final layout | `fig.subplots_adjust(hspace=0.1, left=0.15, right=0.97, bottom=0.12, top=0.96)` |
| upper panel x tick labels | suppressed: `ax_main.tick_params(labelbottom=False)` |

## Marks

- **Reference sample**: a plain `step` and nothing else --
  `ax.step(edges, np.append(counts, counts[-1]), where='post')`.
  No markers, no error bars.
- **Every other sample**: an `errorbar(..., fmt='o', ms=4)` *plus* a faint
  companion `step` at `alpha=0.55` (`0.7` in `plot_wb_mass`) in the same colour.
- **Open-marker variant** (`open_markers=True`):
  `markerfacecolor='none'`, `markeredgecolor=color`, `markeredgewidth=1.2`.
- Ratio panel repeats the same `fmt='o', ms=4` marks, one series per
  non-reference sample.

## Labels, legend, ratio panel

| convention | value |
|---|---|
| upper y label | `'Events'` (or `r'Events ($\mathcal{L}=1.2$ fb$^{-1}$)'`) |
| ratio y label | `'Ratio'` |
| x label | on the ratio axis only |
| legend | on the upper panel only, `loc='best'` (the default); `plot_wb_mass` uses `fontsize=11, borderpad=0.3, labelspacing=0.3, handletextpad=0.4` |
| ratio legend | deliberately omitted -- "curve styles only, keep legend off for a cleaner view" |
| reference line | `ax_ratio.axhline(1, linestyle='--', color=...)` -- colour is `'red'`, `'C0'`, or the reference sample's own colour depending on the function |

## Ratio y limits

The function default is `ratio_ylim=(0.99, 1.01)`, but **no real call site uses
it**.  Every actual call in the user's script picks one of:

    (0.85, 1.15)    (0.75, 1.25)    (0.5, 1.5)

so that ladder, not the signature default, is the operative convention.

## Errors

- Unweighted paths use Poisson `np.sqrt(counts)` and
  `ratio * sqrt(1/numer + 1/denom)`.
- The **weighted** path, `weighted_hist` (l.9), already uses
  `sqrt(sum of w^2)` -- so `sqrt(sumw2)` is the user's own convention for
  weighted events, not a departure from it.
- `plot_wb_mass` propagates the ratio error as
  `ratio * sqrt((err_num/num)^2 + (err_den/den)^2)`, identical to the
  `ratio()` helper in `plot_interference.py`.

## Output

| convention | value |
|---|---|
| save | `plt.savefig(outname, bbox_inches='tight', dpi=300)` |
| close | `plt.close(fig)` |

---

# Where `plot_interference_userstyle.py` departs, and why

1. **Errors are `sqrt(sumw2)`, never Poisson.**  These are weighted events --
   the interference blocks carry signed weights and `(I,I)` has a negative
   integral, so `sqrt(N)` is meaningless here.  As noted above this matches the
   user's own `weighted_hist`, so it is a departure from their *unweighted*
   helpers only.

2. **Upper y label is `d sigma per bin [pb]`, not `Events`.**  The committed bin
   contents are cross sections in picobarns (summing an observable's bins
   reproduces that sample's `xsec`), so `Events` would be wrong.

3. **Ratio limits are chosen per figure from the user's ladder.**  The smallest
   of `(0.99,1.01) / (0.85,1.15) / (0.75,1.25) / (0.5,1.5)` that contains every
   plotted point *plus its error bar* is used.  Nothing is clipped.  For `C_nn`
   the 4-block ratio runs from 0.63 to 1.37 and escapes even `(0.5, 1.5)` once
   error bars are included, so that one figure uses `(0.4, 1.6)` -- rounded up
   to a clean 0.1 so the axis still reads like theirs.  Clipping to the
   signature default `(0.99, 1.01)` would have hidden the entire effect the
   plots exist to show.

4. **Headroom is added to the upper panel** (`x1.18`, `x1.45` on the
   block-decomposition figures) before calling `legend()`.  The user's script
   calls a bare `.legend()`; these distributions are sharply peaked and fill
   their frame, so `loc='best'` landed the legend on top of the data.  The
   convention (`loc='best'`, upper panel) is kept; only the axis range moved.

5. **A dotted grey line at y=0** is drawn on the block-decomposition figures.
   The user's script has no such line because none of their observables go
   negative; the summed interference here does, and the zero crossing is the
   physics.

6. **chi2 is written into the legend labels.**  The user's script puts no text
   boxes on its figures, so the legend was the only in-style place to carry the
   numbers.  `plot_interference.py` uses a text box instead.
