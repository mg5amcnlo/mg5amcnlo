#!/usr/bin/env python3
"""Compare the independent fixed-order runs from resonance_recoil.md.

Usage: compare_resonance_recoil.py --baseline old.HwU --local new.HwU
       --cutcheck varied.HwU --output comparison

Input errors must be Monte Carlo errors, including correlated counterevents.
The script compares absolute bin cross sections. It does not assume independent
bins, form a diagonal chi-square p-value, or normalize away rate differences.
Bonferroni-adjusted marginal Gaussian tests account for all bins and all pairs;
this correction does not require independence between bins. The normal
approximation to the Monte Carlo uncertainty still needs sufficient sampling.
"""

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist

import numpy as np


@dataclass
class Histogram:
    title: str
    bins: np.ndarray

    @property
    def values(self):
        return self.bins[:, 2]

    @property
    def errors(self):
        return self.bins[:, 3]


def read_hwu(path):
    """Read central values and their MC errors from a single-weight HwU file."""
    histograms = {}
    title = None
    rows = []
    expected = 0
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        match = re.match(r'<histogram>\s+(\d+)\s+"([^"]+)"', line)
        if match:
            if title is not None:
                raise ValueError(f"Unclosed histogram in {path}")
            expected = int(match[1])
            title = match[2].split("|", 1)[0].strip()
            rows = []
        elif line.startswith("<\\histogram>"):
            if title is None or len(rows) != expected:
                raise ValueError(f"Wrong number of bins in {path}: {title}")
            if title in histograms:
                raise ValueError(f"Duplicate histogram {title}")
            bins = np.asarray(rows, dtype=float)
            if not np.all(np.isfinite(bins)) or np.any(bins[:, 3] < 0):
                raise ValueError(f"Invalid histogram data: {title}")
            if np.any(bins[:, 1] <= bins[:, 0]):
                raise ValueError(f"Invalid bin boundaries: {title}")
            histograms[title] = Histogram(title, bins)
            title = None
        elif title is not None and line and not line.startswith("#"):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("Expected central value and MC error only")
            rows.append([float(v.replace("D", "E")) for v in fields])
    if title is not None or not histograms:
        raise ValueError(f"Incomplete HwU file {path}")
    return histograms


def compare_pair(reference, candidate, pair):
    rows = []
    if reference.keys() != candidate.keys():
        raise ValueError(f"Histogram sets differ for {pair}")
    for title, ref in reference.items():
        other = candidate[title]
        if ref.bins.shape != other.bins.shape or not np.array_equal(
            ref.bins[:, :2], other.bins[:, :2]
        ):
            raise ValueError(f"Bin boundaries differ: {pair}, {title}")
        for index, (a, b) in enumerate(zip(ref.bins, other.bins)):
            sigma = math.hypot(a[3], b[3])
            difference = b[2] - a[2]
            if sigma == 0 and difference != 0:
                raise ValueError(f"Difference with no error: {pair}, {title}")
            pull = float(difference / sigma) if sigma else None
            ratio = float(b[2] / a[2]) if a[2] != 0 else None
            ratio_error = (
                float(math.hypot(b[3] / a[2], b[2] * a[3] / a[2] ** 2))
                if a[2] != 0 else None
            )
            rows.append({
                "pair": pair, "histogram": title, "bin": index + 1,
                "low": float(a[0]), "high": float(a[1]),
                "reference": float(a[2]), "reference_error": float(a[3]),
                "candidate": float(b[2]), "candidate_error": float(b[3]),
                "difference": float(difference), "difference_error": sigma,
                "ratio": ratio, "ratio_error": ratio_error, "pull": pull,
                "p_value": math.erfc(abs(pull) / math.sqrt(2))
                if pull is not None else None,
                "sparse": bool(max(abs(a[2]), abs(b[2])) < 3 * sigma),
            })
    return rows


LABELS = {
    "m bjet e nu broad": r"$m(b\mathrm{\ jet},e^+,\nu_e)$ [GeV]",
    "m bjet e nu peak": r"$m(b\mathrm{\ jet},e^+,\nu_e)$ [GeV]",
    "m e nu": r"$m(e^+,\nu_e)$ [GeV]",
    "pt bjet": r"$p_T(b\mathrm{\ jet})$ [GeV]",
    "eta bjet": r"$\eta(b\mathrm{\ jet})$",
    "pt recoil jet": r"$p_T(\mathrm{recoil\ jet})$ [GeV]",
    "eta recoil jet": r"$\eta(\mathrm{recoil\ jet})$",
    "pt positron": r"$p_T(e^+)$ [GeV]",
    "eta positron": r"$\eta(e^+)$",
    "pt bjet e nu": r"$p_T(b\mathrm{\ jet},e^+,\nu_e)$ [GeV]",
    "delta R bjet positron": r"$\Delta R(b\mathrm{\ jet},e^+)$",
    "jet multiplicity": r"Resolved jet multiplicity",
}
COLORS = {"baseline": "#3b3b3b", "local": "#c65a12", "cutcheck": "#087f8c"}
NAMES = {"baseline": "Previous Granny", "local": "Local recoil",
         "cutcheck": "Local recoil, varied cutoffs"}


def draw_spectrum(axes, runs, title, compact=False):
    upper, ratio_ax = axes[:2]
    ref = runs["baseline"][title]
    centers = np.mean(ref.bins[:, :2], axis=1)
    edges = np.r_[ref.bins[:, 0], ref.bins[-1, 1]]
    upper.stairs(ref.values, edges, color=COLORS["baseline"], label=NAMES["baseline"])
    upper.fill_between(centers, ref.values-ref.errors, ref.values+ref.errors,
                       step="mid", color=COLORS["baseline"], alpha=0.18)
    good = ref.values > 3 * ref.errors
    relative_error = np.divide(ref.errors, ref.values, out=np.zeros_like(ref.values),
                               where=good)
    ratio_ax.fill_between(centers, 1-relative_error, 1+relative_error,
                          step="mid", color=COLORS["baseline"], alpha=0.18)
    ratio_limits = [0.98, 1.02]
    for label, marker in [("local", "o"), ("cutcheck", "s")]:
        hist = runs[label][title]
        upper.errorbar(centers, hist.values, yerr=hist.errors, fmt=marker,
                       ms=2.5 if compact else 3.5, elinewidth=0.8,
                       color=COLORS[label], label=NAMES[label])
        # Numerator uncertainty is shown separately from the reference band.
        y = hist.values[good] / ref.values[good]
        dy = hist.errors[good] / ref.values[good]
        ratio_ax.errorbar(centers[good], y, yerr=dy, fmt=marker,
                          ms=2.5, elinewidth=0.8, color=COLORS[label])
        ratio_limits.extend((y-dy).tolist() + (y+dy).tolist())
        if len(axes) > 2:
            sigma = np.hypot(ref.errors, hist.errors)
            valid = sigma > 0
            pull = (hist.values[valid]-ref.values[valid])/sigma[valid]
            axes[2].plot(centers[valid], pull, marker, ms=3, color=COLORS[label])
    upper.set_ylabel("Cross section / bin [pb]")
    upper.set_title(LABELS.get(title, title), fontsize=11)
    positive = ref.values[ref.values > 0]
    if len(positive) and positive.max()/positive.min() > 100:
        from matplotlib.ticker import FixedLocator, SymmetricalLogLocator
        linear_threshold = max(positive.max()*1e-3, 1e-9)
        upper.set_yscale("symlog", linthresh=linear_threshold)
        locator = SymmetricalLogLocator(base=10, linthresh=linear_threshold)
        locator.set_params(numticks=7)
        ticks = locator.tick_values(*upper.get_ylim())
        # Decade ticks inside the linear region crowd the zero label.
        ticks = [tick for tick in ticks if tick == 0 or abs(tick) >= linear_threshold]
        upper.yaxis.set_major_locator(FixedLocator(ticks))
    upper.grid(alpha=0.2)
    ratio_ax.axhline(1, color="black", lw=0.7)
    ratio_ax.set_ylabel("Ratio")
    low, high = min(ratio_limits), max(ratio_limits)
    span = max(high-low, 0.04)
    ratio_ax.set_ylim(low-0.1*span, high+0.1*span)
    ratio_ax.grid(alpha=0.2)
    axes[-1].set_xlabel(LABELS.get(title, title))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    if len(axes) > 2:
        axes[2].axhspan(-2, 2, color="grey", alpha=0.12)
        axes[2].axhline(0, color="black", lw=0.7)
        axes[2].set_ylabel("Pull")
        axes[2].grid(alpha=0.2)
    if title == "jet multiplicity":
        axes[-1].set_xticks([2, 3])
    axes[-1].set_xlim(edges[0], edges[-1])


def make_plots(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    titles = [title for title in runs["baseline"] if not title.startswith("total rate")]
    with PdfPages(output / "differential_comparison.pdf") as pdf:
        for title in titles:
            fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1, 1]})
            draw_spectrum(axes, runs, title)
            axes[0].legend(frameon=False, fontsize=9)
            fig.suptitle("Resonance recoil validation: fixed-order NLO", fontsize=14)
            fig.text(0.5, 0.018,
                     "First/last bins include underflow/overflow. Errors are MC statistical errors.\n"
                     "Ratios with baseline significance below 3 are omitted; pulls use both run errors.",
                     ha="center", fontsize=8)
            fig.tight_layout(rect=(0, 0.055, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    overview = [title for title in ["m bjet e nu peak", "m e nu", "pt bjet",
                "pt recoil jet", "pt positron", "eta recoil jet"] if title in titles]
    if overview:
        fig = plt.figure(figsize=(14, 13))
        grid = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.25)
        for index, title in enumerate(overview):
            inner = grid[index//2, index % 2].subgridspec(2, 1,
                       height_ratios=[3, 1], hspace=0.04)
            top = fig.add_subplot(inner[0])
            ratio = fig.add_subplot(inner[1], sharex=top)
            draw_spectrum([top, ratio], runs, title, compact=True)
            if index == 0:
                top.legend(frameon=False, fontsize=8)
        fig.suptitle("Previous Granny and local recoil: differential NLO comparison", fontsize=15)
        fig.subplots_adjust(top=0.94, bottom=0.075)
        fig.text(0.5, 0.02, "First/last bins include underflow/overflow. Error bars: MC statistics.",
                 ha="center", fontsize=10)
        fig.savefig(output / "differential_overview.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for label in NAMES:
        parser.add_argument("--" + label, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {label: read_hwu(getattr(args, label)) for label in NAMES}
    rates = {}
    for label, histograms in runs.items():
        rate = histograms["total rate"]
        if len(rate.values) != 1:
            raise ValueError("Use the resonance recoil analysis with one total-rate bin")
        total = float(rate.values[0])
        closure = {title: float(hist.values.sum()-total)
                   for title, hist in histograms.items()
                   if not title.startswith("total rate")}
        # HwU's iteration weighting can give small closure residuals when a
        # sparse bin is empty in some iterations. Require these to be below
        # one percent of the total MC uncertainty, or text-rounding precision.
        closure_tolerance = max(1e-6*max(abs(total), 1), 0.01*rate.errors[0])
        if any(abs(v) > closure_tolerance for v in closure.values()):
            raise ValueError(f"Histogram normalization does not close for {label}: {closure}")
        rates[label] = {"value_pb": total, "error_pb": float(rate.errors[0]),
                        "relative_error": float(rate.errors[0]/abs(total)),
                        "closure_pb": closure,
                        "closure_tolerance_pb": float(closure_tolerance)}
    rows = []
    for reference, candidate in [("baseline", "local"), ("local", "cutcheck"),
                                  ("baseline", "cutcheck")]:
        rows.extend(compare_pair(runs[reference], runs[candidate],
                                 candidate + " vs " + reference))
    trials = sum(row["pull"] is not None for row in rows)
    for row in rows:
        row["p_bonferroni"] = min(1.0, row["p_value"]*trials) if row["p_value"] is not None else None
    threshold = NormalDist().inv_cdf(1-0.05/(2*trials)) if trials else None
    worst = sorted((row for row in rows if row["pull"] is not None),
                   key=lambda row: abs(row["pull"]), reverse=True)
    flagged = [row for row in worst if row["p_bonferroni"] < 0.05]
    summary = {"inputs": {label: {"path": str(getattr(args, label)),
                "sha256": hashlib.sha256(getattr(args, label).read_bytes()).hexdigest()}
                for label in NAMES}, "rates": rates, "number_of_tests": trials,
                "familywise_5pct_pull_threshold": threshold,
                "worst_bins": worst[:20], "flagged_bins": flagged,
                "statistical_method": "Independent runs; correlated counterevents included by HwU. "
                "Absolute bin cross sections, Gaussian marginal pulls, Bonferroni correction "
                "over all bins and all pairs. No independence assumption between bins. "
                "Sparse tails require caution about the Gaussian approximation."}
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    with (args.output / "bin_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Differential recoil comparison", "",
             "| Run | Cross section [pb] | Relative MC error |", "| --- | ---: | ---: |"]
    for label, rate in rates.items():
        lines.append(f"| {NAMES[label]} | {rate['value_pb']:.7f} +/- {rate['error_pb']:.7f} | "
                     f"{100*rate['relative_error']:.4f}% |")
    lines += ["", f"{trials} marginal tests; 5% familywise threshold: |pull| > {threshold:.3f}.",
              f"Bins exceeding that threshold: {len(flagged)}.", "",
              "| Comparison | Spectrum | Maximum absolute pull | Bins above 3 sigma |",
              "| --- | --- | ---: | ---: |"]
    keys = dict.fromkeys((row["pair"], row["histogram"]) for row in rows)
    for pair, title in keys:
        pulls = [abs(row["pull"]) for row in rows if row["pair"] == pair and
                 row["histogram"] == title and row["pull"] is not None]
        if pulls:
            lines.append(f"| {pair} | {title} | {max(pulls):.3f} | {sum(p > 3 for p in pulls)} |")
    lines += ["", summary["statistical_method"], "",
              "Every spectrum includes underflow and overflow in its first and last bins. "
              "Each spectrum must sum to the total rate within text precision or 1% of the "
              "total MC uncertainty, allowing for HwU iteration weighting in sparse bins. "
              "The requested 0.1% precision "
              "refers to the integrated cross section; individual bins have larger errors.", ""]
    (args.output / "comparison.md").write_text("\n".join(lines))
    make_plots(runs, args.output)
    print("\n".join(lines[:10]))
    print(f"Flagged bins: {len(flagged)}; output: {args.output}")


if __name__ == "__main__":
    main()
