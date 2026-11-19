#!/usr/bin/env python3
"""Compare SCALUP in the four central LHE samples, separated into S/H events."""

import argparse
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Analyses.matched_run_inputs import validate_completion

STYLES = (
    ("C_GLB", "default recoil", "#2463a4"),
    ("C_LOC", "local recoil", "#d35400"),
    ("C_Q", "local + collinear patch", "#228545"),
    ("C_QT", "local + both patches", "#9553a4"),
)


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_events(stream):
    """Read explicit MC@NLO types; never infer the type from a weight's sign."""
    scales, weights, types = array("d"), array("d"), array("b")
    event_norm = None
    extra_scale_records = 0
    for line in stream:
        norm = re.search(rb"\b(average|sum|bias)\s*=\s*event_norm\b", line)
        if norm:
            event_norm = norm[1].decode()
        if line.strip() != b"<event>":
            continue
        header = next(stream).split()
        while not header:
            header = next(stream).split()
        if len(header) != 6:
            raise ValueError("Expected six LHE event-header fields")
        nup = int(header[0])
        weight, scale = (float(x.replace(b"D", b"E")) for x in header[2:4])
        if not math.isfinite(weight) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid weight/SCALUP: {weight}, {scale}")
        for _ in range(nup):
            if len(next(stream).split()) != 13:
                raise ValueError("Malformed LHE particle row")
        kind = None
        for line in stream:
            stripped = line.strip()
            if stripped.startswith(b"#aMCatNLO "):
                kind = int(stripped.split()[1])
            if stripped.startswith(b"<scales"):
                extra_scale_records += 1
            if stripped == b"</event>":
                break
        else:
            raise ValueError("Unclosed LHE event")
        if kind not in (1, 2) or nup != (4 if kind == 1 else 5):
            raise ValueError(f"Inconsistent stable-top S/H record: type={kind}, NUP={nup}")
        scales.append(scale)
        weights.append(weight)
        types.append(kind)
    if event_norm not in ("average", "sum"):
        raise ValueError(f"Unsupported or missing event_norm: {event_norm}")
    if not scales:
        raise ValueError("Empty event file")
    return {"scales": np.frombuffer(scales, dtype=np.float64),
            "weights": np.frombuffer(weights, dtype=np.float64),
            "types": np.frombuffer(types, dtype=np.int8),
            "event_norm": event_norm, "extra_scale_records": extra_scale_records}


def read_sample(name, directory):
    completion = validate_completion(directory, name)
    path = directory / "Events" / name / "events.lhe.gz"
    expected = completion["output_sha256"][path.name]
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f"Changed event file: {path}")
    with gzip.open(path, "rb") as stream:
        result = read_events(stream)
    count = len(result["scales"])
    if count != int(completion["inputs"]["run_spec"]["nevents"]):
        raise ValueError(f"Wrong event count in {path}: {count}")
    if result["extra_scale_records"]:
        raise ValueError(f"Additional scale records require inspection: {path}")
    result.update(input=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
                  else str(path), sha256=digest)
    return name, result


def bin_events(sample, kind, edges):
    selected = sample["types"] == kind
    q = sample["scales"][selected]
    raw_weights = sample["weights"][selected]
    factor = 1 / len(sample["scales"]) if sample["event_norm"] == "average" else 1
    weights = raw_weights * factor
    signed = np.histogram(q, edges, weights=weights)[0]
    variance = np.histogram(q, edges, weights=weights**2)[0]
    absolute = np.histogram(q, edges, weights=np.abs(weights))[0]
    counts = np.histogram(q, edges)[0]
    negatives = np.histogram(q[weights < 0], edges)[0]
    total_abs = float(np.abs(weights).sum())
    probability = absolute / total_abs
    # Cov(bin,total)=Var(bin) for this normalized subset distribution.
    shape_variance = ((1 - probability)**2 * variance
                      + probability**2 * np.maximum(variance.sum() - variance, 0)) / total_abs**2
    assert counts.sum() == len(q), "Histogram does not cover all SCALUP values"
    assert np.isclose(signed.sum(), weights.sum(), rtol=1e-10, atol=1e-8)
    assert np.isclose(probability.sum(), 1, rtol=1e-12)
    quantiles = np.quantile(q, [0, .01, .1, .5, .9, .99, 1])
    summary = {
        "events": int(selected.sum()), "positive": int((weights > 0).sum()),
        "negative": int((weights < 0).sum()), "zero_weight": int((weights == 0).sum()),
        "signed_cross_section_pb": float(weights.sum()),
        "sumw2_error_pb": float(np.sqrt(variance.sum())),
        "absolute_cross_section_pb": total_abs,
        "SCALUP_GeV_quantiles": dict(zip(("min", "p01", "p10", "median", "p90", "p99", "max"),
                                         map(float, quantiles))),
    }
    histogram = {"signed_pb": signed, "signed_error_pb": np.sqrt(variance),
                 "absolute_pb": absolute, "counts": counts, "negative_counts": negatives,
                 "abs_weight_probability": probability,
                 "abs_weight_probability_error": np.sqrt(shape_variance)}
    return histogram, summary


def draw_column(axes, kind, data, edges):
    widths = np.diff(np.log10(edges))
    upper, lower = axes
    for name, label, colour in STYLES:
        values = data[name][kind]
        for axis, key, err_key in (
                (upper, "signed_pb", "signed_error_pb"),
                (lower, "abs_weight_probability", "abs_weight_probability_error")):
            density = values[key] / widths
            error = values[err_key] / widths
            axis.stairs(density, edges, color=colour, label=label, linewidth=1.6)
            axis.fill_between(edges, np.r_[density - error, (density - error)[-1]],
                              np.r_[density + error, (density + error)[-1]],
                              step="post", color=colour, alpha=.12, linewidth=0)
    upper.set_title("S events (Born-like)" if kind == 1 else "H events (real-emission)")
    upper.set_ylabel(r"$d\sigma/d\log_{10}(Q/\mathrm{GeV})$ [pb]")
    lower.set_ylabel(r"$\frac{1}{\sum |w|}\,\frac{d\sum |w|}{d\log_{10}(Q/\mathrm{GeV})}$")
    lower.set_xlabel(r"LHE shower starting scale $Q=\mathrm{SCALUP}$ [GeV]")
    upper.legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlim(edges[0], edges[-1])
        axis.axhline(0, color=".55", linewidth=.65)
        axis.grid(alpha=.18)
    lower.set_ylim(bottom=0)


def plot(data, edges, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    footer = ("All 1,000,000 LHE events/sample; no analysis cuts. "
              "Top: signed weights. Bottom: unit-normalized absolute weights.")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex="col",
                             gridspec_kw={"height_ratios": [1.3, 1], "hspace": .12, "wspace": .3})
    for column, kind in enumerate((1, 2)):
        draw_column(axes[:, column], kind, data, edges)
    fig.suptitle("Shower starting scales in the four central predictions", fontsize=14, y=.98)
    fig.text(.5, .018, footer, ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.1)
    fig.savefig(output / "shower_starting_scales.pdf")
    fig.savefig(output / "shower_starting_scales.png", dpi=170)
    plt.close(fig)
    for kind, label in ((1, "S"), (2, "H")):
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3, 1], "hspace": .12})
        draw_column(axes, kind, data, edges)
        fig.text(.5, .02, "Signed weights above; unit-normalized absolute weights below. No analysis cuts.",
                 ha="center", fontsize=8)
        fig.subplots_adjust(left=.16, right=.97, top=.94, bottom=.1)
        fig.savefig(output / f"{label}_events.pdf")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--central-dir", type=Path, default=ROOT / "central_runs_fixed")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "Analyses/lhe_shower_scales")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    samples = {}
    with ProcessPoolExecutor(max_workers=min(4, args.workers)) as pool:
        futures = [pool.submit(read_sample, name, args.central_dir.resolve() / name)
                   for name, _, _ in STYLES]
        for future in as_completed(futures):
            name, sample = future.result()
            samples[name] = sample
            print(f"{name}: {len(sample['scales']):,} events, "
                  f"SCALUP {sample['scales'].min():.6g}--{sample['scales'].max():.6g} GeV", flush=True)
    lower = math.floor(math.log10(min(x["scales"].min() for x in samples.values())))
    upper = math.ceil(math.log10(max(x["scales"].max() for x in samples.values())))
    edges = np.geomspace(10.**lower, 10.**upper, (upper - lower) * 12 + 1)
    data, metadata = {}, {}
    for name, _, _ in STYLES:
        sample = samples[name]
        data[name] = {}
        metadata[name] = {key: sample[key] for key in ("input", "sha256", "event_norm", "extra_scale_records")}
        metadata[name]["events"] = len(sample["scales"])
        metadata[name]["absolute_LHE_weight_range_pb"] = [float(np.abs(sample["weights"]).min()),
                                                        float(np.abs(sample["weights"]).max())]
        for kind, label in ((1, "S"), (2, "H")):
            data[name][kind], metadata[name][label] = bin_events(sample, kind, edges)
        for raw in ("scales", "weights", "types"):
            del sample[raw]
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "observable": "LHE event-header SCALUP in GeV; no shower or analysis cuts applied",
        "classification": "#aMCatNLO 1=S, 2=H; checked against NUP=4 and 5",
        "normalization": "Signed XWGTUP/N_all for event_norm=average; S/H are contributions to the full sample",
        "errors": "sqrt(sum(w^2))/N_all; no additional integration-normalisation error",
        "shape": "Per-S/H unit-normalized absolute weights, including subset-normalisation covariance",
        "bins": "12 equal bins per decade; every event is inside the common range",
        "bin_edges_GeV": edges.tolist(), "samples": metadata,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = ["sample", "event_type", "Q_low_GeV", "Q_high_GeV"] + list(data["C_GLB"][1])
    with (output / "histograms.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, _, _ in STYLES:
            for kind, label in ((1, "S"), (2, "H")):
                for index in range(len(edges) - 1):
                    row = dict(sample=name, event_type=label, Q_low_GeV=edges[index], Q_high_GeV=edges[index + 1])
                    row.update({key: values[index] for key, values in data[name][kind].items()})
                    writer.writerow(row)
    plot(data, edges, output)
    (output / "README.txt").write_text(
        "LHE SHOWER STARTING SCALES\n\n"
        "shower_starting_scales.pdf/png: overview, separate S/H columns.\n"
        "S_events.pdf and H_events.pdf: individual plots.\n"
        "The four curves are C_GLB, C_LOC, C_Q and C_QT.\n\n"
        "Q is the scalar SCALUP in each event header, before showering;\n"
        "it is not the transverse momentum of an emitted parton. These\n"
        "non-Delta, unmerged samples contain no extra scale records.\n"
        "Pythia also applies its kinematic/dipole limits to this input.\n"
        "S/H labels use #aMCatNLO 1/2 and agree with the Born/real\n"
        "particle multiplicity; the sign of the event weight is unrelated.\n\n"
        "Upper panels: signed differential cross sections per log10 Q,\n"
        "with sum-of-squared-weights errors. S and H use the same full\n"
        "sample denominator; their integrals add to the LHE prediction.\n"
        "H distributions or their integrals can be negative.\n"
        "Lower panels: separately unit-normalized |weight| distributions,\n"
        "showing the assigned scales without positive/negative cancellations.\n"
        "For these equal-absolute-weight event samples this is the event\n"
        "count distribution. These lower panels are not cross sections.\n"
        "Shading gives statistical errors; integration errors are separate.\n\n"
        "All four complete million-event files are included, without\n"
        "jet cuts or selecting events accepted by the shower analysis.\n"
        "No reweighting to the generation-summary cross section is applied.\n"
        "The common logarithmic bins include every event, without clipping.\n"
        "CSV entries are bin-integrated quantities; plots divide by\n"
        "Delta log10 Q. summary.json records input hashes, counts,\n"
        "signed contributions and SCALUP quantiles.\n\n"
        "Reproduce from the repository root:\n"
        "  python3 Analyses/plot_pp_tj_lhe_scales.py\n")
    print(f"Plots and numerical data: {output}", flush=True)


if __name__ == "__main__":
    main()
