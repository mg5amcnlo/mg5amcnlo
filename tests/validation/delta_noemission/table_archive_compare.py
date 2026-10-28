"""Compare shipped Sudakov alternatives using their unchanged Fortran interpolators.

Usage: python table_archive_compare.py OUTPUT_DIRECTORY BUILD_DIRECTORY
All probabilities are stored T(q,m)/T(Q,m), without roots or external PDFs.
No shower is run and no production table is replaced.
"""
import csv
import ctypes as C
import json
from pathlib import Path
import sys

import numpy as np

from delta_evaluator import DeltaEvaluator, ROOT, _array, _ptr


def main():
    out, work = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=True)
    tables = {"default": ROOT / "vendor/SudGen/sudakov_template.f"}
    tables.update({f"PDFmode{i}": ROOT / f"vendor/SudGen/sudakov_r515_PDFmode{i}.f.gz"
                   for i in range(3)})
    masses = _array([.33, .33, .5, 1.5, 4.8, 0, 0, 0, 0, 0,
                     .000510998928, 0, .1056583715, 0, 1.77682, 0, 0, 0, 0, 0, 0])
    qs = np.unique(np.concatenate((np.geomspace(5., 200., 121), [10., 20., 50., 100.])))
    rows, provenance = [], {}
    for label, table in tables.items():
        with DeltaEvaluator(work=work / label, table=table) as evaluator:
            fn = evaluator.lib.pysudakov_
            fn.restype = C.c_double

            def value(q, mass, pdg, topology):
                args = [C.c_double(q), C.c_double(mass), C.c_int(pdg), C.c_int(topology)]
                return fn(*[C.byref(x) for x in args], _ptr(masses))

            for mass in [100., 500., 1000.]:
                for pdg in [2, 6, 21]:
                    for topology in [1, 2, 3, 4]:
                        denominator = value(200., mass, pdg, topology)
                        for q in qs:
                            numerator = value(q, mass, pdg, topology)
                            rows.append(dict(table=label, pdg=pdg, type=topology,
                                             mass=mass, stop=float(q), start=200.,
                                             num=numerator, den=denominator,
                                             ratio=numerator / denominator))
            provenance[label] = evaluator.provenance
    with (out / "ratios.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    benchmark = [r for r in rows if r["mass"] == 500. and r["stop"] == 20.]
    (out / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps(dict(
        definition="T(q,m)/T(Q,m); Q=200 GeV; no roots or external beam-PDF factors",
        topology={1: "II", 2: "FF", 3: "IF", 4: "FI"},
        top_note="ID 6 in these tables has zero mass; this is not a physical massive-top calculation",
        tables=provenance), indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (pdg, topology, title) in zip(axes.flat, [
            (21, 1, "Incoming gluon, II"), (2, 1, "Incoming up quark, II"),
            (6, 4, "Final quark (ID 6), FI"), (6, 2, "Final quark (ID 6), FF")]):
        for label in tables:
            selected = [r for r in rows if r["table"] == label and r["pdg"] == pdg
                        and r["type"] == topology and r["mass"] == 500.]
            ax.plot([r["stop"] for r in selected], [r["ratio"] for r in selected],
                    label=label, linestyle="--" if label == "default" else "-",
                    linewidth=2 if label == "default" else 1.3,
                    alpha=1 if label == "default" else .8)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.grid(alpha=.25)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r"Stored ratio $T(q,m)/T(Q,m)$")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(1.e-6, 1.2)
    axes[0, 0].legend(fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel(r"Stopping scale $q$ [GeV]")
    fig.suptitle(r"Shipped tables: $Q=200$ GeV, $m_{\rm dip}=500$ GeV")
    fig.text(.5, .012, "Stored factors, without normalization conversion or beam PDFs. "
             "All three archive FSR curves coincide. ID 6 has zero table mass.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, .97))
    fig.savefig(out / "comparison.pdf")
    fig.savefig(out / "comparison.png", dpi=160)
    print(json.dumps([r for r in benchmark if (r["pdg"], r["type"]) in
                     [(21, 1), (21, 3), (2, 1), (2, 3), (6, 4), (6, 2)]], indent=2))


if __name__ == "__main__":
    main()
