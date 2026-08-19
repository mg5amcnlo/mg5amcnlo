#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- MG7 paper style ------------------------------------------------------
# Self-contained rcParams block: paste it into any script, nothing below it
# needs to know about it.  Follows the MG7 paper's plotexample/dummyplot.py.
import shutil
import matplotlib.colors as _mcolors

_MG7_COLORS = list(_mcolors.TABLEAU_COLORS.values())
_MG7_COLORS[0], _MG7_COLORS[1], _MG7_COLORS[3] = 'blue', 'black', 'red'
matplotlib.rcParams.update({
    # LaTeX text if it is installed, Computer Modern mathtext otherwise
    'text.usetex': bool(shutil.which('latex')) and bool(shutil.which('dvipng')),
    'mathtext.fontset': 'cm', 'font.family': 'serif', 'font.size': 14,
    'figure.figsize': (7 * 0.75, 6.0),   # paper width -- do NOT change 7*0.75
    'lines.linewidth': 1.2, 'lines.markersize': 4, 'errorbar.capsize': 2,
    'axes.prop_cycle': matplotlib.cycler(color=_MG7_COLORS),
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'legend.frameon': False, 'legend.fontsize': 9,
    'legend.handlelength': 2.0, 'legend.columnspacing': 1.4,
    'savefig.bbox': 'tight',
})
# --------------------------------------------------------------------------
from LHEParser import EventFile, FourMomentum

def weighted_hist(values, bins, weights):
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    sumw2, _ = np.histogram(values, bins=edges, weights=weights**2)
    errs = np.sqrt(sumw2)
    return counts, errs, edges


def delta_phi(phi_a, phi_b):
    return abs((phi_a - phi_b + np.pi) % (2 * np.pi) - np.pi)


def delta_r(mom_a, mom_b):
    return np.sqrt(
        (mom_a.pseudorapidity - mom_b.pseudorapidity) ** 2
        + delta_phi(mom_a.phi, mom_b.phi) ** 2
    )


def two_body_rest_frame_momentum(parent_mom, child_a_mom, child_b_mom):
    parent_mass = parent_mom.mass
    if parent_mass <= 0:
        return 0.0

    parent_mass2 = parent_mass ** 2
    child_sum = child_a_mom.mass + child_b_mom.mass
    child_diff = child_a_mom.mass - child_b_mom.mass
    radicand = (
        (parent_mass2 - child_sum ** 2)
        * (parent_mass2 - child_diff ** 2)
    )
    return np.sqrt(max(0.0, radicand)) / (2.0 * parent_mass)


def passes_wb_mass_cut(w_part, b_part, mass_cut):
    if mass_cut is None:
        return True

    wb_mass = (FourMomentum(w_part) + FourMomentum(b_part)).mass
    distance = abs(wb_mass - mass_cut["center"])
    if mass_cut["mode"] == "outside":
        return distance > mass_cut["width"]
    if mass_cut["mode"] == "inside":
        return distance < mass_cut["width"]
    raise ValueError(f"Unknown Wb mass-cut mode: {mass_cut['mode']}")


def event_passes_wb_mass_cut(w_plus, b_parts, w_minus, bbar_parts, mass_cut):
    if mass_cut is None:
        return True

    candidates = list(zip(w_plus, b_parts)) + list(zip(w_minus, bbar_parts))
    return any(passes_wb_mass_cut(w_part, b_part, mass_cut)
               for w_part, b_part in candidates)


def bb_observable_value(b_part, bbar_part, observable):
    b_mom = FourMomentum(b_part)
    bbar_mom = FourMomentum(bbar_part)
    if observable == "bb-dphi":
        return delta_phi(b_mom.phi, bbar_mom.phi)
    if observable == "bb-dr":
        return delta_r(b_mom, bbar_mom)
    raise ValueError(f"Unknown bb observable: {observable}")


def read_wp_momenta(lhe_path, pid=5, final_state_only=False):
    """
    Parse events from an LHE file and return arrays of particle/antiparticle angles.
    """
    lhe = EventFile(lhe_path)
    thetas, phis, pts, dphis, wgts = [], [], [], [], []

    for event in lhe:
        weight = float(event.wgt)
        wp = FourMomentum()
        wp_anti = FourMomentum()
        for part in event:
            if final_state_only and part.status != 1:
                continue
            if part.pid == pid:  # particle
                wp += FourMomentum(part)
            elif part.pid == -pid:
                wp_anti += FourMomentum(part)
        thetas.append(wp.theta)
        phis.append(wp.phi)
        dphis.append(delta_phi(wp.phi, wp_anti.phi))
        pts.append(wp.pt)
        wgts.append(weight)


    return np.array(thetas), np.array(phis), np.array(pts), np.array(dphis), np.array(wgts)


def wb_observable_value(w_part, b_part, observable):
    w_mom = FourMomentum(w_part)
    b_mom = FourMomentum(b_part)
    wb = w_mom + b_mom
    if observable == "mass":
        return wb.mass
    if observable in ("wb-pt", "top-pt"):
        return wb.pt
    if observable == "b-pt":
        return b_mom.pt
    if observable == "b-pstar":
        return two_body_rest_frame_momentum(wb, w_mom, b_mom)
    if observable == "wb-dr":
        return delta_r(w_mom, b_mom)
    raise ValueError(f"Unknown Wb observable: {observable}")


def read_wb_observable(lhe_paths, include_antitop=False, observable="mass",
                       mass_cut=None):
    """
    Parse events from one or more LHE files and return one observable for
    reconstructed Wb systems using direct charge pairing:
    - always W+ + b
    - optionally (include_antitop=True) also W- + b~
    """
    values = []
    wgts = []
    paths = [lhe_paths] if isinstance(lhe_paths, str) else lhe_paths

    for path in paths:
        lhe = EventFile(path)
        for event in lhe:
            try:
                weight = float(event.wgt)
                final_particles = [p for p in event if p.status == 1]
                final_w_plus = [p for p in final_particles if p.pid == 24]
                final_w_minus = [p for p in final_particles if p.pid == -24]
                final_b = [p for p in final_particles if p.pid == 5]
                final_bbar = [p for p in final_particles if p.pid == -5]

                if observable in ("bb-dphi", "bb-dr"):
                    if not final_b or not final_bbar:
                        continue
                    if not event_passes_wb_mass_cut(
                        final_w_plus, final_b, final_w_minus, final_bbar,
                        mass_cut
                    ):
                        continue
                    for b_part, bbar_part in zip(final_b, final_bbar):
                        values.append(bb_observable_value(
                            b_part, bbar_part, observable
                        ))
                        wgts.append(weight)
                    continue

                if not final_w_plus or not final_b:
                    continue
                for w_part, b_part in zip(final_w_plus, final_b):
                    if not passes_wb_mass_cut(w_part, b_part, mass_cut):
                        continue
                    values.append(wb_observable_value(w_part, b_part, observable))
                    wgts.append(weight)

                if include_antitop:
                    for w_part, b_part in zip(final_w_minus, final_bbar):
                        if not passes_wb_mass_cut(w_part, b_part, mass_cut):
                            continue
                        values.append(wb_observable_value(w_part, b_part, observable))
                        wgts.append(weight)
            except AssertionError:
                continue
            except ValueError:
                continue

    return np.array(values), np.array(wgts)


def read_wb_mass(lhe_paths, include_antitop=False):
    return read_wb_observable(lhe_paths, include_antitop=include_antitop,
                              observable="mass")


def make_ratio(numer, denom):
    """
    Compute bin-by-bin ratio numer/denom and propagate Poisson errors.
    """
    mask = denom > 0
    ratio = np.zeros_like(numer, dtype=float)
    ratio_err = np.zeros_like(numer, dtype=float)

    try:
        ratio[mask] = numer[mask] / denom[mask]
        ratio_err[mask] = ratio[mask] * np.sqrt((1 / np.where(numer > 0, numer, 1)) + (1 / denom[mask]))
    except:
        pass

    return ratio, ratio_err


def plot_hist_with_ratio(theta_d, theta_o, bins, xlabel, outname, legend=""):
    """
    Plot histograms of density (d) and onshell (o) samples with ratio inset.
    """
    counts_d, edges = np.histogram(theta_d, bins=bins)
    counts_o, _ = np.histogram(theta_o, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    ratio, ratio_err = make_ratio(counts_d, counts_o)

    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.4], hspace=0.06)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    # Main histogram
    ax_main.errorbar(centers, counts_d, yerr=np.sqrt(counts_d), fmt='o', label='onshell')
    ax_main.step(edges, np.append(counts_o, counts_o[-1]), where='post', label='onshell_v1' if not legend else legend)
    ax_main.set_ylabel('Events')
    ax_main.legend()
    ax_main.tick_params(labelbottom=False)

    # Ratio inset
    ax_ratio.errorbar(centers, ratio, yerr=ratio_err, fmt='o')
    ax_ratio.axhline(1, lw=0.8, linestyle='--', color='gray')  # reference line at ratio = 1
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.set_ylim(0.99, 1.01)

    # Improve layout to fill page
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def make_count_ratio(numer, denom):
    """
    Compute a count ratio numer/denom with simple Poisson error propagation.
    """
    ratio = np.full_like(numer, np.nan, dtype=float)
    ratio_err = np.full_like(numer, np.nan, dtype=float)
    mask = denom > 0

    ratio[mask] = numer[mask] / denom[mask]
    ratio_err[mask] = np.sqrt(
        np.where(numer[mask] > 0, numer[mask], 0.0) / denom[mask]**2
        + numer[mask]**2 / denom[mask]**3
    )

    return ratio, ratio_err


def find_histogram_index(hists, name):
    """
    Find a histogram by sample key first, then by label.
    """
    target = str(name).lower()
    for idx, (sample, _counts, _errs) in enumerate(hists):
        if sample.get("key", "").lower() == target:
            return idx
    for idx, (sample, _counts, _errs) in enumerate(hists):
        if sample["label"].lower() == target:
            return idx
    for idx, (sample, _counts, _errs) in enumerate(hists):
        if target in sample["label"].lower():
            return idx
    return None


def plot_hist_with_ratio_multi(samples, bins, xlabel, outname, reference_index=0,
                               ratio_ylim=(0.99, 1.01), open_markers=False):
    """
    Plot two or more histograms on one canvas and ratio all non-reference
    histograms to the reference sample.
    """
    if len(samples) < 2:
        raise ValueError("Need at least two samples to plot a ratio")

    ref_counts, edges = np.histogram(samples[reference_index]["values"], bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    histograms = []
    for sample in samples:
        counts, _ = np.histogram(sample["values"], bins=edges)
        histograms.append((sample, counts, np.sqrt(counts)))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.4], hspace=0.06)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    for idx, (sample, counts, errs) in enumerate(histograms):
        label = sample["label"]
        color = sample.get("color", None)
        if idx == reference_index:
            ax_main.step(edges, np.append(counts, counts[-1]), where='post',
                         label=f"{label}", color=color)
            continue
        marker_style = {}
        if open_markers:
            marker_style = {
                "markerfacecolor": "none",
                "markeredgecolor": color,
                "markeredgewidth": 1.2,
            }
        ax_main.errorbar(centers, counts, yerr=errs, fmt='o', ms=4,
                         label=label, color=color, **marker_style)
        ax_main.step(edges, np.append(counts, counts[-1]), where='post',
                     color=color, alpha=0.55)

    ax_main.set_ylabel('Events')
    ax_main.legend()
    ax_main.tick_params(labelbottom=False)

    for idx, (sample, counts, _errs) in enumerate(histograms):
        if idx == reference_index:
            continue
        ratio, ratio_err = make_count_ratio(counts, ref_counts)
        finite = np.isfinite(ratio)
        marker_style = {}
        if open_markers:
            marker_style = {
                "markerfacecolor": "none",
                "markeredgecolor": sample.get("color", None),
                "markeredgewidth": 1.2,
            }
        ax_ratio.errorbar(centers[finite], ratio[finite], yerr=ratio_err[finite],
                          fmt='o', ms=4, label=sample["label"],
                          color=sample.get("color", None), **marker_style)

    ax_ratio.axhline(1, lw=0.8, linestyle='--', color='gray')
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.set_ylim(*ratio_ylim)

    fig.subplots_adjust(hspace=0.1, left=0.15, right=0.97, bottom=0.12, top=0.96)
    plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_hist_with_ratio_weighted(arr_d, arr_o, bins, xlabel, outname, legend="", w_d=None, w_o=None):
    if w_d is None:
        w_d = np.ones_like(arr_d, dtype=float)
    if w_o is None:
        w_o = np.ones_like(arr_o, dtype=float)

    counts_d, err_d, edges = weighted_hist(arr_d, bins=bins, weights=w_d)
    counts_o, err_o, _     = weighted_hist(arr_o, bins=edges, weights=w_o)
    centers = 0.5 * (edges[:-1] + edges[1:])

    tol = 1e-6
    for i, (x, y) in enumerate(zip(arr_d, arr_o)):
        diff = x - y
        if abs(diff) < tol:
            diff = 0.0
        if diff > 0:
            print(f"ARRAYS DIFFER - {i:4d}: d = {x},  o = {y},  diff = {diff}")
    

    # Ratio with safe mask (denom!=0)
    ratio = np.full_like(counts_d, np.nan, dtype=float)
    ratio_err = np.full_like(counts_d, np.nan, dtype=float)
    mask = counts_o != 0.0

    ratio[mask] = counts_d[mask] / counts_o[mask]

    # Propagate using weighted errors: dr/r = sqrt((σd/d)^2 + (σo/o)^2)
    safe_d = np.where(counts_d != 0.0, counts_d, 1.0)
    ratio_err[mask] = ratio[mask] * np.sqrt((err_d[mask] / safe_d[mask])**2 + (err_o[mask] / counts_o[mask])**2)

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    ax_main.errorbar(centers, counts_d, yerr=err_d, fmt='o', label='onshell')
    ax_main.step(edges, np.append(counts_o, counts_o[-1]), where='post',
                 label='onshell_v1' if not legend else legend)
    ax_main.set_ylabel('Events (weighted)')
    ax_main.legend()
    ax_main.tick_params(labelbottom=False)

    finite = np.isfinite(ratio)
    ax_ratio.errorbar(centers[finite], ratio[finite], yerr=ratio_err[finite], fmt='o')
    ax_ratio.axhline(1, linestyle='--', color='C1')
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.set_ylim(0.65, 1.35)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot(pid, lhe_path_d, lhe_path_o, name, legend=''):

    theta_d, phi_d, pt_d, dphi_d, w_d = read_wp_momenta(lhe_path_d, pid)
    theta_o, phi_o, pt_o, dphi_o, w_o = read_wp_momenta(lhe_path_o, pid)

    plot_theta = False

    plot_hist_with_ratio_weighted(
        dphi_d if not plot_theta else theta_d, 
        dphi_o if not plot_theta else theta_o, 
        bins=32, 
        xlabel=r'$\Delta\phi$' if not plot_theta else r'$\theta$', 
        outname=f'dphi_{name}.pdf' if not plot_theta else f'theta_{name}.pdf',
        legend=legend,
        w_d=w_d,
        w_o=w_o
    )


def plot_wb_mass(samples, bins=60, outname='wb_mass.pdf', xlabel=r'$m_{Wb}$ [GeV]',
                 include_antitop=False, ratio_ref='wwbb', mass_range=(150, 200),
                 ratio_uncertainty=None, observable="mass",
                 ratio_ylim=(0.5, 1.5), mass_cut=None):
    """
    Plot a weighted W+b observable for one or more tagged samples.
    samples: list of dicts with keys:
      - path: path to LHE file
      - label: legend label
      - color: optional matplotlib color
      - key: optional short identifier used for ratio selections
    ratio_uncertainty: optional pair of sample keys/labels. If provided, the
      absolute bin-by-bin difference of those two histograms is shown as a
      band around one in the ratio panel, scaled by the reference histogram.
    """
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.0)
    ax = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax)

    all_values = []
    all_weights = []
    hists = []
    edges = np.linspace(mass_range[0], mass_range[1], bins + 1)
    ref_idx = None

    for sample in samples:
        values, weights = read_wb_observable(
            sample["path"], include_antitop=include_antitop,
            observable=observable, mass_cut=mass_cut
        )
        if len(values) == 0:
            continue
        all_values.extend(values)
        all_weights.extend(weights)
        counts, errs, edges = weighted_hist(values, bins=edges, weights=weights)
        centers = 0.5 * (edges[:-1] + edges[1:])
        color = sample.get("color", None)
        ax.errorbar(centers, counts, yerr=errs,
                    fmt='o', ms=4, label=sample["label"], color=color, alpha=0.7)
        ax.step(edges, np.append(counts, counts[-1]),
                where='post', color=color, alpha=0.7)
        hists.append((sample, counts, errs))

    if all_values:
        ref_idx = find_histogram_index(hists, ratio_ref)
        if ref_idx is None:
            for i in range(len(hists)):
                if any(tag in hists[i][0]["label"].lower() for tag in ("wwbb", "wbwb")):
                    ref_idx = i
                    break
        ax.set_xlim(*mass_range)
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
        ax.set_ylabel(r'Events ($\mathcal{L}=1.2$ fb$^{-1}$)', fontsize=14, labelpad=10)
        ax.legend(fontsize=11, borderpad=0.3, labelspacing=0.3, handletextpad=0.4)
        ax.tick_params(axis='both', labelsize=12, labelbottom=False)

        if ref_idx is not None:
            ref_counts = hists[ref_idx][1]
            ref_errs = hists[ref_idx][2]
            ref_color = hists[ref_idx][0].get("color", "orange")
            centers = 0.5 * (edges[:-1] + edges[1:])
            if ratio_uncertainty is not None:
                band_a_idx = find_histogram_index(hists, ratio_uncertainty[0])
                band_b_idx = find_histogram_index(hists, ratio_uncertainty[1])
                if band_a_idx is None or band_b_idx is None:
                    raise ValueError(
                        "Could not find both samples for the ratio uncertainty band: "
                        f"{ratio_uncertainty}"
                    )
                band_a_counts = hists[band_a_idx][1]
                band_b_counts = hists[band_b_idx][1]
                band_half_width = np.full_like(ref_counts, np.nan, dtype=float)
                # For a WWbb reference, the uncertainty is defined by the
                # MadSpin--PA envelope relative to MadSpin, bin by bin.
                band_denominator = (
                    band_a_counts if ratio_ref == "wwbb" else ref_counts
                )
                band_mask = band_denominator != 0
                band_half_width[band_mask] = (
                    np.abs(band_a_counts[band_mask] - band_b_counts[band_mask])
                    / np.abs(band_denominator[band_mask])
                )
                band_low = 1.0 - band_half_width
                band_high = 1.0 + band_half_width
                inner_band_low = 1.0 - 0.5 * band_half_width
                inner_band_high = 1.0 + 0.5 * band_half_width
                band_finite = np.isfinite(band_low) & np.isfinite(band_high)
                for i in np.where(band_finite)[0]:
                    ax_ratio.fill_between(
                        edges[i:i + 2],
                        [band_low[i], band_low[i]],
                        [band_high[i], band_high[i]],
                        color='#74c476',
                        alpha=0.22,
                        linewidth=0,
                        zorder=0,
                    )
                    ax_ratio.fill_between(
                        edges[i:i + 2],
                        [inner_band_low[i], inner_band_low[i]],
                        [inner_band_high[i], inner_band_high[i]],
                        color='#238b45',
                        alpha=0.24,
                        linewidth=0,
                        zorder=0.5,
                    )
            for sample, counts, errs in hists:
                if sample is hists[ref_idx][0]:
                    continue
                ratio = np.full_like(counts, np.nan, dtype=float)
                ratio_err = np.full_like(counts, np.nan, dtype=float)
                mask = ref_counts != 0
                ratio[mask] = counts[mask] / ref_counts[mask]
                count_rel_err = np.zeros_like(counts, dtype=float)
                count_nonzero = counts != 0
                count_rel_err[count_nonzero] = errs[count_nonzero] / counts[count_nonzero]
                ref_rel_err = np.zeros_like(ref_counts, dtype=float)
                ref_rel_err[mask] = ref_errs[mask] / ref_counts[mask]
                ratio_err[mask] = ratio[mask] * np.sqrt(
                    count_rel_err[mask] ** 2 + ref_rel_err[mask] ** 2
                )
                finite = np.isfinite(ratio)
                ax_ratio.errorbar(centers[finite], ratio[finite], yerr=ratio_err[finite], fmt='o', ms=4,
                                  label=sample["label"], color=sample.get("color", None), alpha=0.7,
                                  zorder=2)
            ax_ratio.axhline(1.0, linestyle='--', color=ref_color, zorder=1)
            ax_ratio.set_ylabel('Ratio', fontsize=14, labelpad=10)
            ax_ratio.set_ylim(*ratio_ylim)
            # ratio panel uses curve styles only; keep legend off for a cleaner view

        ax_ratio.set_xlabel(xlabel, fontsize=14, labelpad=10)
        ax_ratio.tick_params(axis='both', labelsize=12)
        if ref_idx is None:
            ax_ratio.set_visible(False)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.0)
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_pp_jz_had(pid=1, observable='dphi'):
    base = 'pp_jz/Events/run_02'
    samples = [
        {
            "path": f'{base}/pp_jz_had_onshell_first.lhe',
            "label": "onshell_v1 first",
            "color": "C0",
        },
        {
            "path": f'{base}/pp_jz_had_onshell_average.lhe',
            "label": "onshell_v1 average",
            "color": "C1",
        },
        {
            "path": f'{base}/pp_jz_had_density.lhe',
            "label": "onshell",
            "color": "C2",
        },
    ]

    for sample in samples:
        theta, _phi, _pt, dphi, _weight = read_wp_momenta(
            sample["path"], pid=pid, final_state_only=True
        )
        sample["values"] = theta if observable == 'theta' else dphi

    labels = {
        "theta": (r'$\theta$', 'theta_pp_jz_had.pdf'),
        "dphi": (r'$\Delta\phi$', 'dphi_pp_jz_had.pdf'),
    }
    xlabel, outname = labels[observable]

    plot_hist_with_ratio_multi(
        samples=samples,
        bins=32,
        xlabel=xlabel,
        outname=outname,
        reference_index=0,
        ratio_ylim=(0.85, 1.15),
        open_markers=True,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mass", action="store_true",
                      help="Plot predefined Wb observable comparison")
    mode.add_argument("--jz-had", action="store_true",
                      help="Plot pp_jz hadronic comparison with onshell_first as reference")
    parser.add_argument("--include-antitop", action="store_true",
                        help="Include W- bbar combinations in the Wb observable plot")
    parser.add_argument("--mass-reference", choices=("wwbb", "madspin", "pa", "both"), default="wwbb",
                        help="Ratio reference for --mass: WWbb, MadSpin, PA, or the existing WWbb+PA pair")
    parser.add_argument("--wb-observable", choices=(
                            "mass", "wb-pt", "top-pt", "b-pt", "b-pstar", "wb-dr",
                            "bb-dphi", "bb-dr",
                            "top-pt-offshell", "top-pt-onshell",
                            "b-pstar-offshell", "b-pstar-onshell",
                            "b-pt-offshell", "bb-dphi-offshell",
                            "wb-dr-offshell", "wb-dr-onshell",
                            "bb-dr-offshell", "bb-dr-onshell", "all"
                        ), default="mass",
                        help="Observable used with --mass (default: mass)")
    parser.add_argument("--pid", type=int, default=None,
                        help="PDG id used for angular plots (default: 5, or 1 with --jz-had)")
    parser.add_argument("--observable", choices=("theta", "dphi"), default="dphi",
                        help="Observable used with --jz-had (default: dphi)")
    args = parser.parse_args()

    if args.mass:
        samples = [
            {"path": [
                "pp_tt_testSmearing_DCA_MEmodes/Events/run_01/unweighted_events_decayed_madspin.lhe"
            ], "label": r"$t\bar{t}$ (LO) madspin", "color": "C0", "key": "madspin"},
            {"path": [
                "pp_tt_testSmearing_DCA_MEmodes/Events/run_01/unweighted_events_decayed_madspin_v1.lhe"
            ], "label": r"$t\bar{t}$ (LO) madspin_v1", "color": "C1", "key": "madspin_v1"},    
            {"path": [
                "pp_tt_testSmearing_DCA_MEmodes/Events/run_01/unweighted_events_decayed_PA_jac.lhe"
            ], "label": r"$t\bar{t}$ (LO) PA", "color": "C2", "key": "pa"},
            {"path": [
                #"pp_tt_testSmearing_full/Events/run_01/unweighted_events.lhe.gz"
                "pp_tt_testSmearing_full/Events/run_01/unweighted_events.lhe.gz"
            ], "label": "WWbb LO", "color": "C3", "key": "wwbb"}
        ]
        off_shell_wb_cut = {"center": 173.0, "width": 7.5, "mode": "outside"}
        on_shell_wb_cut = {"center": 173.0, "width": 7.5, "mode": "inside"}
        off_shell_bins = 8
        observable_settings = {
            "mass": {
                "xlabel": r'$m_{Wb}$ [GeV]',
                "range": (150, 200),
                "ratio_ylim": (0.5, 1.5),
                "outname": "wb_mass_compare",
                "uncertainty_band": True,
            },
            "wb-pt": {
                "xlabel": r'$p_T(Wb)$ [GeV]',
                "range": (0, 300),
                "ratio_ylim": (0.85, 1.15),
                "outname": "wb_pt_compare",
                "uncertainty_band": True,
            },
            "top-pt": {
                "observable": "top-pt",
                "xlabel": r'$p_T(t_{\mathrm{reco}})$ [GeV]',
                "range": (0, 300),
                "ratio_ylim": (0.85, 1.15),
                "outname": "top_pt_compare",
                "uncertainty_band": True,
            },
            "top-pt-offshell": {
                "observable": "top-pt",
                "xlabel": r'$p_T(t_{\mathrm{reco}})$ [GeV]',
                "range": (0, 300),
                "bins": off_shell_bins,
                "ratio_ylim": (0.75, 1.25),
                "outname": "top_pt_offshell_compare",
                "uncertainty_band": True,
                "mass_cut": off_shell_wb_cut,
            },
            "top-pt-onshell": {
                "observable": "top-pt",
                "xlabel": r'$p_T(t_{\mathrm{reco}})$ [GeV]',
                "range": (0, 300),
                "ratio_ylim": (0.85, 1.15),
                "outname": "top_pt_onshell_compare",
                "uncertainty_band": True,
                "mass_cut": on_shell_wb_cut,
            },
            "b-pt": {
                "xlabel": r'$p_T(b)$ [GeV]',
                "range": (0, 300),
                "ratio_ylim": (0.85, 1.15),
                "outname": "b_pt_compare",
                "uncertainty_band": True,
            },
            "b-pt-offshell": {
                "observable": "b-pt",
                "xlabel": r'$p_T(b)$ [GeV]',
                "range": (0, 300),
                "bins": 10,
                "ratio_ylim": (0.75, 1.25),
                "outname": "b_pt_offshell_compare",
                "uncertainty_band": True,
                "mass_cut": off_shell_wb_cut,
            },
            "b-pstar": {
                "observable": "b-pstar",
                "xlabel": r'$p_b^*$ in $Wb$ rest frame [GeV]',
                "range": (40, 100),
                "ratio_ylim": (0.5, 1.5),
                "outname": "b_pstar_compare",
                "uncertainty_band": True,
            },
            "b-pstar-offshell": {
                "observable": "b-pstar",
                "xlabel": r'$p_b^*$ in $Wb$ rest frame [GeV]',
                "range": (40, 100),
                "bins": off_shell_bins,
                "ratio_ylim": (0.5, 1.5),
                "outname": "b_pstar_offshell_compare",
                "uncertainty_band": True,
                "mass_cut": off_shell_wb_cut,
            },
            "b-pstar-onshell": {
                "observable": "b-pstar",
                "xlabel": r'$p_b^*$ in $Wb$ rest frame [GeV]',
                "range": (40, 100),
                "ratio_ylim": (0.5, 1.5),
                "outname": "b_pstar_onshell_compare",
                "uncertainty_band": True,
                "mass_cut": on_shell_wb_cut,
            },
            "wb-dr": {
                "xlabel": r'$\Delta R(W,b)$',
                "range": (0, 6),
                "ratio_ylim": (0.5, 1.5),
                "outname": "wb_dr_compare",
                "uncertainty_band": True,
            },
            "wb-dr-offshell": {
                "observable": "wb-dr",
                "xlabel": r'$\Delta R(W,b)$',
                "range": (0, 6),
                "bins": off_shell_bins,
                "ratio_ylim": (0.5, 1.5),
                "outname": "wb_dr_offshell_compare",
                "uncertainty_band": True,
                "mass_cut": off_shell_wb_cut,
            },
            "wb-dr-onshell": {
                "observable": "wb-dr",
                "xlabel": r'$\Delta R(W,b)$',
                "range": (0, 6),
                "ratio_ylim": (0.5, 1.5),
                "outname": "wb_dr_onshell_compare",
                "uncertainty_band": True,
                "mass_cut": on_shell_wb_cut,
            },
            "bb-dphi": {
                "xlabel": r'$\Delta\phi(b,\bar{b})$',
                "range": (0, np.pi),
                "ratio_ylim": (0.85, 1.15),
                "outname": "bb_dphi_compare",
                "uncertainty_band": True,
            },
            "bb-dphi-offshell": {
                "observable": "bb-dphi",
                "xlabel": r'$\Delta\phi(b,\bar{b})$',
                "range": (0, np.pi),
                "bins": off_shell_bins,
                "ratio_ylim": (0.85, 1.15),
                "outname": "bb_dphi_offshell_compare",
                "uncertainty_band": True,
                "mass_cut": off_shell_wb_cut,
            },
            "bb-dr": {
                "xlabel": r'$\Delta R(b,\bar{b})$',
                "range": (0, 6),
                "ratio_ylim": (0.85, 1.15),
                "outname": "bb_dr_compare",
                "uncertainty_band": True,
            },
            "bb-dr-offshell": {
                "observable": "bb-dr",
                "xlabel": r'$\Delta R(b,\bar{b})$',
                "range": (0, 6),
                "bins": off_shell_bins,
                "ratio_ylim": (0.85, 1.15),
                "outname": "bb_dr_offshell_compare",
                "uncertainty_band": True,
                "mass_cut": off_shell_wb_cut,
            },
            "bb-dr-onshell": {
                "observable": "bb-dr",
                "xlabel": r'$\Delta R(b,\bar{b})$',
                "range": (0, 6),
                "ratio_ylim": (0.85, 1.15),
                "outname": "bb_dr_onshell_compare",
                "uncertainty_band": True,
                "mass_cut": on_shell_wb_cut,
            },
        }
        wb_observables = (
            tuple(observable_settings)
            if args.wb_observable == "all"
            else (args.wb_observable,)
        )
        mass_references = (
            ("wwbb", "pa") if args.mass_reference == "both"
            else (args.mass_reference,)
        )
        for wb_observable in wb_observables:
            settings = observable_settings[wb_observable]
            for mass_reference in mass_references:
                outname = settings["outname"]
                if mass_reference == "pa":
                    outname += "_pa_ref"
                elif mass_reference == "madspin":
                    outname += "_madspin_ref"
                plot_wb_mass(
                    samples=samples,
                    bins=settings.get("bins", 25),
                    outname=f"{outname}.pdf",
                    xlabel=settings["xlabel"],
                    include_antitop=args.include_antitop,
                    ratio_ref=mass_reference,
                    mass_range=settings["range"],
                    ratio_uncertainty=(
                        ("madspin", "pa")
                        if mass_reference in ("madspin", "pa")
                        or settings["uncertainty_band"]
                        else None
                    ),
                    observable=settings.get("observable", wb_observable),
                    ratio_ylim=settings["ratio_ylim"],
                    mass_cut=settings.get("mass_cut"),
                )
    elif args.jz_had:
        plot_pp_jz_had(pid=1 if args.pid is None else args.pid,
                       observable=args.observable)
    else:
        base = 'pp_ttj/Events/run_01'
        lhe_path_d = f'{base}/pp_tt_NLO_2topdecay_density.lhe'
        lhe_path_o = f'{base}/pp_tt_NLO_2topdecay_onshell.lhe'

        pid = 5 if args.pid is None else args.pid
        theta_d, phi_d, pt_d, dphi_d, w_d = read_wp_momenta(lhe_path_d, pid=pid)
        theta_o, phi_o, pt_o, dphi_o, w_o = read_wp_momenta(lhe_path_o, pid=pid)

        plot_hist_with_ratio(theta_d, theta_o, bins=32, xlabel=r'$\theta$', outname='theta_W.pdf')
        #plot_hist_with_ratio(phi_d, phi_o, bins=32, xlabel=r'$\phi$', outname='phi_W.png')
        #plot_hist_with_ratio(pt_d, pt_o, bins=100, xlabel=r'$p_{T}\;(\mathrm{GeV})$', outname='pt_W.pdf')
