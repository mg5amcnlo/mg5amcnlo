#!/usr/bin/env python3
"""Render the saved FxFx investigation and export its bin values to CSV."""

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = np.load(HERE / 'djr_data.npz')
META = json.loads((HERE / 'samples.json').read_text())
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.2,
                     'legend.frameon': False, 'savefig.dpi': 160})


def panel(title):
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(8.5, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(title, fontsize=13)
    ax.set_ylabel(r'$d\sigma/d\log_{10}(d_{1\to0}/{\rm GeV})$ [pb]')
    ratio.set_ylabel('Ratio')
    ratio.set_xlabel(r'$d_{1\to0}$ [GeV], $k_T$ with $R=1$')
    ratio.axhline(1, color='0.4', lw=1)
    ratio.set_xscale('log')
    ratio.set_xlim(8, 160)
    ax.set_yscale('log')
    ax.set_ylim(4e2, 1.3e4)
    ratio.set_xticks([10, 20, 30, 40, 60, 100, 160], labels=['10', '20', '30', '40', '60', '100', '160'])
    return fig, ax, ratio


def line(ax, ratio, key, label, color, ref, ls='-', band=True):
    a = DATA[key]
    x = 10 ** ((a[:, 0] + a[:, 1]) / 2)
    y = a[:, 2] / (a[:, 1] - a[:, 0])
    e = a[:, 3] / (a[:, 1] - a[:, 0])
    keep = (x >= 8) & (x <= 160) & (y > 0)
    ax.plot(x[keep], y[keep], label=label, color=color, ls=ls, lw=1.6)
    if band:
        ax.fill_between(x[keep], np.maximum(1e-6, (y-e)[keep]), (y+e)[keep], color=color, alpha=0.13)
    denom = DATA[ref][:, 2]
    keep &= denom > 0
    r = np.divide(a[:, 2], denom, out=np.full_like(denom, np.nan), where=denom > 0)
    re = np.divide(a[:, 3], denom, out=np.full_like(denom, np.nan), where=denom > 0)
    ratio.plot(x[keep], r[keep], color=color, ls=ls, lw=1.5)
    if band:
        ratio.fill_between(x[keep], (r-re)[keep], (r+re)[keep], color=color, alpha=0.13)


def finish(fig, ax, ratio, pdf, filename, note):
    ax.legend(loc='lower left', fontsize=9)
    fig.text(0.12, 0.015, note, fontsize=8, va='bottom')
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    pdf.savefig(fig)
    fig.savefig(HERE / filename)
    plt.close(fig)


with PdfPages(HERE / 'fxfx_investigation.pdf') as pdf:
    fig, ax, ratio = panel('W+1j: the fixed-order FxFx reference uses the wrong counterevent scales')
    ref = 'w1j_fixed-order'
    for key, label, color, ls in [
        (ref, 'NLO, ickkw=0 (original)', '0.35', ':'),
        ('w1j_fixed-order-ckkw', 'NLO, ickkw=3 (original)', '#d55e00', '--'),
        ('w1j_fo_pilot_restored', 'NLO, ickkw=3 (restored Born scales)', '#0072b2', '-'),
        ('w1j_showered-ckkw_LHE', 'MC@NLO, ickkw=3 (before shower)', '#009e73', '-'),
        ('w1j_showered-ckkw', 'FxFx npNLO=1, Qcut=20 GeV (after shower)', '#cc79a7', '-'),
    ]:
        line(ax, ratio, key, label, color, ref, ls)
    ratio.set_ylabel('Ratio to\nNLO ickkw=0')
    ratio.set_ylim(0.4, 1.4)
    finish(fig, ax, ratio, pdf, 'fixed_order_reference.png',
           'Bands: statistical errors on each numerator. Corrected fixed order: all six subprocesses,\n'
           '2 x 65536 integration points per subprocess. Original LHE and shower samples: 500000 events.')

    fig, ax, ratio = panel('FxFx W+0,1j: residual merging-scale dependence')
    for q, color in [(20, '#0072b2'), (30, '#d55e00'), (40, '#009e73')]:
        line(ax, ratio, f'merged_scan_q{q}', f'Qcut={q} GeV', color, 'merged_scan_q20')
        ax.axvline(q, color=color, lw=0.8, alpha=0.5)
    ratio.set_ylabel('Ratio to\nQcut=20 GeV')
    ratio.set_ylim(0.65, 1.2)
    finish(fig, ax, ratio, pdf, 'merging_scale_scan.png',
           'Same FxFx LHE inputs, first 100000 events per multiplicity; Pythia 8.318, random seed 12345.\n'
           'Generation cut ptj=8 GeV. Ratio bands show numerator errors; scan correlations are not estimated.')

    fig, ax, ratio = panel('W+1j FxFx input: the agreement with an unmerged shower is already present')
    ref = 'w1j_showered-ckkw_LHE'
    for key, label, color, ls in [
        (ref, 'ickkw=3, before shower (500000 events)', '0.35', ':'),
        ('w1j_scan_no_merging', 'Same ickkw=3 input, shower with FxFx hooks off', '#0072b2', '-'),
        ('w1j_scan_q20', 'Same ickkw=3 input, FxFx Qcut=20 GeV', '#d55e00', '-'),
        ('w1j_showered', 'ickkw=0, ordinary shower (500000 events)', '#009e73', '--'),
    ]:
        line(ax, ratio, key, label, color, ref, ls)
    ax.axvline(20, color='0.5', lw=0.8)
    ratio.set_ylabel('Ratio to\nFxFx LHE')
    ratio.set_ylim(0.5, 1.35)
    finish(fig, ax, ratio, pdf, 'before_after_shower.png',
           'Reshower controls use 100000 events and random seed 12345; all other shower settings are unchanged.\n'
           'The merging veto suppresses npNLO=1 below Qcut; it does not explain the low old fixed-order curve.')

    fig, ax, ratio = panel('Near the merging boundary, the two multiplicities differ by the first-jet NLO correction')
    ref = 'w0j_fixed-order-ckkw'
    for key, label, color, ls in [
        (ref, 'NLO W+0j, first resolved jet (LO accuracy)', '0.35', ':'),
        ('w1j_LO_ickkw3', 'Born W+1j with FxFx weights', '#d55e00', '--'),
        ('w1j_fo_pilot_restored', 'NLO W+1j, restored FxFx scales', '#0072b2', '-'),
        ('merged_original_q20', 'Merged W+0,1j, Qcut=20 GeV', '#009e73', '-'),
    ]:
        line(ax, ratio, key, label, color, ref, ls)
    ax.axvline(20, color='0.5', lw=0.8)
    ratio.set_ylabel('Ratio to\nNLO W+0j')
    ratio.set_ylim(0.6, 2)
    finish(fig, ax, ratio, pdf, 'multiplicity_transition.png',
           'Correcting the fixed-order reference explains its discrepancy with the one-jet MC@NLO input.\n'
           'A sizable NLO correction to the first resolved jet remains at scales near the merging boundary.')

with (HERE / 'djr_bins_pb.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['sample', 'log10_djr_min', 'log10_djr_max', 'djr_min_GeV', 'djr_max_GeV', 'bin_pb', 'stat_pb'])
    for key in DATA.files:
        for lo, hi, value, error in DATA[key]:
            writer.writerow([key, lo, hi, 10**lo, 10**hi, value, error])
with (HERE / 'rates_pb.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['sample', 'cross_section_pb', 'stat_pb'])
    for key, (value, error) in META['rates_pb'].items():
        writer.writerow([key, value, error])

print(HERE / 'fxfx_investigation.pdf')
