"""t-channel single-top production at NLO+PS: the four MC@NLO predictions of the paper, with
global recoil (the MG5aMC default, the reference) and with local recoil (stock Pythia8, and
Pythia8 with the r_q and r_q r_t kernel corrections).

Reads the corrected C_GLB/C_LOC/C_Q/C_QT exports in ../../central_runs_fixed
(bin contents are cross sections per bin in pb, with their MC statistical uncertainties; for each
observable the selection, ALL, TAG or GAP, is given by the TYPE@ tag) and writes the figures of the
paper in this directory. Main panel: the four predictions, differential in the observable (or the
gap-veto efficiency); lower panel: ratios to the global-recoil prediction. Statistical
uncertainties: a band around each curve, and the uncertainty of the reference as a gray band
around one in the ratio panel.
Style of mc_counterterm_singletop/make_plots.py of the MadGraph7 paper; PDFs go through the pgf
backend (the usetex PDF writer drops the math minus on this machine).

usage: python3 make_plots.py [--central-dir ../../central_runs_fixed]
"""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.font_manager as font_manager
from matplotlib.ticker import AutoMinorLocator, FixedLocator, MultipleLocator
mpl.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})
mpl.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False})
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Analyses.matched_run_inputs import CENTRAL_RUNS, find_input
font_legend = font_manager.FontProperties(size=12)
lw = 1.2

REF = 'default'
SAMPLES = ['default', 'local', 'local_patch', 'local_2patch']
LABELS = {'default': 'global', 'local': 'local',
          'local_patch': r'local, $\times\, r_q$', 'local_2patch': r'local, $\times\, r_q r_t$'}
COLOURS = {'default': 'black', 'local': '#2a5db0', 'local_patch': '#1e9e5a', 'local_2patch': '#c40524'}
STYLES = {'default': 'solid', 'local': 'solid', 'local_patch': 'dashed', 'local_2patch': 'solid'}
RATIO_LABEL = r'ratio to global'
SELECTION = {'ALL': 'all events', 'TAG': 'TAG', 'GAP': 'GAP'}

# obs and sel select the histogram; nrebin bins are merged into one; the main panel shows the
# cross section per unit of the observable (per bin when perbin), or the efficiency (veto);
# legend: (loc, bbox_to_anchor); label: (x, y) of the selection label in axes coordinates
PLOTS = [
    dict(obs='top pT [GeV]', sel='ALL', out='ptt_all.pdf', xlabel=r'$p_T(t)$ [GeV]',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}p_T(t)$ [pb/GeV]', xlim=(0., 400.), logy=True,
         nrebin=2, rlim=(0.85, 1.2), rticks=[0.9, 1.0, 1.1], legend=('upper right', (0.99, 0.98)),
         label=(0.05, 0.06)),
    dict(obs='j1 y', sel='TAG', out='yj1_tag.pdf', xlabel=r'$y(j_1)$',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}y(j_1)$ [pb]', xlim=(-4.5, 4.5), logy=False,
         nrebin=2, ymax=21., rlim=(0.85, 1.2), rticks=[0.9, 1.0, 1.1],
         legend=('upper center', (0.5, 0.99)), label=(0.05, 0.06)),
    dict(obs='Njets20 (last >=8)', sel='TAG', out='njets_tag.pdf', xlabel=r'$N_{\mathrm{jets}}$',
         ylabel=r'$\sigma$ per bin [pb]', xlim=(0.5, 6.5), logy=True, perbin=True,
         rlim=(0.15, 1.25), rticks=[0.25, 0.5, 0.75, 1.0], xticks=[1, 2, 3, 4, 5, 6],
         xticklabels=['1', '2', '3', '4', '5', r'$\geq 6$'], last=6,
         legend=('upper right', (0.99, 0.98)), label=(0.05, 0.06)),
    dict(obs='Ngap jets20 (last >=5)', sel='GAP', out='ngap_gap.pdf', xlabel=r'$N_{\mathrm{gap}}$',
         ylabel=r'$\sigma$ per bin [pb]', xlim=(-0.5, 4.5), logy=True, perbin=True,
         rlim=(0.15, 1.25), rticks=[0.25, 0.5, 0.75, 1.0], xticks=[0, 1, 2, 3, 4],
         xticklabels=['0', '1', '2', '3', r'$\geq 4$'], last=4,
         legend=('upper right', (0.99, 0.98)), label=(0.05, 0.06)),
    dict(obs='dphi(t,j1) [rad]', sel='TAG', out='dphi_tag.pdf', xlabel=r'$\Delta\phi(t,j_1)$',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}\Delta\phi(t,j_1)$ [pb]', xlim=(0., np.pi),
         logy=True, nrebin=2, rlim=(0.45, 1.25), rticks=[0.6, 0.8, 1.0, 1.2],
         legend=('upper left', (0.01, 0.98)), label=(0.05, 0.42)),
    dict(obs='pT(t+j1) low [GeV]', sel='GAP', out='ptsys_gap.pdf', xlabel=r'$p_T(t+j_1)$ [GeV]',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}p_T(t+j_1)$ [pb/GeV]', xlim=(0., 50.), logy=False,
         nrebin=2, ymax=1.05, rlim=(0.5, 2.6), rticks=[1.0, 1.5, 2.0, 2.5],
         legend=('upper right', (0.99, 0.98)), label=(0.05, 0.86)),
    dict(obs='j2 zstar (dy>1)', sel='TAG', out='zstar_tag.pdf', xlabel=r'$z^*_{j_2}$',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}z^*_{j_2}$ [pb]', xlim=(-2., 2.), logy=False,
         nrebin=1, ymax=26., rlim=(0.6, 1.4), rticks=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
         legend=('upper right', (0.99, 0.98)), label=(0.05, 0.86), vlines=[-0.5, 0.5]),
    dict(obs='j2 zstar (dy>1)', sel='GAP', out='zstar_gap.pdf', xlabel=r'$z^*_{j_2}$',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}z^*_{j_2}$ [pb]', xlim=(-1.5, 1.2), logy=False,
         nrebin=1, ymax=10.5, rlim=(0.4, 1.6), rticks=[0.6, 0.8, 1.0, 1.2, 1.4],
         legend=('upper right', (0.99, 0.98)), label=(0.05, 0.86), vlines=[-0.5, 0.5]),
    dict(obs='j2 dy from top (oriented)', sel='GAP', out='dyj2t_gap.pdf',
         xlabel=r'$s\,[y(j_2)-y(t)]$',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}(s\Delta y)$ [pb]', xlim=(-4., 6.), logy=False,
         nrebin=2, ymax=2.45, rlim=(0.4, 1.6), rticks=[0.6, 0.8, 1.0, 1.2, 1.4],
         legend=('upper right', (0.99, 0.98)), label=(0.05, 0.86)),
    dict(obs='j2 dy from j1 (oriented)', sel='GAP', out='dyj2j1_gap.pdf',
         xlabel=r'$s\,[y(j_2)-y(j_1)]$',
         ylabel=r'$\mathrm{d}\sigma/\mathrm{d}(s\Delta y)$ [pb]', xlim=(-6., 2.), logy=False,
         nrebin=2, ymax=2.45, rlim=(0.4, 1.6), rticks=[0.6, 0.8, 1.0, 1.2, 1.4],
         legend=('upper right', (0.99, 0.98)), label=(0.05, 0.86)),
    dict(obs='veto', sel='TAG', out='veto_tag.pdf', xlabel=r'$Q_0$ [GeV]',
         ylabel=r'gap fraction $f_{\mathrm{gap}}(Q_0)$', xlim=(20., 100.), logy=False, veto=True,
         ymin=0.75, ymax=1.01, rlim=(0.99, 1.11), rticks=[1.0, 1.02, 1.04, 1.06, 1.08, 1.1],
         legend=('lower right', (0.99, 0.03)), label=(0.05, 0.86)),
    dict(obs='veto', sel='GAP', out='veto_gap.pdf', xlabel=r'$Q_0$ [GeV]',
         ylabel=r'gap fraction $f_{\mathrm{gap}}(Q_0)$', xlim=(20., 100.), logy=False, veto=True,
         ymin=0.75, ymax=1.01, rlim=(0.99, 1.11), rticks=[1.0, 1.02, 1.04, 1.06, 1.08, 1.1],
         legend=('lower right', (0.99, 0.03)), label=(0.05, 0.86)),
]


def read_hwu(fname):
    """{(observable, type): (edges, y, dy)} for all the histograms in the HwU file fname"""
    hists = {}
    for block in open(fname).read().split('<histogram>')[1:]:
        header, body = block.split('\n', 1)
        name = header.split('"')[1]
        rows = np.array([[float(v) for v in l.split()] for l in body.split('<\\histogram>')[0].split('\n') if l.strip()])
        hists[(name.split(' |')[0].strip(), name.split('TYPE@')[-1])] = (
            np.append(rows[:, 0], rows[-1, 1]), rows[:, 2], rows[:, 3])
    return hists


def rebin(h, nrebin):
    """nrebin bins merged into one; the uncertainties are added in quadrature"""
    edges, y, dy = h
    if len(y) % nrebin:
        raise Exception('cannot rebin %d bins by %d' % (len(y), nrebin))
    return (edges[::nrebin], y.reshape(-1, nrebin).sum(axis=1),
            np.sqrt((dy**2).reshape(-1, nrebin).sum(axis=1)))


def veto_fraction(hists, sel):
    """the gap fraction, survived/reference at each Q0, the numerator being a subset of the
    denominator: for Poisson-type (sum of squared weights) uncertainties,
    Cov(N, D) = Var(N), so that Var(N/D) = [(1-f)^2 Var(N) + f^2 (Var(D) - Var(N))]/D^2"""
    edges, n, dn = hists[('gap veto survived Q0 [GeV]', sel)]
    _, d, dd = hists[('gap veto reference Q0 [GeV]', sel)]
    f = n / d
    err = np.sqrt(((1. - f)**2 * dn**2 + f**2 * np.maximum(dd**2 - dn**2, 0.)) / d**2)
    return edges, f, err


def get(hists, cfg):
    """edges, values and uncertainties shown in the main panel"""
    if cfg.get('veto'):
        return veto_fraction(hists, cfg['sel'])
    edges, y, dy = rebin(hists[(cfg['obs'], cfg['sel'])], cfg.get('nrebin', 1))
    if 'last' in cfg:
        # the multiplicity histograms: bins above the last shown one are added to it
        i = np.searchsorted(edges, cfg['last'] - 0.5)
        y = np.append(y[:i], y[i:].sum())
        dy = np.append(dy[:i], np.sqrt((dy[i:]**2).sum()))
        edges = edges[:i + 2]
    if not cfg.get('perbin'):
        width = np.diff(edges)
        y, dy = y / width, dy / width
    return edges, y, dy


def ratio(num, den):
    """num/den, not defined (nan) where the denominator is not positive"""
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def band(ax, edges, lo, hi, colour, alpha=0.15):
    """a band from lo to hi in each bin; bins where lo or hi is nan are skipped"""
    ax.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(lo, 2), np.repeat(hi, 2),
                    color=colour, alpha=alpha, linewidth=0)


def update_legend(ax, loc, pos, font=font_legend, handlelength=2.):
    """lines, not squares, in the legend"""
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if 'fill' in dir(h):
            new_handles.append(mlines.Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle(),
                                            linewidth=lw))
        else:
            new_handles.append(h)
    ax.legend(handles=new_handles, labels=labels, prop=font, frameon=False, ncol=1,
              loc=loc, bbox_to_anchor=pos, handlelength=handlelength)


def make(hists, cfg, output):
    hs = dict((s, get(hists[s], cfg)) for s in SAMPLES)
    edges, ref, ref_err = hs[REF]
    shown = (edges[:-1] >= cfg['xlim'][0] - 1e-9) & (edges[1:] <= cfg['xlim'][1] + 1e-9)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
    fig.set_size_inches(7 * 0.75, 5.6)
    fig.subplots_adjust(hspace=0.06)
    for ax in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True)

    ax = axes[0]
    ax.set_ylabel(cfg['ylabel'])
    ymax, ymin = 0., np.inf
    for s in SAMPLES:
        _, y, dy = hs[s]
        ax.stairs(y, edges, color=COLOURS[s], linestyle=STYLES[s], linewidth=lw, label=LABELS[s],
                  baseline=None)
        # no band in empty bins (on the log scale)
        band(ax, edges, np.where(y > 0, y - dy, np.nan), np.where(y > 0, y + dy, np.nan), COLOURS[s])
        ymax = max(ymax, y[shown].max())
        ymin = min(ymin, y[shown & (y > 0)].min())
    if cfg['logy']:
        ax.set_yscale('log')
        ax.set_ylim(0.5 * ymin, 8. * ymax)
    else:
        ax.set_ylim(cfg.get('ymin', 0.), cfg.get('ymax', 1.5 * ymax))
    for v in cfg.get('vlines', []):
        for a in axes:
            a.axvline(v, color='gray', linewidth=0.6, linestyle='dotted')
    update_legend(ax, *cfg['legend'])
    ax.text(cfg['label'][0], cfg['label'][1], SELECTION[cfg['sel']], transform=ax.transAxes,
            fontsize=13)

    ax = axes[1]
    rel = ratio(ref_err, ref)
    ax.stairs(1. + rel, edges, baseline=1. - rel, fill=True, color='gray', alpha=0.3, linewidth=0)
    ax.axhline(1., color='black', linewidth=0.8)
    for s in SAMPLES[1:]:
        _, y, dy = hs[s]
        r, err = ratio(y, ref), ratio(dy, ref)
        # baseline=None: no vertical lines at the ends, bins where the ratio is nan are skipped
        ax.stairs(r, edges, baseline=None, color=COLOURS[s], linestyle=STYLES[s], linewidth=lw)
        band(ax, edges, r - err, r + err, COLOURS[s])
    ax.set_ylim(*cfg['rlim'])
    ax.yaxis.set_major_locator(FixedLocator(cfg['rticks']))
    ax.set_ylabel(RATIO_LABEL)
    fig.align_ylabels(axes)
    ax.set_xlim(*cfg['xlim'])
    if 'xticks' in cfg:
        ax.xaxis.set_major_locator(FixedLocator(cfg['xticks']))
        ax.set_xticklabels(cfg['xticklabels'])
        for a in axes:
            a.xaxis.set_minor_locator(FixedLocator([]))
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if cfg['obs'] == 'dphi(t,j1) [rad]':
        ax.xaxis.set_major_locator(FixedLocator([0., np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]))
        ax.set_xticklabels(['$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.set_xlabel(cfg['xlabel'])
    out = output / cfg['out']
    fig.savefig(out, backend='pgf', bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--central-dir', type=Path, default=ROOT / 'central_runs_fixed',
                        help='corrected C_GLB/C_LOC/C_Q/C_QT exports')
    parser.add_argument('--output-dir', type=Path, default=Path(HERE))
    args = parser.parse_args()
    try:
        hists = {s: read_hwu(find_input(args.central_dir / CENTRAL_RUNS[s])) for s in SAMPLES}
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cfg in PLOTS:
        print(make(hists, cfg, args.output_dir))


if __name__ == '__main__':
    main()
