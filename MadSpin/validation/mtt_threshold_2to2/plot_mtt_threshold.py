#!/usr/bin/env python3
"""``(1/sigma) dsigma/dm_tt`` near the ``2 m_t`` threshold for ``p p > t t~``
-- MadSpin's four spinmodes against the doubly-resonant off-shell MadGraph
truth, in the MG7 paper's plotting style.

The figure is the same figure as
``MadSpin/validation/mtt_threshold/plot_mtt_threshold.py`` draws for
``p p > t t~ j``, and it is drawn by that module's own code: this file imports
it and re-points three module globals (the process label and the two legend
label tables) at the ``2 -> 2`` process.  Nothing about the style, the binning,
the normalisation, the clipping or the arrows is re-decided here, so the two
figures cannot drift apart.

What is different is the physics, and it is the reason the study exists.

``Event.reshuffle_production`` hands every top-level final-state momentum to
RAMBO's ``mass_shuffle`` at fixed ``sqrt(shat)``.  With a recoiling jet in the
final state that rescales the jet as well as the tops, the ``t t~``
four-momentum is not preserved, and ``m_tt`` moves -- which is how the density
spinmodes populate ``m_tt < 2 m_t`` in the ``t t~ j`` study.  Take the jet away
and the two tops are the entire final state: ``m_tt = sqrt(shat)`` identically,
and ``sqrt(shat)`` is exactly what the reshuffle holds fixed.  So no spinmode
can move ``m_tt`` at all, and the sub-threshold region is structurally empty for
every one of them, not just for ``onshell``.

That is measured, not assumed.  ``run_mtt_threshold.py`` pairs each decayed
event with the production event it came from -- the pairing checked by
``max |Delta sqrt(shat)| = 0`` -- and the ``Delta m_tt`` moments in
``meta.json`` are what
``mtt_threshold/plot_mtt_threshold.py:preserves_mtt`` reads to decide which
modes count as structurally empty below ``2 m_t``.  If any mode had moved
``m_tt``, the figure would show its sub-threshold content.

Usage::

    python3 plot_mtt_threshold.py [--data DIR] [--out DIR]

``--check-minus`` (on by default) re-opens the written PDF and asserts a math
minus survived matplotlib's usetex Type1 subsetting.  It is discriminating on
THIS figure -- ``NO_MINUS_FIX=1`` makes it report ``False`` -- and must not be
pointed at ``plots_userstyle/``, which renders without usetex.
"""

import argparse
import importlib.util
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIBLING = os.path.join(os.path.dirname(_HERE), 'mtt_threshold',
                        'plot_mtt_threshold.py')


def _load_sibling():
    """Import the ``t t~ j`` figure module by path, under its own name.

    By path and not by ``sys.path``: this file has the same basename, and
    ``sys.path[0]`` is this directory when the script is run, so a plain
    ``import plot_mtt_threshold`` would find *this* module and re-execute it.
    """
    if not os.path.exists(_SIBLING):
        raise SystemExit('cannot find the t t~ j figure module at %s' % _SIBLING)
    spec = importlib.util.spec_from_file_location(
        'plot_mtt_threshold_ttj', _SIBLING)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['plot_mtt_threshold_ttj'] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load_sibling()

# Everything below is that module's, re-exported so this file reads as one
# script and so ``plot_mtt_threshold_userstyle.py`` can import from here.
Data = P.Data
ratio = P.ratio
structurally_empty = P.structurally_empty
preserves_mtt = P.preserves_mtt
offscale_arrows = P.offscale_arrows
agreement_threshold = P.agreement_threshold
shape_scale = P.shape_scale
anchor_scale = P.anchor_scale
check_minus = P.check_minus
make_figure = P.make_figure
MODES = P.MODES
REF = P.REF
RATIO_CLIP = P.RATIO_CLIP
AGREE_HI = P.AGREE_HI
ANCHOR = P.ANCHOR
USETEX = P.USETEX
MINUS_FIX = P.MINUS_FIX

# ------------------------------------------------------------------ the process
# Three globals, and they are the whole of the re-pointing.  ``make_figure``
# reads them at call time.
P.PROC_TEX = r'pp \to t\bar t'
P.PROC_PLAIN = r'pp \to t\bar t'
P.CURVES = [
    ('truth',      r'$pp \to t\bar t$, $t \to W^+ b$ (off shell)'),
    ('madspin',    r'MadSpin, \texttt{spinmode = madspin}'),
    ('PA',         r'MadSpin, \texttt{spinmode = PA}'),
    ('onshell',    r'MadSpin, \texttt{spinmode = onshell}'),
    ('madspin_v1', r'MadSpin, \texttt{spinmode = madspin\_v1} (legacy)'),
]
P.CURVES_PLAIN = {
    'truth':      r'$pp \to t\bar t$, $t \to W^+ b$ (off shell)',
    'madspin':    'MadSpin, spinmode = madspin',
    'PA':         'MadSpin, spinmode = PA',
    'onshell':    'MadSpin, spinmode = onshell',
    'madspin_v1': 'MadSpin, spinmode = madspin_v1 (legacy)',
}
CURVES = P.CURVES
CURVES_PLAIN = P.CURVES_PLAIN
PROC_TEX = P.PROC_TEX
PROC_PLAIN = P.PROC_PLAIN


# --------------------------------------------------------------------------
def normalisation_report(d, p):
    """The raw ``truth/mode`` rate ratios, and what accounts for them.

    The figure is a SHAPE comparison -- every curve divided by its own total
    cross section -- so this offset is divided out of it.  It is reported anyway,
    on the brief's instruction and for a specific reason: a change now in review
    makes MadSpin's reported cross section carry the Breit-Wigner truncation,
    which is exactly what this offset is.  This branch does NOT contain that
    change, so these are the pre-change numbers and are on record to be checked
    against the corrected code.
    """
    bw = float(d.meta['bwcutoff'])
    masses = d.meta.get('param_card_masses', {})
    mt = float(masses.get('MT', 173.0))
    gt = float(masses.get('WT', 1.4915))
    mb = float(masses.get('MB', 4.7))
    mw = P._sm_mw()

    p('')
    p('-- the RAW rate ratios, and what accounts for them -------------------')
    p('The figure never shows these: it divides every curve by its own total')
    p('cross section.  They are recorded because a change in review makes')
    p('MadSpin\'s reported sigma carry the Breit-Wigner truncation, and this')
    p('branch predates it -- so these are the numbers that change.')
    p('')
    p('   %-12s %14s %14s' % ('sample', 'sigma [pb]', 'truth/sample'))
    for key in [REF] + MODES:
        if key not in d.meta['runs']:
            continue
        p('   %-12s %14.5f %14s'
          % (key, d.sigma(key),
             '-' if key == REF else '%.5f' % (d.sigma(REF) / d.sigma(key))))
    if 'production' in d.meta['runs']:
        p('   %-12s %14.5f %14s   (undecayed; MadSpin normalises to'
          % ('production', d.sigma('production'), '-'))
        p('   %-12s %14s %14s    sigma_prod * BR)' % ('', '', ''))
    p('')
    ig = d.meta.get('mg5_integration_pb', {})
    if ig and 'truth' in ig and 'production' in ig:
        p('   not statistics: MadEvent quotes %.4f +- %.4f pb for the truth'
          % (ig['truth']['cross_pb'], ig['truth']['error_pb']))
        p('   (one run of %d) and %.4f +- %.4f pb for the production, i.e.'
          % (len(d.meta.get('truth_runs', [])) or 1,
             ig['production']['cross_pb'], ig['production']['error_pb']))
        p('   %.3f%% and %.3f%%.'
          % (100 * ig['truth']['error_pb'] / ig['truth']['cross_pb'],
             100 * ig['production']['error_pb'] / ig['production']['cross_pb']))
    runs = d.meta.get('truth_runs', [])
    if len(runs) > 1:
        sig = [r['sigma'] for r in runs]
        p('   the %d truth runs: %s pb' % (len(sig),
                                           ', '.join('%.4f' % s for s in sig)))
        p('   spread (max-min)/mean = %.4f%%'
          % (100 * (max(sig) - min(sig)) / (sum(sig) / len(sig))))
    p('')
    p('   not a branching-ratio mismatch either.  MadSpin normalises to')
    p('   sigma_prod * BR and the truth\'s decay-chain propagator is')
    p('   normalised by the param card\'s WT, so a WT that did not equal the')
    p('   model\'s own LO Gamma(t -> W b) would show up here as a pure rate')
    p('   offset:')
    p('      LO Gamma(t -> W b) at m_t = %g, m_b = %g, MW = %.5f : %.5f GeV'
      % (mt, mb, mw, P.top_width_lo(mt, mb, mw)))
    p('      param card WT                                       : %.5f GeV'
      % gt)
    p('      implied BR                                          : %.5f'
      % (P.top_width_lo(mt, mb, mw) / gt))
    p('')
    p('what IS in it: MG5\'s decay-chain truth cuts every phase-space point')
    p('with |m - m_t| >= %g Gamma_t (myamp.f, the gForceBW=1 branch: the point'
      % bw)
    p('is rejected, so the truncation is in the integrated cross section).')
    p('MadSpin takes no such loss.  The same four evaluations of that integral')
    p('the t t~ j study makes, NWA-normalised -- they depend only on the model')
    p('parameters and on bwcutoff, both of which are the same here, so they')
    p('come out at the same numbers:')
    p('')
    p('   %-58s %8s %8s' % ('per-resonance kept fraction', 'per res', 'pair'))
    rows = [
        ('non-relativistic BW, flat numerator  [1-2*atan(2*%g)/pi]' % bw,
         P.bw_kept_fraction(mt, gt, mb, mw, bw, decay_numerator=False,
                            relativistic=False)),
        ('fixed-width relativistic BW, flat numerator',
         P.bw_kept_fraction(mt, gt, mb, mw, bw, decay_numerator=False)),
        ('  + the decay numerator m*Gamma(m)/(m_t*Gamma_t)',
         P.bw_kept_fraction(mt, gt, mb, mw, bw)),
        ('  + d ln sigma_prod/dm_t = -1.5%/GeV per top (an INPUT)',
         P.bw_kept_fraction(mt, gt, mb, mw, bw, dlnsig_dm=-0.015)),
    ]
    for lab, k in rows:
        p('   %-58s %8.5f %8.5f' % (lab, k, k * k))
    ref_mode = 'PA' if 'PA' in d.meta['runs'] else MODES[0]
    meas = d.sigma(REF) / d.sigma(ref_mode)
    p('')
    p('   %-58s %8s %8.5f'
      % ('MEASURED truth/%s' % ref_mode, '', meas))
    p('')
    p('   The truncation is again the dominant term: it removes about %.1f%% of'
      % (100 * (1 - rows[1][1] ** 2)))
    p('   the rate where the whole measured difference is %.1f%%.  What is left'
      % (100 * (1 - meas)))
    p('   over depends on which numerator variant it is measured against --')
    p('   listed below -- and as in the t t~ j study that is a range and not a')
    p('   measurement, because d ln sigma_prod / d m_t is an input this study')
    p('   never varied.  It is however a DIFFERENT')
    p('   production process from the t t~ j one, with a different slope, so')
    p('   the two residuals are not required to agree and the comparison')
    p('   between them is not evidence of anything.')
    p('')
    for lab, k in rows[1:]:
        p('      vs %-52s %+8.2f%%' % (lab.strip(), 100 * (meas / k ** 2 - 1)))
    p('')


def mechanism_report(d, p):
    """The measurement the whole study turns on.

    ``m_tt(decayed) - m_tt(production)``, event by event, for every mode.  For
    a ``2 -> 2`` production the expectation is that it is zero to the LHE's own
    text precision in every event of every mode; this is where that is either
    confirmed or refuted.
    """
    dm = d.meta.get('delta_mtt')
    if not dm:
        return
    p('-- THE MEASUREMENT: m_tt(decayed) - m_tt(production), event by event --')
    p('   MadSpin writes one decayed event per production event, in order.')
    p('   sqrt(shat) is RAMBO-invariant, so max |Delta sqrt(shat)| = 0 is what')
    p('   PROVES the two streams are actually paired -- without it the rest of')
    p('   this table would be a comparison of two unrelated events.')
    p('')
    p('   %-11s %11s %9s %9s %11s %14s %10s %8s'
      % ('mode', 'N', 'mean', 'rms', 'max|d|', 'max|dsqrt(s)|',
         'down-xing', 'up-xing'))
    for key in MODES:
        m = dm.get(key)
        if not m:
            continue
        p('   %-11s %11d %+9.2g %9.2g %11.3g %14.3g %10d %8d'
          % (key, m['n'], m['mean'], m['rms'], m['max_abs'], m['max_dshat'],
             m['crossed_down'], m['crossed_up']))
    p('')
    p('   VERDICT, per mode, from max|d| against a %g GeV bar.  That bar sits'
      % P.STRUCTURAL_TOL)
    p('   between two scales four orders of magnitude apart, so nothing hinges')
    p('   on where in between it is: the LHE writes momenta as decimal text,')
    p('   which puts a ~1e-5 GeV floor under any mode that never touches them,')
    p('   while a mode that DOES move m_tt moves it by tens of GeV (38-141 GeV')
    p('   for the t t~ j study\'s largest events).')
    moved = [k for k in MODES if dm.get(k) and not preserves_mtt(d, k)]
    kept = [k for k in MODES if dm.get(k) and preserves_mtt(d, k)]
    for key in kept:
        p('      %-11s m_tt UNCHANGED  (max|d| = %.3g GeV, at the text floor)'
          % (key, dm[key]['max_abs']))
    for key in moved:
        p('      %-11s m_tt MOVED      (max|d| = %.3g GeV, %d events crossed '
          'below 2 m_t)' % (key, dm[key]['max_abs'], dm[key]['crossed_down']))
    if moved:
        p('   %s move m_tt, so %s sub-threshold content is real and is drawn.'
          % (', '.join(moved), 'their' if len(moved) > 1 else 'its'))
        p('   Only %s carr%s the open-circle structural-zero mark.'
          % (', '.join(kept) if kept else 'nothing',
             'ies' if len(kept) == 1 else 'y'))
    else:
        p('   No mode moves m_tt.  That is the result: for this production')
        p('   multiplicity the reshuffle holds the observable fixed, so all')
        p('   four modes inherit the production sample\'s m_tt exactly and all')
        p('   four carry the open-circle structural-zero mark below 2 m_t.')
    p('')
    p('   Mechanism.  reshuffle_production hands every top-level final-state')
    p('   momentum to RAMBO mass_shuffle at fixed sqrt(shat).  Here the tops')
    p('   ARE the final state, so m_tt = sqrt(shat) identically and the')
    p('   reshuffle is holding the observable fixed by construction, whatever')
    p('   virtualities were drawn.  onshell never reshuffles at all.')
    p('   madspin_v1 rebuilds the point from the decay-chain topology holding')
    p('   shat and the production tree\'s invariants fixed -- and with two')
    p('   final-state particles the t t~ invariant IS shat.  Four different')
    p('   code paths, one kinematic reason.')
    p('')
    p('   "unchanged" fractions, i.e. |Delta m_tt| / m_tt < 1e-6:')
    for key in MODES:
        m = dm.get(key)
        if not m:
            continue
        p('      %-11s %10.4f%%  of %d events   (bit-exact: %.4f%%)'
          % (key, 100.0 * m['n_tiny'] / m['n'], m['n'],
             100.0 * m['n_exact'] / m['n']))
    p('   Read the 1e-6 column, not the bit-exact one: the bit-exact figure is')
    p('   a statement about how many decimal digits the LHE carries.')
    p('')


def overweight_report(d, p):
    """Where the curves differ at all, once the counts are known to be equal.

    If every mode's event COUNTS reproduce the production sample's bin for bin,
    then the only thing that can separate the drawn densities is the event
    WEIGHTS -- and an unweighted MadSpin sample has unit weights except where
    the overweight safety net carried a trial weight that exceeded its
    accept/reject bound.  This locates that excess on the ``m_tt`` axis.

    It is worth locating rather than quoting as a total, because the total is
    tiny (a few tens of event-equivalents in a million) while its position is
    not arbitrary: the accept/reject bound is hardest to satisfy exactly where
    the phase space is tightest.
    """
    if 'production' not in d.meta['runs']:
        return
    prod_cnt = d.z['production_cnt']
    if not all(np.array_equal(d.z['%s_cnt' % k], prod_cnt) for k in MODES):
        return                       # the counts differ; this is not the story
    fine = d.fine
    centres = 0.5 * (fine[:-1] + fine[1:])
    w0 = d.sigma('production')       # the nominal per-event weight
    p('-- what is left to separate the curves: the overweight events --------')
    p('   Every mode reproduces the production sample\'s event COUNTS bin for')
    p('   bin (above), so the only thing that can move a drawn density is the')
    p('   event WEIGHTS.  An unweighted MadSpin sample has unit weights except')
    p('   where the overweight safety net carried a trial weight that exceeded')
    p('   its accept/reject bound.  In units of one nominal event (%.4f pb):'
      % w0)
    p('')
    p('   %-11s %14s %14s %10s' % ('mode', 'excess [events]',
                                   'of which < 2m_t+1', 'rel. sigma'))
    for key in MODES:
        cnt = d.z['%s_cnt' % key]
        sw = d.z['%s_sumw' % key]
        ex = np.where(cnt > 0, (sw - cnt * w0) / w0, 0.0)
        near = ex[(centres >= d.two_mt) & (centres < d.two_mt + 1.0)].sum()
        rel = d.sigma(key) / w0 - 1.0
        p('   %-11s %14.2f %14.2f %9.4f%%'
          % (key, ex.sum(), near, 100 * rel))
    p('')
    p('   That is the whole of the spread visible in the first bin above')
    p('   threshold, and it is an accept/reject artefact rather than a')
    p('   spinmode difference: onshell and madspin_v1 took no overweights at')
    p('   all and are therefore identical to the production sample and to')
    p('   each other, bin for bin AND weight for weight.  The excess is')
    p('   concentrated within ~1 GeV of 2 m_t because that is where the decay')
    p('   weight is hardest to bound: the pair has almost no phase space left')
    p('   for two off-shell tops.  It is carried, not clipped -- see each')
    p('   run\'s log in data/logs/ -- so it is in both the central values and')
    p('   the errors here.')
    p('')


def write_numbers(d, out, fh=sys.stdout):
    """The full numeric report.  Same layout as the ``t t~ j`` study's; the
    prose is this study's, because the findings are."""
    p = lambda *a: print(*a, file=fh)
    two_mt = d.two_mt
    p('=' * 78)
    p('m_tt near threshold, 2 -> 2: MadSpin spinmodes vs the off-shell truth')
    p('=' * 78)
    p('code            : %s (%s)' % (d.meta['code_sha'][:12],
                                     d.meta.get('code_branch')))
    p('production      : %s' % d.meta['production_process'])
    p('truth           : %s' % d.meta['truth_process'])
    p('MadSpin decays  : %s' % ', '.join(d.meta['madspin_decays']))
    p('bwcutoff/BW_cut : %s / %s' % (d.meta['bwcutoff'],
                                     d.meta['madspin_BW_cut']))
    p('2 m_t           : %.4f GeV  (param card MT = %s)'
      % (two_mt, d.meta.get('param_card_masses', {}).get('MT')))
    p('observable      : %s' % d.meta.get('observable'))
    p('')
    p('THE POINT OF THIS STUDY.  For p p > t t~ j the off-shell spinmodes')
    p('populate m_tt < 2 m_t, because reshuffle_production rescales the recoil')
    p('jet and so moves the t t~ four-momentum.  With no jet there is nothing')
    p('to rescale: m_tt = sqrt(shat), which is what the reshuffle holds fixed.')
    p('The prediction is that EVERY spinmode is structurally empty below')
    p('2 m_t.  The two sections marked THE MEASUREMENT and BELOW 2 m_t are')
    p('where that is tested.')
    p('')
    p('%-11s %12s %14s %14s' % ('sample', 'events', 'sigma [pb]',
                                'banner [pb]'))
    for key in [REF] + MODES + ['production']:
        if key not in d.meta['runs']:
            continue
        p('%-11s %12s %14.4f %14s'
          % (key, '%d' % d.meta['runs'][key]['nevents'], d.sigma(key),
             '%.4f' % d.banner_sigma(key) if d.banner_sigma(key) else '-'))
    runs = d.meta.get('truth_runs', [])
    if len(runs) > 1:
        p('   truth is %d independent MG5 runs, pooled: %s events each'
          % (len(runs), ', '.join('%d' % r['nevents'] for r in runs)))
        p('   (MG5 caps one generate_events at 1M -- check_nb_events -- so')
        p('    more truth statistics can only come from more runs.)')
    p('')
    p('accept/reject scheme each run ACTUALLY used, from its own log:')
    for key in MODES + list(d.meta.get('controls', [])):
        if key not in d.meta['runs']:
            continue
        r = d.meta['runs'][key]
        p('   %-11s %-10s  (%s)' % (key, r.get('unweighting'),
                                    r.get('unweighting_why')))
    p('   NB: "legacy" is not one of the four `set unweighting` schemes.')
    p('   spinmode=madspin_v1 never reaches _unweighting_mode -- that')
    p('   dispatcher lives in run_onshell, which the legacy path does not')
    p('   call -- so the card\'s `unweighting` value is inert for it.')
    p('')

    mechanism_report(d, p)

    # --- the sub-threshold region ----------------------------------------
    st, ste, stk = d.integral(REF, d.fine[0], two_mt)
    tot = d.sigma(REF)
    p('-- BELOW 2 m_t ------------------------------------------------------')
    p('The claim is an emptiness, so it is stated as a COUNT and not only')
    p('drawn.  "0 of N" means: N events in the sample, none of them below')
    p('2 m_t, over the full histogram range (%g GeV up).' % d.fine[0])
    p('')
    p('truth       sigma(m_tt < 2m_t) = %.4f +- %.4f pb   (%d of %d events, '
      '%.3f%% +- %.3f%% of the truth total %.3f pb)'
      % (st, ste, stk, d.meta['runs'][REF]['nevents'],
         100 * st / tot, 100 * ste / tot, tot))
    for key in MODES:
        s, e, k = d.integral(key, d.fine[0], two_mt)
        n = int(d.meta['runs'][key]['nevents'])
        if s > 0:
            rel = (s / st) if st else float('nan')
            rele = (rel * math.sqrt((e / s) ** 2 + (ste / st) ** 2)
                    if st else float('nan'))
            p('%-11s sigma(m_tt < 2m_t) = %.4f +- %.4f pb   (%d of %d events)'
              '   ratio to truth = %.3f +- %.3f'
              % (key, s, e, k, n, rel, rele))
        else:
            p('%-11s sigma(m_tt < 2m_t) = 0 exactly              '
              '(0 of %d events)   %s'
              % (key, n,
                 'STRUCTURAL: m_tt is unchanged event by event (see above), '
                 'and the production sample has no support here'
                 if preserves_mtt(d, key)
                 else 'NOT structural -- %s can move m_tt, so this is a '
                      'statement about N' % key))
    p('')
    if 'production' in d.meta['runs']:
        s, e, k = d.integral('production', d.fine[0], two_mt)
        p('   and the production sample itself: %d of %d events below 2 m_t.'
          % (k, int(d.meta['runs']['production']['nevents'])))
        p('   That is the on-shell kinematic boundary, and it is the thing all')
        p('   four modes are inheriting.')
        p('')

    # --- the windows just above threshold ---------------------------------
    for width in (5.0, 10.0):
        lo_w, hi_w = d.fine[0], two_mt + width
        s0, e0, k0 = d.integral(REF, lo_w, hi_w)
        p('-- m_tt < 2 m_t + %g GeV  (= %.1f GeV) -------------------------'
          % (width, hi_w))
        p('truth       %.4f +- %.4f pb  (%d events, %.3f%% of the truth total)'
          % (s0, e0, k0, 100 * s0 / tot))
        for key in MODES:
            s, e, k = d.integral(key, lo_w, hi_w)
            diff = s - s0
            diffe = math.sqrt(e ** 2 + e0 ** 2)
            p('%-11s %.4f +- %.4f pb  (%d events)   %+.4f +- %.4f pb '
              'vs truth = %+.1f%% +- %.1f%%'
              % (key, s, e, k, diff, diffe, 100 * diff / s0,
                 100 * diffe / s0))
        p('')

    normalisation_report(d, p)

    # --- where agreement returns -----------------------------------------
    p('-- where each spinmode enters agreement with the truth --------------')
    p('   scanned downwards from %g GeV over the plot binning; "strict" uses'
      % AGREE_HI)
    p('   the central ratio, "within errors" allows each bin its own 1 sigma.')
    p('')
    p('   Three normalisations.  The FIRST is the figure\'s.')
    p('     SHAPE, total sigma  -- each side divided by its own total cross')
    p('                            section over the full m_tt range.  This is')
    p('                            what the figure draws.')
    p('     SHAPE, %g-%g GeV  -- the same idea with a local anchor instead;'
      % ANCHOR)
    p('                            a cross-check that the rate offset is flat.')
    p('     ABSOLUTE            -- no rescaling at all.')
    p('')
    shape_sc, anchor_sc = {}, {}
    for key in MODES:
        shape_sc[key] = shape_scale(d, key)
        sc, rel = anchor_scale(d, key)
        anchor_sc[key] = sc
        p('   %-11s truth/mode: total %.4f   %g-%g GeV %.4f +- %.4f'
          % (key, shape_sc[key], ANCHOR[0], ANCHOR[1], sc, sc * rel))
    p('')
    for what, scales in (
            ('SHAPE, each side over its own TOTAL sigma  (THE FIGURE)',
             shape_sc),
            ('SHAPE, each mode rescaled by its %g-%g GeV anchor (cross-check)'
             % ANCHOR, anchor_sc),
            ('ABSOLUTE normalisation', None)):
        p('   -- %s --' % what)
        for tol in (0.05, 0.10):
            p('   tolerance %d%%:' % int(100 * tol))
            for key in MODES:
                a = agreement_threshold(d, key, tol,
                                        1.0 if scales is None else scales[key])

                def fmt(name):
                    if a.get(name) is None:
                        return 'never, up to %g GeV' % AGREE_HI
                    return ('m_tt >= %.0f GeV  (= 2 m_t %+.0f GeV; first bin '
                            'ratio %.3f +- %.3f)'
                            % (a[name], a[name] - two_mt,
                               a[name + '_ratio'], a[name + '_err']))
                p('      %-11s strict       : %s' % (key, fmt('strict')))
                p('      %-11s within errors: %s' % ('', fmt('compat')))
        p('')

    # --- every mode IS the production sample ------------------------------
    if 'production' in d.meta['runs']:
        b = d.z['production_cnt']
        p('-- each mode against the undecayed production sample ----------------')
        p('   Same fine-grid m_tt histogram, bin for bin?  The production')
        p('   sample is on-shell t t~, so a "yes" is the strongest form of the')
        p('   claim: the mode did not touch m_tt anywhere, not just below')
        p('   threshold.')
        for key in MODES:
            a = d.z['%s_cnt' % key]
            same = bool(np.array_equal(a, b))
            p('   %-11s identical: %-5s  (%d vs %d entries in range, '
              'max |difference| = %d)'
              % (key, same, int(a.sum()), int(b.sum()),
                 int(np.abs(a - b).max())))
        p('   A non-zero max |difference| would be a bin-edge effect and')
        p('   nothing else at this max |Delta m_tt|: an event within 1e-5 GeV')
        p('   of a 0.25 GeV boundary can land on either side of it.  What')
        p('   matters for the figure is the 2 m_t boundary, and the')
        p('   down-crossing count in THE MEASUREMENT above speaks to that.')
        p('')
        overweight_report(d, p)

    # --- what the clipped ratio pane cannot show --------------------------
    den_c, dene_c, dcnt_c = d.shape(REF)
    p('-- SHAPE ratio points outside the figure\'s clipped pane %s -------'
      % (RATIO_CLIP,))
    any_out = False
    for key in MODES:
        y, ye, cnt = d.shape(key)
        r, re = ratio(y, ye, den_c, dene_c)
        struct = structurally_empty(d, key) & (dcnt_c > 0)
        stat = (cnt == 0) & (dcnt_c > 0) & ~struct
        for i in range(len(r)):
            if struct[i] or stat[i] or not np.isfinite(r[i]):
                continue
            if RATIO_CLIP[0] <= r[i] <= RATIO_CLIP[1]:
                continue
            any_out = True
            p('   %-11s %4.0f-%-4.0f  ratio = %.3f +- %.3f   (drawn as an '
              'arrow at %.1f)'
              % (key, d.edges[i], d.edges[i + 1], r[i], re[i],
                 RATIO_CLIP[1] if r[i] > RATIO_CLIP[1] else RATIO_CLIP[0]))
    if not any_out:
        p('   none.')
    p('   The sub-threshold zeros are NOT in this list: they are exact')
    p('   structural zeros, drawn with open markers on the lower boundary and')
    p('   no arrow.  Which modes carry one is listed here:')
    for key in MODES:
        n_struct = int((structurally_empty(d, key) & (dcnt_c > 0)).sum())
        p('      %-11s open circles in %d bin(s)   [preserves m_tt: %s]'
          % (key, n_struct, preserves_mtt(d, key)))
    p('   They are drawn CONCENTRIC, smallest on top, because they coincide:')
    p('   the turn-on is binned at 1 GeV and four markers cannot be nudged')
    p('   apart inside it.  Coincidence is the result, so it is what is drawn.')
    p('')

    # --- per-bin tables ---------------------------------------------------
    p('-- per-bin table (absolute, pb/GeV; ratios are ABSOLUTE) -------------')
    den, dene, dcnt = d.density(REF)
    head = '%9s %12s %9s' % ('bin [GeV]', 'truth', '+-')
    for key in MODES:
        head += ' %12s %8s %8s' % (key, '+-', 'ratio')
    p(head)
    for i in range(len(d.centres)):
        row = '%4.0f-%4.0f %12.5g %9.2g' % (d.edges[i], d.edges[i + 1],
                                            den[i], dene[i])
        for key in MODES:
            y, ye, cnt = d.density(key)
            r, re = ratio(y, ye, den, dene)
            if cnt[i] == 0:
                row += ' %12.5g %8s %8s' % (0.0, '-', '0 exact')
            else:
                row += ' %12.5g %8.2g %8s' % (
                    y[i], ye[i],
                    '%.3f' % r[i] if np.isfinite(r[i]) else '-')
        p(row)
    p('')

    p('-- per-bin table in the FIGURE\'s normalisation ((1/sigma) dsigma/dm,')
    p('   1/GeV; ratios are SHAPE ratios and are what the lower pane draws) --')
    sden, sdene, _ = d.shape(REF)
    head = '%9s %12s %9s' % ('bin [GeV]', 'truth', '+-')
    for key in MODES:
        head += ' %12s %8s %8s' % (key, '+-', 'ratio')
    p(head)
    for i in range(len(d.centres)):
        row = '%4.0f-%4.0f %12.5g %9.2g' % (d.edges[i], d.edges[i + 1],
                                            sden[i], sdene[i])
        for key in MODES:
            y, ye, cnt = d.shape(key)
            r, re = ratio(y, ye, sden, sdene)
            if cnt[i] == 0:
                row += ' %12.5g %8s %8s' % (0.0, '-', '0 exact')
            else:
                row += ' %12.5g %8.2g %8s' % (
                    y[i], ye[i],
                    '%.3f' % r[i] if np.isfinite(r[i]) else '-')
        p(row)
    p('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(_HERE, 'data'))
    ap.add_argument('--out', default=os.path.join(_HERE, 'plots'))
    ap.add_argument('--check-minus', action='store_true', default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    d = Data(args.data)
    base = make_figure(d, args.out)
    print('wrote %s.pdf / .png   (usetex=%s, minus fix applied=%s)'
          % (base, USETEX, MINUS_FIX))
    if args.check_minus:
        ok, detail = check_minus(base + '.pdf')
        print('minus-sign check: %s -- %s' % (ok, detail))
    with open(os.path.join(args.out, 'numbers.txt'), 'w') as fh:
        write_numbers(d, args.out, fh)
    write_numbers(d, args.out)


if __name__ == '__main__':
    main()
