#!/usr/bin/env python3
"""Turn the results.json of measure_unweighting.py into the LaTeX table.

    python3 make_table.py results.json > table.tex                  (one column)
    python3 make_table.py results_after.json --before results_before.json \\
            > table.tex                                    (before and after)

Column definitions -- all four come straight from counters MadSpin prints at
the end of the run, none is re-derived here:

  eps_m     "MadSpin sequential mass stage: X mass sets per accepted event"
  eps_t     "MadSpin sequential slot P: X decay events per accepted one", for
  eps_tbar  the position whose slot holds t / t~ (resolved from the LHE layout)
  N_dec     staged schemes: sum over positions of the "(N drawn)" counts on
            those same slot lines.  joint: (number of decaying particles) x the
            trial count of "MadSpin unweight efficiency: ... (W written /
            T trials, R trials/event)", because one joint trial draws exactly
            one decay per decaying particle.

eps is always "generated points per accepted point": lower is better, floor 1.

With ``--before`` the two campaigns are put side by side.  They must be the same
process, the same event count and the same seeds, and that is asserted rather
than trusted: the whole point of the comparison is that the two columns differ
only by the code under test.
"""

from __future__ import division

import argparse
import json
import math
import sys


# (results.json key, family label for the table, unweighting scheme).  The
# family label carries the density_keep_jacobian axis: two rows both saying
# "PA" with different numbers would read as a typo.
#
# ``onshell`` + ``sequential_global_retry`` IS in the table, with a dash in the
# eps_m column like the other two onshell rows.  Both halves of that are
# deliberate:
#
#   * it belongs in the table.  It is a scheme the card offers, it runs, and its
#     cost (eps_t 3.53 against 1.10 for plain sequential at 100 000 events) is
#     the measurement of what a whole-chain restart wastes when there is no
#     normalisation for it to cancel -- which is exactly the thing a reader
#     comparing unweighting schemes wants to see.  Leaving it out and describing
#     it in prose beside the table, as the first version of this table did, hides
#     the one number that makes the argument.
#
#   * its eps_m must NOT be printed.  ``onshell`` samples no virtuality
#     (``_density_do_reshuffle`` is False, ``slot_mass`` comes back empty), so
#     there is no mass stage and no mass set anywhere in the run.
#     ``_report_sequential_stats`` nevertheless prints its "mass stage" line as
#     soon as ``nb_exact_restart`` is non-zero, and under this scheme those
#     "mass sets" are chain restarts with no mass in them -- which is why the
#     number it prints (3.53) is identical to the slot-0 draw count.  Putting it
#     under a column defined as "virtuality (mass) sets drawn per written event"
#     would be a wrong number, not a rounded one.
ROW_ORDER = [
    ('PA_joint', 'PA', 'joint'),
    ('PA_sequential', 'PA', 'sequential'),
    ('PA_sequential_global_retry', 'PA', 'sequential_global_retry'),
    ('PAnojac_joint', 'PA (no jac.)', 'joint'),
    ('PAnojac_sequential', 'PA (no jac.)', 'sequential'),
    ('PAnojac_sequential_global_retry', 'PA (no jac.)',
     'sequential_global_retry'),
    ('madspin_joint', 'madspin', 'joint'),
    ('madspin_sequential', 'madspin', 'sequential'),
    ('madspin_sequential_global_retry', 'madspin', 'sequential_global_retry'),
    ('onshell_joint', 'onshell', 'joint'),
    ('onshell_sequential', 'onshell', 'sequential'),
    ('onshell_sequential_global_retry', 'onshell', 'sequential_global_retry'),
]

# Cells whose printed "mass stage" line does not count virtualities: see the
# block above.  The dash is the honest entry.
NO_MASS_STAGE = {'onshell_joint', 'onshell_sequential',
                 'onshell_sequential_global_retry'}

# Rows the per-event mass bound can reach.  ``_mass_stage_bound`` is called under
#     probe is None and maxwgts and upfront and draw_mass and not offshell
# with ``draw_mass = (spinmode == 'PA' and nb_prod_final > 1)`` and
# ``upfront = mode not in ('joint', 'sequential_with_mass')``.  So: PA only, and
# among the PA schemes only the two up-front ones.  madspin/full keep the global
# bound by decision (its w_mass carries Tr(rho_off)/|M_prod|^2_on, which has no
# cheap maximum -- measured, and the obvious partial bound is worth 2%);
# ``onshell`` has no mass stage at all.  Every other row is expected to be
# unchanged, and ``--before`` checks that rather than asserting it.
BOUND_ROWS = {'PA_sequential', 'PA_sequential_global_retry',
              'PAnojac_sequential', 'PAnojac_sequential_global_retry'}

# The keep-jacobian twins.  Moving the production-reshuffling jacobian out of
# the accept/reject changes what is being unweighted against, so these pairs
# MUST differ; identical numbers would mean the setting never took effect.
KEEP_JAC_TWINS = [('PA_joint', 'PAnojac_joint'),
                  ('PA_sequential', 'PAnojac_sequential'),
                  ('PA_sequential_global_retry',
                   'PAnojac_sequential_global_retry')]


def eps_error(eps, n_written):
    """1-sigma on a trials-per-acceptance ratio, geometric-count model.

    ``eps = D/N`` with ``N`` the number of written events -- a FIXED number, the
    one that was asked for -- and ``D`` the total number of trials, which is the
    random quantity.  Model each written event's trial count as geometric with
    success probability ``1/eps``: mean ``eps``, variance ``eps(eps-1)``.  Summed
    over ``N`` independent events and divided by ``N``,

        sigma(eps) = sqrt(eps (eps - 1) / N).

    This is a FLOOR, not a bound, and it is worth being explicit about why,
    because the caption quotes it:

      * the acceptance probability is not one number.  It varies event by event,
        and a mixture of geometrics has variance ``eps^2 - eps + 2 Var(eps_e)``,
        i.e. the model above plus ``2 Var(eps_e)``.  Only the run totals are in
        the logs, so ``Var(eps_e)`` is not measurable from them.
      * ``sequential_global_retry`` is not a per-stage geometric at all: a
        rejected decay discards the whole chain, so the stages are coupled and
        the mass stage's count carries the angle stages' rejections.
      * all rows share one production sample, so they share its fluctuation.

    Reported separately from the table for those reasons, and quoted in the
    caption as "at least", never as "at most".
    """
    if not n_written or eps is None or eps <= 1.0:
        return 0.0
    return math.sqrt(eps * (eps - 1.0) / n_written)


def row_numbers(run, position_to_pdg, key=None):
    """(eps_m, eps_t, eps_tbar, N_dec, joint_eps) for one run.

    ``joint_eps`` is not None only for the joint scheme, where there are no
    stages and the single accept/reject spans the three eps columns.
    """
    mode = run.get('reported_mode')
    if mode == 'joint':
        n_dec = run['n_trials'] * len(position_to_pdg)
        return None, None, None, n_dec, run['trials_per_event']

    slots = {int(k): v for k, v in run['slots'].items()}
    pdg_to_eps = {}
    n_dec = 0
    for position, info in slots.items():
        pdg = position_to_pdg[str(position)]
        pdg_to_eps[pdg] = info['per_accepted']
        n_dec += info['drawn']

    mass = run.get('mass_stage')
    eps_m = mass['per_accepted'] if mass else None
    if key in NO_MASS_STAGE:
        eps_m = None                      # chain restarts, not virtualities
    return eps_m, pdg_to_eps.get(6), pdg_to_eps.get(-6), n_dec, None


def fmt(value, digits=2):
    """Two decimals: MadSpin prints these counters to two decimals itself, and
    at the event counts used here the statistical error sits in the second
    decimal, so a third would be invented precision."""
    if value is None:
        return r'--'
    return ('%.' + str(digits) + 'f') % value


def num(value):
    return '{:,}'.format(value).replace(',', r'\,')


def cells_for(run, position_to_pdg, key, errors, label):
    """The four LaTeX cells of one campaign for one row, and its error entries."""
    eps_m, eps_t, eps_tb, n_dec, joint = row_numbers(run, position_to_pdg, key)
    written = run['n_written']
    if joint is not None:
        errors.append((key, label, 'joint', joint, eps_error(joint, written)))
        return r'\multicolumn{3}{c}{%s} & %s' % (fmt(joint), num(n_dec))
    for name, value in (('eps_m', eps_m), ('eps_t', eps_t), ('eps_tbar', eps_tb)):
        if value is not None:
            errors.append((key, label, name, value, eps_error(value, written)))
    return '%s & %s & %s & %s' % (fmt(eps_m), fmt(eps_t), fmt(eps_tb),
                                  num(n_dec))


def check(key, run, family):
    unw = run['unweighting_asked']
    assert run['reported_mode'] == unw, (
        '%s: asked for %s, ran %s' % (key, unw, run['reported_mode']))
    assert run['reported_why'] == 'set explicitly', (
        '%s: scheme was not set explicitly (%s)' % (key, run['reported_why']))
    want_nojac = 'no jac' in family
    assert (run.get('density_keep_jacobian') == 'False') == want_nojac, (
        '%s: density_keep_jacobian bookkeeping does not match the row' % key)
    if want_nojac:
        assert any(line.replace(' ', '').lower() == 'setdensity_keep_jacobianfalse'
                   for line in run.get('card_set_lines', [])), (
            '%s: the run did not receive "set density_keep_jacobian False"' % key)


CAPTION_COMMON = (
    r'Each $\epsilon$ is the average number of generated points needed to '
    r'accept one, so \emph{lower is better and the floor is~1}: $\epsilon_m$ '
    r'counts virtuality (mass) sets drawn per written event, $\epsilon_t$ and '
    r'$\epsilon_{\bar t}$ count decay events drawn for the $t$ and $\bar t$ '
    r'slot per written event, and $N_{\mathrm{dec}}$ is the total number of '
    r'decay events consumed over the whole run. \texttt{joint} has no separate '
    r'phases -- it makes a single accept/reject over the virtualities and both '
    r'decays at once -- so it yields one efficiency, spanning the three '
    r'$\epsilon$ columns. \texttt{onshell} keeps the production kinematics and '
    r'never samples a virtuality, so it has no mass stage at all and '
    r'$\epsilon_m$ is not defined there (dash, not~1); under '
    r'\texttt{sequential\_global\_retry} MadSpin still prints a ``mass stage'''
    r"''"
    r' line, but what it counts there is chain restarts with no mass in them '
    r'(hence a number identical to the slot-0 draw count), so that entry is a '
    r'dash as well. '
    r'The $t$/$\bar t$ split of the staged schemes is positional rather than '
    r'physical: the accept/reject fills slot~0 -- $t$, the first final-state '
    r'particle of every production event here -- against a production density '
    r'traced over the other slot, which is close to unpolarised and so accepts '
    r'readily, and slot~1 ($\bar t$) against the fully conditioned parent. '
    r'The two \texttt{PA} blocks differ only in '
    r'\texttt{density\_keep\_jacobian}: with the default \texttt{True} the '
    r'production-reshuffling phase-space jacobian is folded into the '
    r'accept/reject weight, while ``no jac.'''
    r"''"
    r' (\texttt{False}) leaves it out and applies the reshuffle as a '
    r'post-acceptance kinematic dressing instead. The two therefore unweight '
    r'against different weights and are expected to have different '
    r'efficiencies; the option is \texttt{PA}-only and is ignored by '
    r'\texttt{madspin} (which always carries that jacobian) and by '
    r'\texttt{onshell} (which never reshuffles). ')


def uncertainty_sentence(errors):
    worst = max(errors, key=lambda e: e[4]) if errors else None
    if worst is None:
        return ''
    biggest = worst[4]
    tied = [e for e in errors if e[4] > 0.9 * biggest]
    symbol = {'eps_m': r'$\epsilon_m$', 'eps_t': r'$\epsilon_t$',
              'eps_tbar': r'$\epsilon_{\bar t}$',
              'joint': r'the \texttt{joint} $\epsilon$'}
    names = ', '.join(sorted(set(
        r'%s of \texttt{%s}' % (symbol[e[2]], e[0].replace('_', r'\_'))
        for e in tied)))
    return (r'The statistical uncertainties quoted here use a geometric-count '
            r'model: a redraw-until-accepted loop makes a geometric number of '
            r'trials per written event, mean $\epsilon$ and variance '
            r'$\epsilon(\epsilon-1)$, so the mean over the $N$ written events '
            r'has $\sigma = \sqrt{\epsilon(\epsilon-1)/N}$. On that model the '
            r'largest entry in the table carries $\sigma = %.3f$ (%s) and every '
            r'other is smaller, so the second decimal is the last digit worth '
            r'reading. It is a floor rather than a bound: the acceptance '
            r'probability varies from event to event, which adds '
            r'$2\,\mathrm{Var}(\epsilon_e)$ to the variance, and '
            r'\texttt{sequential\_global\_retry} couples its stages so its mass '
            r'count is not a per-stage geometric at all. '
            % (biggest, names))


def emit_single(data, label):
    position_to_pdg = data['position_to_pdg']
    runs = data['runs']
    errors, lines, present = [], [], []
    nev = None
    for key, family, unw in ROW_ORDER:
        run = runs.get(key)
        if run is None:
            continue
        check(key, run, family)
        present.append((key, family, unw))
        nev = run['n_written'] if nev is None else nev
        fam_tex = (r'\texttt{PA} (no jac.)' if 'no jac' in family
                   else r'\texttt{%s}' % family)
        unw_tex = r'\texttt{%s}' % unw.replace('_', r'\_')
        lines.append('%s & %s & %s \\\\'
                     % (fam_tex, unw_tex,
                        cells_for(run, position_to_pdg, key, errors, 'after')))

    out = [r'\begin{table}[htbp]', r'  \centering', r'  \begin{tabular}{llrrrr}',
           r'    \toprule',
           r'    mode & unweighting & $\epsilon_m$ & $\epsilon_t$ '
           r'& $\epsilon_{\bar t}$ & $N_{\mathrm{dec}}$ \\',
           r'    \midrule']
    previous = None
    for (key, family, unw), line in zip(present, lines):
        if previous is not None and family != previous:
            out.append(r'    \midrule')
        previous = family
        out.append('    ' + line)
    out += [r'    \bottomrule', r'  \end{tabular}',
            r'  \caption{MadSpin unweighting cost per accept/reject stage for '
            r'$p\,p \to t\bar t$ with both tops decayed ($t \to b\,W^+$, '
            r'$W^+ \to \ell^+\nu$ and charge conjugate), one production sample '
            r'of %s events and one MadSpin seed shared by every row. %s%s}'
            % (num(nev or 0), CAPTION_COMMON, uncertainty_sentence(errors)),
            r'  \label{%s}' % label, r'\end{table}']
    return out, errors, nev


def emit_both(after, before, label):
    position_to_pdg = after['position_to_pdg']
    for field in ('process', 'nevents_requested', 'seed', 'nb_core'):
        assert after[field] == before[field], (
            'the two campaigns differ in %s: %r vs %r'
            % (field, before[field], after[field]))
    assert after['decays'] == before['decays'], 'the two campaigns differ in decays'
    assert after['position_to_pdg'] == before['position_to_pdg'], (
        'the two campaigns laid the slots out differently')

    errors, lines, present, moved = [], [], [], []
    nev = None
    for key, family, unw in ROW_ORDER:
        run_a, run_b = after['runs'].get(key), before['runs'].get(key)
        if run_a is None or run_b is None:
            continue
        check(key, run_a, family)
        check(key, run_b, family)
        present.append((key, family, unw))
        nev = run_a['n_written'] if nev is None else nev
        fam_tex = (r'\texttt{PA} (no jac.)' if 'no jac' in family
                   else r'\texttt{%s}' % family)
        unw_tex = r'\texttt{%s}' % unw.replace('_', r'\_')
        cb = cells_for(run_b, position_to_pdg, key, errors, 'before')
        ca = cells_for(run_a, position_to_pdg, key, errors, 'after')
        lines.append('%s & %s & %s & %s \\\\' % (fam_tex, unw_tex, cb, ca))
        if cb != ca:
            moved.append(key)

    out = [r'\begin{table}[htbp]', r'  \centering', r'  \small',
           r'  \begin{tabular}{ll rrrr rrrr}', r'    \toprule',
           r'    & & \multicolumn{4}{c}{before} & \multicolumn{4}{c}{after} \\',
           r'    \cmidrule(lr){3-6} \cmidrule(lr){7-10}',
           r'    mode & unweighting & $\epsilon_m$ & $\epsilon_t$ '
           r'& $\epsilon_{\bar t}$ & $N_{\mathrm{dec}}$ '
           r'& $\epsilon_m$ & $\epsilon_t$ & $\epsilon_{\bar t}$ '
           r'& $N_{\mathrm{dec}}$ \\',
           r'    \midrule']
    previous = None
    for (key, family, unw), line in zip(present, lines):
        if previous is not None and family != previous:
            out.append(r'    \midrule')
        previous = family
        out.append('    ' + line)
    moved_tex = ', '.join(r'\texttt{%s}' % k.replace('_', r'\_') for k in moved)
    out += [r'    \bottomrule', r'  \end{tabular}',
            r'  \caption{MadSpin unweighting cost per accept/reject stage for '
            r'$p\,p \to t\bar t$ with both tops decayed ($t \to b\,W^+$, '
            r'$W^+ \to \ell^+\nu$ and charge conjugate), before and after the '
            r'per-event mass-stage bound. One production sample of %s events '
            r'and one MadSpin seed are shared by every row \emph{and by both '
            r'campaigns}, so the two halves of each row differ only by the code '
            r'under test. %s'
            r'The bound is what changed: the sequential mass stage used to test '
            r'its weight against a single run-wide number extrapolated from the '
            r'probe (mean $+\,$\texttt{nb\_sigma}$\,\times\,$sd over the first '
            r'\texttt{Nevents\_for\_max\_weight} events, times 1.10) and now '
            r'tests it against an exact per-production-event maximum. It is '
            r'reached only for \texttt{spinmode = PA} and only in the two '
            r'up-front schemes, so exactly the rows %s move; '
            r'\texttt{madspin} keeps the global bound by decision (its mass '
            r'weight carries a factor with no cheap maximum) and '
            r'\texttt{onshell} has no mass stage to bound, and their halves are '
            r'identical here rather than merely close. %s'
            r'Because the accepted density is $q_e(m)\,\min(1, w/C)$, any '
            r'$C \ge \max w$ cancels out of it: the bound sets the cost and not '
            r'the sample, and the lineshape measurement in '
            r'\texttt{MadSpin/validation/mt\_lineshape} is the check that it '
            r'does.}'
            % (num(nev or 0), CAPTION_COMMON, moved_tex or 'none',
               uncertainty_sentence(errors)),
            r'  \label{%s}' % label, r'\end{table}']
    return out, errors, nev


def diagnostics(data, errors, label_of_campaign, runs_extra=None):
    position_to_pdg = data['position_to_pdg']
    runs = data['runs']
    w = sys.stderr.write
    w('\n%-42s %-7s %-9s %8s %8s\n'
      % ('run', 'campaign', 'column', 'value', '1sigma'))
    for key, campaign, name, value, err in errors:
        w('%-42s %-7s %-9s %8.4f %8.4f\n' % (key, campaign, name, value, err))
    w('\ndensity_keep_jacobian True vs False (must differ -- identical numbers '
      'would mean the setting never took effect):\n')
    for a, b in KEEP_JAC_TWINS:
        if a not in runs or b not in runs:
            continue
        na = row_numbers(runs[a], position_to_pdg, a)
        nb = row_numbers(runs[b], position_to_pdg, b)
        same = na[:4] == nb[:4] and na[4] == nb[4]
        w('  %-32s %-34s %s\n' % (a, b, 'IDENTICAL -- INVESTIGATE' if same
                                  else 'differ (expected)'))
    w('\nposition -> pdg: %s\n' % position_to_pdg)
    w('final-state layouts seen in the production sample: %s\n'
      % data['slot_diagnostics']['layouts'])
    w('production cross section: %s pb\n' % data.get('cross_in'))
    w('\n%-34s %8s %8s %8s %8s %10s %8s %6s %8s %s\n'
      % ('run', 'written', 'eps_m', 'eps_t', 'eps_tbar', 'N_dec', 'wall_s',
         'ovfl', 'carried', 'mass bound'))
    for key, run in runs.items():
        eps_m, eps_t, eps_tb, n_dec, joint = row_numbers(run, position_to_pdg,
                                                         key)
        if joint is not None:
            eps_m = eps_t = eps_tb = joint
        ow = run.get('overweight')
        mb = run.get('mass_bound')
        w('%-34s %8d %8s %8s %8s %10d %8.0f %6d %8s %s\n'
          % (key, run['n_written'], fmt(eps_m, 4), fmt(eps_t, 4),
             fmt(eps_tb, 4), n_dec, run['wall_seconds'], run['overflows'],
             '--' if ow is None else '%d' % ow['events'],
             '--' if mb is None else
             ('%d/%d per-event' % (mb['per_event'], mb['events'])
              + ('' if not mb['global_fallback']
                 else ', %d global' % mb['global_fallback']))))
    w('\nrows the bound is expected to reach: %s\n' % ', '.join(sorted(BOUND_ROWS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--before', default=None,
                        help='a second results.json, put in a "before" block')
    parser.add_argument('--label', default='tab:madspin-unweighting-eps')
    args = parser.parse_args()

    with open(args.results) as fp:
        after = json.load(fp)
    if args.before:
        with open(args.before) as fp:
            before = json.load(fp)
        out, errors, _nev = emit_both(after, before, args.label)
    else:
        before = None
        out, errors, _nev = emit_single(after, args.label)
    print('\n'.join(out))
    diagnostics(after, errors, 'after')
    if before is not None:
        sys.stderr.write('\n=== before campaign, same diagnostics ===\n')
        diagnostics(before, [], 'before')


if __name__ == '__main__':
    main()
