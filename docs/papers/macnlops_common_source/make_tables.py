#!/usr/bin/env python3
"""Extract manuscript tables from existing records; never run a simulation.

The files produced here are committed/copyable manuscript inputs. Rebuilding
the PDF does not require the large production tree or this extraction step.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-root', type=Path,
                        default=Path('/export/tmp/rikkert/macnlops_common_20261020'))
    args = parser.parse_args()
    root = args.production_root.resolve()
    dest = Path(__file__).resolve().parent
    source = root / 'analysis/final_accounting_review.json'
    report = json.loads(source.read_text())
    rows = {r['process']: r for r in report['processes']}
    order = ('dy', 'zz', 'ttbar', 'wbb')
    names = {'dy': 'DY', 'zz': '$ZZ$', 'ttbar': '$t\\bar t$',
             'wbb': '$W^+b\\bar b$'}
    # Numbers are rounded only for typesetting; full precision is preserved
    # in paper_data.json and the original immutable accounting report.
    rate = [r'\begin{table}[t]', r'\centering\small',
            r'\setlength{\tabcolsep}{5pt}',
            r'\begin{tabular}{lrrrrr}', r'\toprule',
            r'Process & $\sigma_{\rm src}$ & $G_H$ & $L_S$ & '
            r'$\sigma_{\rm acc}$ & Change [\%]\\',
            r'\midrule']
    for key in order:
        r = rows[key]
        digits = 5 if key == 'zz' else 4
        nums = [r[k] for k in ('original_sumw_pb', 'H_gain_pb',
                               'signed_S_loss_pb', 'retained_sumw_pb')]
        rate.append(names[key] + ' & ' + ' & '.join(f'{x:.{digits}f}' for x in nums)
                    + f" & {r['net_shift_source_percent']:+.3f}" + r'\\')
    rate.extend([r'\bottomrule', r'\end{tabular}',
                 r'\caption{Inclusive signed event-weight sums in pb, with no '
                 r'survivor rescaling. $G_H$ is the gain from bounded negative-$H$ '
                 r'removal and $L_S$ the signed loss from $S$ rejection. '
                 r'The last column is $100(G_H-L_S)/\sigma_{\rm src}$. '
                 r'The first and full stages have identical inclusive sums.}',
                 r'\label{tab:rates}', r'\end{table}', ''])
    (dest / 'rate_table.tex').write_text('\n'.join(rate))

    max_ratios = {'dy': 1.8850752438960958, 'zz': 1.3472100641779114,
                  'ttbar': 3.4280561210318115, 'wbb': 5.579695444997095}
    residual = [r'\begin{table}[t]', r'\centering\small',
                r'\setlength{\tabcolsep}{5pt}',
                r'\begin{tabular}{lrrrrrr}', r'\toprule',
                r'Process & Input $H^-$ & $K_g=0$ & $H_-/K_g>1$ & '
                r'Max. ratio & Kept $H^-$ & Kept weight [pb]\\',
                r'\midrule']
    for key in order:
        r = rows[key]
        zero = r['negative_H_with_zero_K']
        finite = r['negative_H_requiring_signed_residual'] - zero
        weight = f"{r['negative_H_retained_weight_pb']:.8f}".rstrip('0')
        residual.append(f"{names[key]} & {r['negative_H_input_events']} & "
                        f"{zero} & {finite} & {max_ratios[key]:.4f} & "
                        f"{r['negative_H_retained_events']} & {weight}" + r'\\')
    residual.extend([r'\bottomrule', r'\end{tabular}',
                     r'\caption{Required and realized negative-hard residuals. '
                     r'The zero-kernel and finite-overshoot columns count source '
                     r'points requiring nonzero survival probability (79 in total). '
                     r'The maximum excludes zero-denominator points. The kept '
                     r'count (54 in total) is one Bernoulli realization.}',
                     r'\label{tab:residuals}', r'\end{table}', ''])
    (dest / 'residual_table.tex').write_text('\n'.join(residual))

    evidence_files = [
        'source_production_index.json',
        'analysis/final_accounting_review.json',
        'analysis/common_matching_statistics/report.json',
        'analysis/frozen_source_independent_audit.json',
        'analysis/source_integrals.json',
        'analysis/required_residual_diagnostics.json',
        'analysis/required_residual_diagnostics_ttbar_wbb.json',
        'analysis/qcd_only_native_runtime_review.json',
        'analysis/common_particle_audit/first_chunks/all_processes_summary.json',
        'analysis/user_audit_skip.json',
        'analysis/dy_full_configuration_audit.json',
        'analysis/zz_full_configuration_audit.json',
        'analysis/ttbar_full_configuration_audit.json',
        'analysis/wbb_full_configuration_audit.json',
        'paper_work/setup_review.json',
        'paper_work/results_review.md',
        'paper_work/implementation_review.md',
        'paper_work/theory_review.md',
        'paper_work/implementation_draft_review.md',
        'paper_work/manuscript_theory_review.json',
    ]
    evidence = {'production_root': str(root), 'records': []}
    for name in evidence_files:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence['records'].append({'relative_path': name, 'sha256': digest(path),
                                    'bytes': path.stat().st_size})
    evidence['sources'] = {k: rows[k]['source'] for k in order}
    # The figures and tables describe the frozen patched production.  Preserve
    # its recorded code hashes: the current stock-Pythia implementation is a
    # later revision, and deleted patch files must not be silently replaced by
    # hashes of the new runtime.  This also keeps extraction portable outside
    # the original repository checkout.
    previous_path = dest / 'evidence_manifest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    evidence['code_at_manuscript_review'] = previous.get('code_at_manuscript_review', [])
    evidence['code_provenance_note'] = (
        'Historical hashes describe the frozen patched production reviewed for '
        'the manuscript. The current runtime uses stock Pythia; extraction '
        'does not recompute historical code hashes from the current checkout.')
    evidence['scope'] = ('Existing records and selected figures were read for manuscript '
                         'verification. Skipped whole-production particle and final '
                         'independent histogram/package audits were not restarted.')
    (dest / 'evidence_manifest.json').write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + '\n')
    data = {'accounting_source_sha256': digest(source), 'processes': rows,
            'max_finite_Hminus_over_Kg': max_ratios,
            'setup': json.loads((root / 'paper_work/setup_review.json').read_text())}
    (dest / 'paper_data.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print('Wrote two LaTeX tables, paper_data.json and evidence_manifest.json.')


if __name__ == '__main__':
    main()
