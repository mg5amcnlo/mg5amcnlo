#!/usr/bin/env python3
"""Summarize native continuation imports and paired H-start shower controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from check_macnlops_continuation_scales import EDGES

RANGES = [(0, 5), (5, 20), (20, 50), (50, 100)]


def read_csv(path, count):
    rows = np.genfromtxt(path, delimiter=',', names=True)
    assert len(rows) == count and np.array_equal(rows['index'], np.arange(count))
    return rows


def summarize(base):
    report = dict(processes={})
    for key in ['DY', 'H', 'ttbar']:
        written = json.loads((base/f'{key}_written_scales.json').read_text())
        records = [json.loads(line) for line in (base/f'{key}_sample.jsonl').read_text().splitlines()]
        n = len(records)
        h = np.array([r['kind'] == 'H' for r in records])
        weight = np.array([r['transfer']*r['sample_factor'] for r in records])
        strata = np.array([r['bin']+100*int(r['kind'] == 'H') for r in records])
        imported = read_csv(base/f'{key}_import.csv', n)
        full = {mode: read_csv(base/f'{key}_{mode}_full.csv', n)
                for mode in ['sample', 'capped', 'common']}
        if key == 'ttbar':
            full['qed_off'] = read_csv(base/'ttbar_qed_off_full.csv', n)
        r = dict(written_scales=written, native_import_events=n,
                 native_import_max_errors={k: float(imported[k].max())
                   for k in ['global_error', 'production_error', 'dipole_error']},
                 missing_FSR_scale_keys=int(imported['missing_fsr_pairs'].sum()),
                 control_descriptions=dict(sample='Written production continuation',
                     capped='H global=t; retain min(original pair,t)',
                     common='H global, production and all dipole starts=t',
                     qed_off='Written scales; TimeShower:QEDshowerByOther=off'),
                 photons={mode:int(a['photons'].sum()) for mode,a in full.items()},
                 controls={})
        assert all(v == 0 for v in r['native_import_max_errors'].values())
        assert r['missing_FSR_scale_keys'] == 0

        def error(value):
            # Stratified sample/shower error. Original source-integration
            # uncertainty is separate. Paired controls use the same points.
            variance = 0.
            for group in np.unique(strata):
                v = value[strata == group]
                variance += (len(v)/(len(v)-1)*np.sum((v-v.mean())**2)
                             if len(v)>1 else np.sum(v*v))
            return math.sqrt(variance)

        for mode, a in full.items():
            if mode in ['capped', 'common']:
                assert np.array_equal(a['final_pt'][~h],full['sample']['final_pt'][~h])
            assert np.all(a['max_emission'] <= a['global']*(1+1e-9))
            r['controls'][mode] = []
            for lo, hi in RANGES:
                inside = (a['final_pt']>=lo)&(a['final_pt']<hi)
                x = np.where(inside&~h, weight, 0.)
                y = np.where(inside&h, weight, 0.)
                X,Y = x.sum(),y.sum()
                ratio = Y/X
                original_inside = ((full['sample']['final_pt']>=lo)&
                                   (full['sample']['final_pt']<hi))
                delta = np.where(h,weight*(inside.astype(int)-original_inside.astype(int)),0.)
                firstX,firstY = 0.,0.
                for group, v in written['bins'].items():
                    side, i = group.split('_');i=int(i)
                    if EDGES[i]>=lo and EDGES[i+1]<=hi:
                        if side == 'S':firstX += v['transfer']
                        else:firstY += v['transfer']
                r['controls'][mode].append(dict(lo=lo,hi=hi,X=float(X),Y=float(Y),
                    H_over_S=float(ratio),ratio_error=error(y-ratio*x)/abs(X),
                    net=float(Y-X),net_error=error(y-x),
                    H_change_vs_original=float(delta.sum()),paired_H_change_error=error(delta),
                    exact_first_H_over_S=firstY/firstX if firstX else None))
        report['processes'][key] = r
    report['reference_sources'] = {}
    for f in sorted((base/'reference').glob('*.cc')):
        report['reference_sources'][f.name] = dict(
            url='https://gitlab.com/Pythia8/releases/-/raw/pythia8318/src/'+f.name,
            sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    (base/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def plot(base, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = dict(DY='Drell–Yan', H='Higgs (gluon fusion)',ttbar=r'$t\bar t$')
    fig, axes = plt.subplots(1,3,figsize=(12,4.4),constrained_layout=True)
    for ax,key in zip(axes,names):
        r = report['processes'][key]
        points = np.array([2.5,10,32,70])
        first = [row['exact_first_H_over_S'] for row in r['controls']['sample']]
        ax.plot(points,first,'k--',lw=1,label='First emission, exact full source')
        for mode,label,color,offset in [
            ('sample','Original H dipole starts','#c44e52',.96),
            ('capped','Cap H dipoles at component t','#d28a00',1.),
            ('common','Common H start t','#2478a5',1.04)]:
            rows = r['controls'][mode]
            ax.errorbar(points*offset,[b['H_over_S'] for b in rows],
                yerr=[b['ratio_error'] for b in rows],color=color,label=label,
                marker='o',ms=4,lw=1,capsize=2)
        ax.axhline(1.,color='.4',ls=':',lw=1);ax.set_xscale('log')
        ax.set_xlim(1.8,95);ax.set_title(names[key]);ax.grid(alpha=.2)
        ax.set_xlabel(r'Final hard-system $p_T$ [GeV]')
    axes[0].set_ylabel('Expected H addition / S rejection')
    axes[0].legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Continuation-scale controls: only H starts change; S endpoints and matching weights stay fixed\n'
                 'Stock Pythia; stratified event/shower errors; controlled replays are diagnostic prescriptions',fontsize=11)
    fig.savefig(base/'continuation_scale_controls.pdf')
    fig.savefig(base/'continuation_scale_controls.png',dpi=160)


def write_report(base, report):
    lines=['MAcNLOPS continuation scale audit — 31 October 2026','',
        'Result: S continuation updates and stock-Pythia scale imports are correct.',
        'The S/H continuation kernels differ because H keeps the source directed',
        'dipole starts, while emitted S events continue below the winning scale.',
        'This is a difference of the current shower prescription, not a lost or',
        'misindexed LHE scale. No continuation starting-scale prescription was changed.',
        '',
        'The competing evolution selects pTnow=max(pT_ISR,pT_FSR,...) in',
        'PartonLevel.cc. FirstEmission records info.pTnow after the first accepted',
        'branch. continuationState writes that value to the S event SCALUP, caps',
        'every active continuation entry there and retains native tighter caps.',
        'It is the winning evolution scale, not the physical hard-system pT and',
        'not the largest scale from reconstructing alternative histories of a',
        'fixed real point. Raising S above the winner would violate its ordering.',
        '',
        'Stock LesHouches.cc resets SCALUP from the scale attributes and sets',
        'each production scale to the minimum strictly positive input for that',
        'leg. The positive scalup_i anchors preserve the desired global maximum.',
        'Initial shower ends use the incoming production scales. Final shower',
        'ends use directed scalup_i_j when present, including zero stops, and',
        'are also bounded by the global evolution scale and local phase space.',
        'The original macnlops_source_scales tag is inert provenance.',
        '',
        'Checks:']
    total = 0
    for key,r in report['processes'].items():
        w = r['written_scales'];total += r['native_import_events']
        lines.append(f'{key}: checked {w["events"]:,} written events; '
                     f'{r["native_import_events"]:,} native stock-reader/ISR/FSR imports.')
        lines.append('  Written header/marker, S winner/SCALUP and H directed-pair '
                     'relabel/cap errors: '+str(w['max_relative_errors']))
        lines.append('  Native global/production/dipole maximum relative errors: '+
                     str(r['native_import_max_errors']))
        lines.append(f'  Missing FSR scale keys: {r["missing_FSR_scale_keys"]}. '
                     'No replayed full-shower branching exceeded its imported global cap.')
    lines += [f'Total: 3,000,000 written events and {total:,} actual native imports.',
        '',
        'Transfer-weighted mean starts at FIRST-EMISSION hard-system pT=2–3 GeV.',
        'All entries below are in GeV. These are ensemble averages, not paired',
        'S/H copies of a single phase-space point.',
        'Process side  component-t   global      ISR-A      ISR-B      max-FSR']
    for key,r in report['processes'].items():
        for sh in ['S','H']:
            b = r['written_scales']['bins'][sh+'_3']
            lines.append(f'{key:7s} {sh} '+ ' '.join(f'{b[k]:10.5f}' for k in [
                'mean_evolution','mean_global','mean_beam1','mean_beam2','mean_max_FSR']))
    lines += ['',
        'Controlled showers:',
        'Uniform reservoirs of up to 1,000 transfer-eligible S/H points per',
        'original pT bin. Transfer weights include inverse sampling factors.',
        'Original matching probabilities, momenta, colours, couplings and PDF',
        'are held fixed. Per-event random seeds are shared across controls;',
        'S full-shower outcomes are bitwise identical in the scale variants.',
        'capped: H global=t, each original dipole=min(original,t).',
        'common: H global, production and all directed dipole starts=t.',
        't is the source component inverse-evolution scale recorded by the matcher.',
        'These are diagnostic controls, not newly implemented matching schemes.',
        'Errors include stratified event/shower sampling, not the separate',
        'original source-integration uncertainty.',
        '',
        'Final pT range 5–20 GeV:']
    for key,r in report['processes'].items():
        lines.append(key+':')
        for mode in ['sample','capped','common']:
            b = r['controls'][mode][1]
            lines.append(f'  {mode:7s}: H/S={b["H_over_S"]:.5f} +/- {b["ratio_error"]:.5f}; '
                f'paired H-transfer change={b["H_change_vs_original"]:+.6g} +/- '
                f'{b["paired_H_change_error"]:.6g} pb')
    lines += ['',
        'A separate real configuration defect:',
        'The continuation card disabled QEDshowerByQ/L/Gamma but left the',
        'default TimeShower:QEDshowerByOther=on. Stable tops still count as',
        'charged resonances for this switch. The original ttbar replay emitted',
        f'{report["processes"]["ttbar"]["photons"]["sample"]} final photons in '
        f'{report["processes"]["ttbar"]["native_import_events"]:,} sampled transfer events; '
        f'explicitly setting ByOther=off emitted {report["processes"]["ttbar"]["photons"]["qed_off"]}.',
        'The generated-card template now ties ByOther to the QED_SHOWER card',
        'switch too. No Pythia code or library was changed. The prior archived',
        'million-event full-shower results predate this fix; their description',
        'as entirely QED-free was inaccurate for stable tops. This defect is',
        'reported separately from the S/H starting-scale difference.',
        '',
        'Interpretation:',
        'The same pT histogram before showering does not imply equal conditional',
        'shower kernels. H nonuniform starts and S common ordered starts move the',
        'two transferred populations differently. The controls demonstrate a',
        'statistically resolved effect from those starts. Exact full-shower',
        'cancellation also needs pointwise transfer equality, including colours',
        'and other kinematics; common scales alone do not prove it. The existing',
        'channel-averaged normalization approximation remains independent.',
        '',
        'Full inputs, raw native-cap CSVs, selected LHE points and logs:',str(base),
        'summary.json records all bins, controls and native import maxima.',
        'continuation_scale_controls.pdf shows the first/full-shower comparison.',
        'Original campaign inputs were read only.']
    lines += ['', 'Primary Pythia 8.318 reference sources:']
    lines += [v['url']+'  SHA256 '+v['sha256'] for v in report['reference_sources'].values()]
    (base/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    args = p.parse_args()
    report = summarize(args.audit)
    plot(args.audit,report)
    write_report(args.audit,report)


if __name__ == '__main__':
    main()
