"""Summarize the completed paired worker replays without rerunning physics."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
runs = {name: json.loads((HERE/name/'summary.json').read_text())
        for name in ['baseline', 'candidate']}
comparison = {}
for label in runs['baseline']:
    rows = {}
    for name, run in runs.items():
        row = run[label]
        values = list(map(float, row['res_dat'].split()))
        row['published_rate'] = dict(absolute=values[0], error_abs=values[1],
                                     signed=values[2], error_signed=values[3])
        pool_lines = (HERE/name/label/'ampli_pool.dat').read_text().splitlines()
        pool_version = int(pool_lines[0].split()[1])
        nepochs = int(pool_lines[1].split()[2])
        row['epoch_details'] = []
        for line in pool_lines[6:6+nepochs]:
            fields = line.split()
            row['epoch_details'].append(dict(epoch=int(fields[0]), trials=int(fields[1]),
                nonzero=int(fields[2]), target_nonzero=int(fields[3]),
                proposal=int(fields[13]) if pool_version == 5 else int(fields[0]),
                eligible=bool(int(fields[12]))))
        row['held_adaptation_after_epochs'] = [a['epoch'] for a,b in
            zip(row['epoch_details'], row['epoch_details'][1:]) if a['proposal']==b['proposal']]
        rows[name] = row
    a, b = rows['baseline'], rows['candidate']
    rows['relative_changes'] = dict(trials=b['trials']/a['trials']-1,
        cpu=b['timing']['Total']/a['timing']['Total']-1,
        wall=b['wall_seconds']/a['wall_seconds']-1,
        candidates=b['candidates']/a['candidates']-1)
    comparison[label] = rows
notes = [
    'Representative single-seed P0_gg_ttx/GF3.0 generation workers; this is not a full ttbar benchmark or statistical coverage study.',
    'Fixed common saved survey, with eight learned bins per coordinate. New survey-bin refinement is present in both source variants but is not exercised.',
    'Baseline snapshots include tail-aware forecast and completion checks independent of adaptation. Candidate adds stable adaptation/useful-history retention.',
    'Both variants start from the same saved survey, worker quotas, card inputs and seed19727. RNG sequences can diverge after adaptation decisions differ.',
    'Reported worker cross sections are generation-only channel estimates, not full physical ttbar rates. Native published uncertainties are read from res.dat.',
    'CPU and wall measurements include a single execution per variant/worker and are subject to host timing noise; trial counts are deterministic for these inputs.',
    'All three full-trial/reserve/worst-final-subset overweight checks must remain strictly below 1%.',
    'The large candidate first holds adaptation after epoch 6: epochs 6 and 7 use the same proposal 6. Earlier trial counts and rates match the baseline. It then finishes in epoch 8 with seven proposals; baseline finishes in epoch 9 and expires epoch 1.',
    'These runs do not isolate adaptation from history effects. The candidate finishes with only eight epochs, so they do not directly exercise retention of more than eight epochs; that path requires separate synthetic tests.',
    'Small-worker candidate LHE payload is byte-identical to baseline and its numerical result is unchanged. The 4.2% timing decrease there is timing noise; do not interpret the similarly sized large-worker CPU decrease as a precision speedup estimate.',
]
collection = json.loads((HERE/'collection_validation.json').read_text()) if (HERE/'collection_validation.json').exists() else None
(HERE/'comparison.json').write_text(json.dumps(dict(notes=notes,workers=comparison,
                                                    actual_collection=collection),indent=2)+'\n')
lines = ['Paired ttbar worker comparison\n', *[note+'\n' for note in notes], '\n']
for label, row in comparison.items():
    lines.append(label+' events/job setting\n')
    for name in ['baseline','candidate']:
        r=row[name]
        lines.append(f"  {name}: trials={r['trials']}, CPU={r['timing']['Total']:.6f}s, wall={r['wall_seconds']:.6f}s, "
                     f"epochs={r['epochs']}, updates={r['adaptation']['updates']}, eligible_epochs={r['eligible_epochs']}, "
                     f"expired_trials={r['expired_trial_fraction']:.6%}, max_tail={max(r['tails'].values()):.6%}\n")
        lines.append('    rates: '+json.dumps(r['published_rate'])+'\n')
        lines.append('    epochs/trials/target/proposal/eligible: '+
                     '; '.join('/'.join(str(e[k]) for k in ['epoch','trials','target_nonzero','proposal','eligible'])
                               for e in r['epoch_details'])+'\n')
        lines.append('    held adaptation after epochs: '+str(r['held_adaptation_after_epochs'])+'\n')
        if collection:
            c=collection[name][label]
            lines.append(f"    Actual diagnostic LHE collection passed: {c['events']} events, "
                         f"selected tail={c['diagnostics']['selected_tail']:.6%}\n")
    lines.append('  Relative changes: '+json.dumps(row['relative_changes'])+'\n\n')
(HERE/'findings.txt').write_text(''.join(lines))
print(''.join(lines))
