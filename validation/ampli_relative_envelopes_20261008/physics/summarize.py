"""Summarize isolated relative-envelope/stream-mixture worker experiments."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
variants=['baseline','baseline_diagnostic','quantile','observed_maximum','stream_mixture','combined','adopted']
runs={name:json.loads((HERE/name/'summary.json').read_text()) for name in variants if (HERE/name/'summary.json').exists()}
collection=json.loads((HERE/'collection_validation.json').read_text()) if (HERE/'collection_validation.json').exists() else {}
workers={}
for name,run in runs.items():
 for label,raw in run.items():
  row=dict(raw)
  lines=(HERE/name/label/'ampli_pool.dat').read_text().splitlines()
  version=int(lines[0].split()[1]);nepochs=int(lines[1].split()[2])
  row['epoch_details']=[]
  for line in lines[6:6+nepochs]:
   f=line.split();row['epoch_details'].append(dict(epoch=int(f[0]),trials=int(f[1]),nonzero=int(f[2]),
    target_nonzero=int(f[3]),proposal=int(f[13]) if version==5 else int(f[0]),eligible=bool(int(f[12]))))
  baseline=runs['baseline'][label]
  timing_baseline=runs.get('baseline_diagnostic',{}).get(label,baseline)
  row['relative_to_current_baseline']=dict(trials=row['trials']/baseline['trials']-1,candidates=row['candidates']/baseline['candidates']-1)
  row['timing_comparison']=dict(cpu_fraction=row['timing']['Total']/timing_baseline['timing']['Total']-1,
   wall_fraction=row['wall_seconds']/timing_baseline['wall_seconds']-1,
   baseline='baseline_diagnostic' if label in runs.get('baseline_diagnostic',{}) else 'baseline',
   instrumentation_matched=(label!='30000' or name=='baseline'))
  if label in collection.get(name,{}):row['collection']=collection[name][label]
  workers.setdefault(label,{})[name]=row
notes=[
 'Fixed saved survey and seed 19727, no folding, one gg ttbar channel per executable. This is not a full ttbar event-generation benchmark or statistical coverage study.',
 'GF3 small final 2242/reserve 2467; GF3 large final 24655/reserve 27121; GF1 large final 25985/reserve 28584.',
 'Baseline includes tail-aware forecast, independent completion checks, stable adaptation, and proposal history retention.',
 'Quantile: upper-five-percent candidate order statistic after 200 candidates; same sparse-pool bootstrap maximum as baseline. Observed maximum: only remove the first-proposal survey floor after 200 candidates, retain observed maximum. Stream mixture: fixed survey-derived upward virtual probability adjustment, capped 4x and 10%, no relative-envelope change.',
 'GF3 baseline results reused after exact numerical/coordinator source hashes matched the previous stable-history candidate. A fresh GF3 small instrumented replay verifies byte-identical pool and candidate LHE; GF1 baseline is freshly instrumented.',
 'All newly run variants include stream diagnostics without extra RNG calls. CPU/wall include diagnostic I/O and host timing noise. For GF3 large, reused baseline is uninstrumented so CPU/wall deltas are not a controlled speed measurement; deterministic trial counts are primary.',
 'All final tail checks remain strict: full-trial, reserve, and worst-final-subset are below 1%. Actual diagnostic collection checks the requested exact event count and selected tail.',
 'Combined means q95 plus the experimental fixed-mixture helper, with ordinary-input arithmetic equal to the mixture-only prototype. The mixture was not adopted in production.',
 'Reported rates are generation-only worker/channel estimates, not the full physical ttbar rate. Shared seeds and stopping correlate the experiments; do not interpret quadrature error differences as independent-seed significance.',
]
extra_pairs={}
for seed,label in [('39727','30000'),('59727','gf1_large')]:
 names=['baseline_seed'+seed,'combined_seed'+seed]+(['quantile_seed'+seed] if seed=='59727' else [])
 rows={name:json.loads((HERE/name/'summary.json').read_text())[label] for name in names if (HERE/name/'summary.json').exists()}
 if not rows:continue
 baseline=rows.get('baseline_seed'+seed)
 for name,row in rows.items():
  if baseline:row['relative_to_current_baseline']=dict(trials=row['trials']/baseline['trials']-1,cpu=row['timing']['Total']/baseline['timing']['Total']-1)
  if label in collection.get(name,{}):row['collection']=collection[name][label]
 extra_pairs[seed]=dict(workload=label,runs=rows,note='Same saved survey/cards/quotas, changed private randinit seed; all variants instrumented.')
report=dict(notes=notes,workers=workers,extra_seed_pairs=extra_pairs)
(HERE/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
text=['Relative-envelope and internal-mixture worker experiments\n',*[n+'\n' for n in notes],'\n']
for label,rows in workers.items():
 text.append(label+'\n')
 for name,r in rows.items():
  rel=r['relative_to_current_baseline']
  text.append(f"  {name}: trials={r['trials']} ({rel['trials']:+.3%}), candidates={r['candidates']}, CPU={r['timing']['Total']:.3f}s, wall={r['wall_seconds']:.3f}s, epochs={r['epochs']}, max_tail={max(r['tails'].values()):.6%}\n")
  text.append('    Rates: '+json.dumps(r['published_rate'])+'\n')
  if 'collection' in r:
   c=r['collection'];text.append(f"    Collected {c['events']} events, selected tail={c['diagnostics']['selected_tail']:.6%}, passed={c['passed']}\n")
 text.append('\n')
for seed,pair in extra_pairs.items():
 text.append('Additional seed '+seed+' '+pair['workload']+'\n')
 for name,r in pair['runs'].items():
  text.append(f"  {name}: trials={r['trials']}, CPU={r['timing']['Total']:.3f}s, max_tail={max(r['tails'].values()):.6%}, relative={r.get('relative_to_current_baseline',{})}\n")
  text.append('    Rates: '+json.dumps(r['published_rate'])+'\n')
  if 'collection' in r:
   c=r['collection'];text.append(f"    Collected {c['events']} events, selected tail={c['diagnostics']['selected_tail']:.6%}, passed={c['passed']}\n")
 text.append('\n')
(HERE/'findings.txt').write_text(''.join(text));print(''.join(text))
