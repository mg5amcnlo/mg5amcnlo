"""Exercise each snapshotted production collector on its representative pools."""
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTIONS = Path(json.loads((HERE/'paths.json').read_text())['isolated'])/'diagnostic_collections'
results = json.loads((HERE/'collection_validation.json').read_text()) if (HERE/'collection_validation.json').exists() else {}
variants = sys.argv[1:] or sorted(path.parent.name for path in HERE.glob('*/summary.json')
                               if (HERE/(path.parent.name+'_sources')).is_dir())
for variant in variants:
    spec = importlib.util.spec_from_file_location('ampli_pool_'+variant,
            HERE/(variant+'_sources')/'madgraph/various/ampli_pool.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = json.loads((HERE/variant/'summary.json').read_text())
    results[variant] = {}
    for label, row in summary.items():
        worker = Path(row['worker'])
        pool = module.read_pool(worker)
        destination = COLLECTIONS/variant/(label+'.lhe')
        destination.parent.mkdir(parents=True,exist_ok=True)
        native_rates = list(map(float, row['res_dat'].split()))
        diagnostics = module.finalize_channel([pool], pool['final_quota'], destination,
            normalization_factor=1., rng=random.Random(743981),
            cross_sections=[(0, native_rates[2], native_rates[3])])
        count = sum(line.strip() == '<event>' for line in destination.open())
        assert count == pool['final_quota'], (count, pool['final_quota'])
        assert diagnostics['selected_tail'] < .01
        results[variant][label] = dict(diagnostics=diagnostics, events=count,
            destination=str(destination), passed=True,
            note='Diagnostic worker-only collection with unit normalization, not a complete physics sample.')
(HERE/'collection_validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
