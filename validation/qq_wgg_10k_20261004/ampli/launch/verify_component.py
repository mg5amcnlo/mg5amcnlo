import json
import re
from pathlib import Path

work = Path(__file__).resolve().parent
base = work / 'base'
registry = json.loads((base / 'Source/BornSupport/registry.json').read_text())
matches = [c for c in registry['contexts'] if [2, -1, 24, 21, 21] in c['born_pdgs']]
assert len(matches) == 1, [c['directory'] for c in matches]
context = matches[0]
directory = base / 'SubProcesses' / context['directory']
tables = registry['histories'][context['directory']]
assert all(not t['unresolved'] and not t['ambiguous'] for t in tables)
source = (directory / 'mc_histories.inc').read_text()
assert '.FALSE.' not in source.upper()
problem_sectors = []
for sector in context['sectors']:
    pdgs = [[leg[0] for leg in process[1]] for process in sector['processes']]
    if [2, -1, 24, 1, -1, 21] in pdgs:
        assert any(i == 6 for i, j in sector['allowed']), sector
        problem_sectors.append(dict(
            sector=sector['sector'], real_pdgs=pdgs,
            fks_pair=[sector['fks']['i'], sector['fks']['j']],
            allowed_pairs=sector['allowed']))
assert problem_sectors
foreign = sorted({h['context'] for t in tables for h in t['histories']} - {context['context']})
by_id = {c['context']: c['directory'] for c in registry['contexts']}
result = dict(
    status='verified', directory=context['directory'], context=context['context'],
    born_pdgs=context['born_pdgs'], grouped_born_flavours=context['nprocesses'],
    integrated_sector_count=len(context['sectors']),
    all_history_tables_complete=True,
    foreign_contexts=[by_id[c] for c in foreign],
    previously_unsubtracted_real_sectors=problem_sectors,
    nFKSconfigs=(directory/'nFKSconfigs.inc').read_text(),
    scope='Selected Born component of the complete pp > W+jj NLO export; not the inclusive W+jj cross section.',
    retain='Keep complete Source/BornSupport and compiled provider library when narrowing subproc.mg.',
)
(work / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
