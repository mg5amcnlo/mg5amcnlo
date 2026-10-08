"""Copy a minimal, isolated ttbar export; leave both original benchmarks untouched."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/tmp/mg5-ttbar-bounded-300k-kqeozrhq/ampli')
LARGE = Path('/tmp/mg5-ttbar-30k-jobs-bx5bg6kc/ampli')
DESTINATION = Path(tempfile.mkdtemp(prefix='mg5-ampli-stable-history-')) / 'ampli'
PROCESS = Path('SubProcesses/P0_gg_ttx')


def ignore(path, names):
    path = Path(path)
    return [name for name in names if (
        path == ORIGINAL and name in ('Events', 'HTML', 'MCatNLO', 'Utilities', '.git')
    ) or (
        path.parent.name == 'SubProcesses' and path.name.startswith('P')
        and name.startswith('G') and (path / name).is_dir()
    )]


shutil.copytree(ORIGINAL, DESTINATION, symlinks=True, ignore=ignore)
shutil.copytree(ORIGINAL / PROCESS / 'GF3.0', DESTINATION / PROCESS / 'GF3.0', symlinks=True)
for name in ['ampli_grids', 'grid.MC_integer']:
    assert (ORIGINAL / PROCESS / 'GF3.0' / name).read_bytes() == (LARGE / PROCESS / 'GF3.0' / name).read_bytes()

metadata = dict(original=str(ORIGINAL), large=str(LARGE), isolated=str(DESTINATION),
                process=str(PROCESS), survey_note='Identical saved 8-learned-bin survey from the October 7 benchmarks; no new survey. This isolates generation changes and does not measure the benefit of newer survey refinement.',
                survey_hashes={name: hashlib.sha256((DESTINATION/PROCESS/'GF3.0'/name).read_bytes()).hexdigest()
                               for name in ['ampli_grids', 'grid.MC_integer']})
(HERE / 'paths.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(DESTINATION)
