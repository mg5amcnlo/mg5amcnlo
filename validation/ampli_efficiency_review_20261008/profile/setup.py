"""Prepare a small isolated export for the exact two representative worker replays."""
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = Path('/tmp/mg5-ttbar-bounded-300k-kqeozrhq/ampli')
large = Path('/tmp/mg5-ttbar-30k-jobs-bx5bg6kc/ampli')
destination = Path(tempfile.mkdtemp(prefix='mg5-ampli-profile-')) / 'ampli'

def ignore(path, names):
    path = Path(path)
    return [name for name in names if (
        path == original and name in ('Events', 'HTML', 'MCatNLO', 'Utilities', '.git')
    ) or (
        path.parent.name == 'SubProcesses' and path.name.startswith('P')
        and name.startswith('G') and (path / name).is_dir()
    )]

shutil.copytree(original, destination, symlinks=True, ignore=ignore)
subprocess = Path('SubProcesses/P0_gg_ttx')
for name in ['GF3.0', 'GF3.0_1']:
    shutil.copytree(original / subprocess / name, destination / subprocess / name, symlinks=True)
for name in ['ampli_grids', 'grid.MC_integer']:
    assert (original / subprocess / 'GF3.0' / name).read_bytes() == (large / subprocess / 'GF3.0' / name).read_bytes()
shutil.copytree(large / subprocess / 'GF3.0_1', destination / subprocess / 'GF3.0_profile30k', symlinks=True)
(HERE / 'paths.json').write_text(json.dumps({
    'original': str(original), 'profile': str(destination),
    'worker': str(subprocess / 'GF3.0_1'),
}, indent=2) + '\n')
print(destination)
