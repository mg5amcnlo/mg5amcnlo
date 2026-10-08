"""Keep numerical evidence and the final sample, without build/candidate spools."""
from pathlib import Path
import hashlib, json, os, shutil

HERE = Path(__file__).resolve().parent
WORK = Path((HERE / 'work_directory.txt').read_text().strip())
OUT = WORK / 'ampli'
DEST = HERE / 'ampli'
RUN = 'benchmark_300k'
finished = json.loads((HERE / 'ampli_finished.json').read_text())
assert finished['cli_returncode'] == 0 and finished['events_exist'] and finished['summary_exists']
assert (OUT / 'SubProcesses/randinit').read_text().strip() == 'r=19727'
assert not list((OUT / 'SubProcesses').glob('P*/G*/log_MINT0.txt'))
assert not list((OUT / 'SubProcesses').glob('P*/G*/res_0.dat'))
files = [p for p in (OUT / 'Cards').iterdir() if p.is_file()]
for pattern in ['events.lhe.gz', '*_banner.txt', 'summary.txt', 'ampli_production.json', 'res_*.txt']:
    files += list((OUT / 'Events' / RUN).glob(pattern))
for name in ['randinit', 'nevents_unweighted', 'ampli_production.json', 'subproc.mg', 'proc_characteristics']:
    p = OUT / 'SubProcesses' / name
    if p.is_file(): files.append(p)
names = ['log_MINT1.txt', 'log_MINT2.txt', 'res_1.dat', 'res_2.dat', 'ampli_grids',
         'grid.MC_integer', 'ampli_pool.dat', 'ampli_job.dat', 'input_app.txt', 'randinit']
for name in names:
    files += list((OUT / 'SubProcesses').glob('P*/G*/' + name))
files += [OUT / name for name in finished['exported_source_sha256']]
hashes = {}
for src in files:
    relative = src.relative_to(OUT)
    dst = DEST / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink() and not os.path.isabs(os.readlink(src)):
        if dst.is_symlink(): dst.unlink()
        if not dst.exists(): dst.symlink_to(os.readlink(src))
    else:
        shutil.copy2(src, dst)
    hashes[str(relative)] = hashlib.sha256(src.read_bytes()).hexdigest()
for name, digest in hashes.items():
    assert hashlib.sha256((DEST / name).read_bytes()).hexdigest() == digest, name
(HERE / 'archive_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
print('Archived and checksummed', len(hashes), 'files in', DEST)
