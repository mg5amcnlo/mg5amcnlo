from pathlib import Path
import os,shutil
HERE=Path(__file__).resolve().parent;WORK=Path((HERE/'work_directory.txt').read_text().strip());OUT=WORK/'drell_yan';ARCHIVE=HERE/'process';NAME='bounded2000'
def copy(src,dst):
 if not src.is_file():return
 dst.parent.mkdir(parents=True,exist_ok=True)
 if src.name in ('ampli_grids','grid.MC_integer') and src.is_symlink() and not Path(os.readlink(src)).is_absolute():
  if dst.exists() or dst.is_symlink():dst.unlink()
  dst.symlink_to(os.readlink(src))
 else:shutil.copy2(src,dst)
for name in ('run_validation.py','run.cmd','run.log','started.json','finished.json'):copy(WORK/name,HERE/name)
for p in (WORK/'sources').rglob('*'):
 if p.is_file():copy(p,HERE/'sources'/p.relative_to(WORK/'sources'))
for p in (OUT/'Cards').glob('*.dat'):copy(p,ARCHIVE/'Cards'/p.name)
for p in (OUT/'Events'/NAME).iterdir():
 if p.name in ('events.lhe.gz','summary.txt','res_1.txt','res_2.txt','ampli_production.json') or p.name.endswith('_banner.txt'):copy(p,ARCHIVE/'Events'/NAME/p.name)
sp=OUT/'SubProcesses'
for name in ('ampli_production.json','nevents_unweighted','randinit','proc_characteristics','subproc.mg'):copy(sp/name,ARCHIVE/'SubProcesses'/name)
for folder in sp.glob('P*/G*'):
 if not folder.is_dir():continue
 for p in folder.iterdir():
  if p.name in ('ampli_grids','grid.MC_integer','ampli_pool.dat','ampli_job.dat','input_app.txt','res_1.dat','res_2.dat','res_1','log_MINT1.txt','log_MINT2.txt'):copy(p,ARCHIVE/'SubProcesses'/p.relative_to(sp))
print(ARCHIVE)
