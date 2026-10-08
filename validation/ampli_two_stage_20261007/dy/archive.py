from pathlib import Path
import argparse,shutil,os
WORK=Path(__file__).resolve().parent;OUT=WORK/'drell_yan';DEST=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny/validation/ampli_two_stage_20261007/dy')
parser=argparse.ArgumentParser();parser.add_argument('name');a=parser.parse_args();base=DEST/a.name;process=base/'drell_yan'
def copy(src,dst):
 if not src.is_file():return
 dst.parent.mkdir(parents=True,exist_ok=True)
 if src.name in ('ampli_grids','grid.MC_integer') and src.is_symlink() and not Path(os.readlink(src)).is_absolute():
  if dst.exists() or dst.is_symlink():dst.unlink()
  dst.symlink_to(os.readlink(src))
 else:shutil.copy2(src,dst)
for p in WORK.glob(a.name+'*'):
 if p.is_file():copy(p,base/p.name)
for name in ('generate.cmd','generate.log','run_validation.py','watch_checkpoints.py','archive.py','first_survey_checkpoint_sha256.json','exported_source_sha256.json'):
 copy(WORK/name,DEST/name)
for p in (WORK/'sources').rglob('*'):
 if p.is_file():copy(p,DEST/'sources'/p.relative_to(WORK/'sources'))
for p in (OUT/'Cards').iterdir():
 if p.is_file():copy(p,process/'Cards'/p.name)
for p in (OUT/'Events'/a.name).iterdir():
 if p.name in ('events.lhe.gz','summary.txt','res_1.txt','res_2.txt','ampli_production.json') or p.name.endswith('_banner.txt'):
  copy(p,process/'Events'/a.name/p.name)
sp=OUT/'SubProcesses'
for p in sp.iterdir():
 if p.name in ('ampli_production.json','nevents_unweighted','randinit','proc_characteristics','subproc.mg'):
  copy(p,process/'SubProcesses'/p.name)
for folder in sp.glob('P*/G*'):
 if not folder.is_dir():continue
 for p in folder.iterdir():
  if p.name in ('ampli_grids','grid.MC_integer','ampli_pool.dat','ampli_job.dat','input_app.txt','res_0.dat','res_1.dat','res_2.dat','res_1','log_MINT0.txt','log_MINT1.txt','log_MINT2.txt'):
   copy(p,process/'SubProcesses'/p.relative_to(sp))
print(base)
