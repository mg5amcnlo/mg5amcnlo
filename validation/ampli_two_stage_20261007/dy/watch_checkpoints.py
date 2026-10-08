from pathlib import Path
import hashlib,json,time
work=Path(__file__).resolve().parent;out=work/'drell_yan';seen={}
while not (work/'fresh2000_finished.json').exists():
 for grid in (out/'SubProcesses').glob('P*/GF*/ampli_grids'):
  if grid.is_symlink():continue
  for p in (grid,grid.parent/'grid.MC_integer'):
   name=str(p.relative_to(out))
   if name not in seen and p.is_file():
    seen[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (work/'first_survey_checkpoint_sha256.json').write_text(json.dumps(seen,indent=2)+'\n')
 time.sleep(.25)
print(len(seen))
