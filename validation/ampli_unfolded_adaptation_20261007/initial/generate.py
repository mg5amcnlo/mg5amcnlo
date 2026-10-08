from pathlib import Path
import os,subprocess,sys
ROOT=Path('/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny')
WORK=Path(__file__).resolve().parent
with (WORK/'generate.log').open('w') as log:
 result=subprocess.run([sys.executable,str(ROOT/'bin/mg5_aMC'),str(WORK/'generate.cmd')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
print('Export return code:',result.returncode)
raise SystemExit(result.returncode)
