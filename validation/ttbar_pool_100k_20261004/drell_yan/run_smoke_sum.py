from pathlib import Path
import subprocess,sys,os
work=Path(__file__).parent; out=work/"drell_yan"
with (work/"restart_sum.log").open("w") as log:
    p=subprocess.run([sys.executable,"-O",str(out/"bin/aMCatNLO"),str(work/"restart_sum.cmd")],cwd=out,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,"OMP_NUM_THREADS":"1"})
print("CLI exit",p.returncode,"work",work,flush=True)
