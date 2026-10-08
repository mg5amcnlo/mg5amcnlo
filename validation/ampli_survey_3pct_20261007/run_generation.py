from pathlib import Path
import subprocess,sys
w=Path(__file__).parent
with (w/"generate.log").open("w") as log:
 r=subprocess.run([sys.executable,"/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny/bin/mg5_aMC",str(w/"generate.cmd")],cwd="/export/tmp/rikkert/git/mg5amcnlo/MCcntRefactor_Sfun_Granny",stdout=log,stderr=subprocess.STDOUT)
print(r.returncode)
raise SystemExit(r.returncode)
