
import subprocess

proc2 = "generate u u~ > all all QED=2 QCD=0 [virt=QED] \n add process u d~ > all all  QED=2 QCD=0 [virt=QED]"

model_old = "loop_qcd_qed_sm-no_widths"
model_new = "loop_qcd_qed_sm_FAconv4-no_widths"

in_old = """import model %s
define all = g u c d s u~ c~ d~ s~ a ve vm vt e- mu- ve~ vm~ vt~ e+ mu+ t b t~ b~ z w+ h w- ta- ta+
%s
output standalone out_old
""" % (model_old, proc2)


in_new = """import model %s
define all = g u c d s u~ c~ d~ s~ a ve vm vt e- mu- ve~ vm~ vt~ e+ mu+ t b t~ b~ z w+ h w- ta- ta+
%s
output standalone out_old
""" % (model_new, proc2)

open('in_old.mg5', 'w').write(in_old)
open('in_new.mg5', 'w').write(in_new)

subprocess.run(['./bin/mg5_aMC', 'in_old.mg5'])
