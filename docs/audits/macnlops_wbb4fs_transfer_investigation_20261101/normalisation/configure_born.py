from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT))
from madgraph.various.banner import RunCardLO
BASE=Path(__file__).resolve().parent;PROC=BASE/'Wbb_Born_LO';original=BASE.parents[1]/'processes/Wbb_4FS'
p=PROC/'Cards/run_card.dat';c=RunCardLO(str(p));c.remove_all_cut()
values=dict(nevents=200000,iseed=4920301,lpp1=1,lpp2=1,ebeam1=6500.,ebeam2=6500.,
            pdlabel='lhapdf',lhaid=334700,maxjetflavor=4,dynamical_scale_choice=3,
            fixed_ren_scale=False,fixed_fac_scale1=False,fixed_fac_scale2=False,
            scalefact=1.,event_norm='sum',use_syst=False,systematics_program='none')
for k,v in values.items():c.set(k,v,user=True)
c.write(str(p));shutil.copy2(original/'Cards/param_card.dat',PROC/'Cards/param_card.dat')
reloaded=RunCardLO(str(p))
record=dict(values={k:reloaded[k] for k in values},cuts={k:reloaded[k] for k in c.cuts_parameter},
            param_card_identical_original=hashlib.sha256((PROC/'Cards/param_card.dat').read_bytes()).hexdigest()==hashlib.sha256((original/'Cards/param_card.dat').read_bytes()).hexdigest(),
            definition='Independent positive Born control; loop_sm Wbb4FS, no cuts, same hard model/PDF334700/couplings. Native LO choice3 H_T/2 matches NLO choice3. No matching or shower.',seed=4920301)
assert record['param_card_identical_original']
(BASE/'born_settings.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print('Configured200k',record['values'])
