from pathlib import Path
import json,math,random,subprocess
D=Path(__file__).resolve().parent
rows=[json.loads(s) for s in (D/'profiles.jsonl').read_text().splitlines()]
positive=[r for r in rows if r['weight']>0 and .02<r['P']<.97]
rng=random.Random(4011111)
selected={}
for lo,hi in [(1.7,5),(5,10),(10,20),(20,50),(50,10000)]:
 eligible=[r for r in positive if lo<=r['pt']<hi]
 ranked=sorted(eligible,key=lambda r:-math.log(rng.random())/(r['weight']*(1-r['keep'])))
 for r in ranked[:4]:selected[r['index']]=r
# Add contexts with the largest bottom FSR exponents, to test the massive kernel directly.
for r in sorted(positive,key=lambda r:r['P_FSR'])[:6]:selected[r['index']]=r
points=list(sorted(selected.values(),key=lambda r:r['index']))
assert all(r['gluon_count']==0 for r in rows)
(D/'selected_profiles.json').write_text(json.dumps(points,indent=2)+'\n')
def particle(p):
 e,x,y,z=p['p'];return ' '.join(map(str,[p['id'],p['status'],*p['mothers'],*p['colors'],x,y,z,e,p['mass'],0.,9.]))
for mode,key in [('ALL','P'),('ISR','P_ISR'),('FSR','P_FSR')]:
 lines=[str(len(points))]
 for r in points:
  lines.append(' '.join(map(str,[r['index'],len(r['born']),len(r['Q']),r['pt'],r[key]])))
  lines.extend(particle(p) for p in r['born'])
  lines.extend(' '.join(map(str,q)) for q in r['Q'])
 (D/('points_'+mode+'.dat')).write_text('\n'.join(lines)+'\n')
reference=Path('/export/slow1/rikkert/macnlops_native_ir_jets_1M_20261031/investigation_wj/warnings/conditional_sudakov_components_unique.cc').read_text()
reference=reference.replace('mode!="ISR"&&mode!="FSR"','mode!="ISR"&&mode!="FSR"&&mode!="ALL"')
reference=reference.replace('NNPDF40_nlo_as_01180/0','NNPDF40_nlo_as_01180_nf_4/0').replace('o.flavourCount=5','o.flavourCount=4')
reference=reference.replace('p.settings.flag(mode=="FSR"?"SpaceShower:QCDshower":"TimeShower:QCDshower",false);','if(mode!="ALL")p.settings.flag(mode=="FSR"?"SpaceShower:QCDshower":"TimeShower:QCDshower",false);')
(D/'conditional_survival.cc').write_text(reference)
cmd=['g++','-O2','-std=c++17','-I/nfs/home/rikkert/Downloads/pythia8/include',str(D/'conditional_survival.cc'),'-L/nfs/home/rikkert/Downloads/pythia8/lib','-Wl,-rpath,/nfs/home/rikkert/Downloads/pythia8/lib','-lpythia8','-ldl','-o',str(D/'conditional_survival')]
with (D/'survival_build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Selected points:',len(points))
print('All 368 profiles have zero final Born gluons.')
print([(r['index'],round(r['pt'],3),round(r['P'],3),round(r['P_ISR'],3),round(r['P_FSR'],3)) for r in points])
