#!/usr/bin/env python3
"""Common-source first-emission and streamed full-parton validation plots."""
from pathlib import Path
import argparse, concurrent.futures, csv, datetime, fcntl, hashlib, html, importlib.util, json, os, signal, subprocess, sys, time, traceback
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_macnlops_campaign as campaign
PRODUCTION=campaign.DEFAULT_BASE
BASE=PRODUCTION/'validation'
CONFIG=None
from madgraph.various import macnlops as mc
import numpy as np
spec=importlib.util.spec_from_file_location('prior_plotter',ROOT/'docs/audits/plot_macnlops_validation.py')
prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
STAGES=prior.STAGES
PYTHIA=Path('/export/tmp/rikkert/Pythia/pythia8318')
PROCESSES={p['key']:(p['analysis'],p['directory']) for p in campaign.PROCESSES}
PROCESS_LABELS={'zz':'ZZ','dy':'Drell–Yan','ttbar':'Top pairs','single_top':'Single top (t channel)','wbb':r'$W^+b\bar b$ (4FS)','wjet':r'$W^++j$ (5FS)'}
CONSUMER=Path(__file__).with_name('macnlops_stream_hist.cc')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(path,data):
 campaign.dump(path,data)
def specs_for(process):
 specs=prior.observable_specs('ttbar' if process=='single_top' else 'dy' if process=='wjet' else process)
 specs['system_pt_soft']=prior.Observable(np.linspace(0,50,101),r'Hard-system $p_T$ (soft region)','GeV',True)
 if process=='ttbar':
  extended=np.linspace(0.,150.,151)
  specs['jet_pt_soft'].edges=extended
  specs['system_pt_soft'].edges=extended
 if process=='single_top':
  specs['system_mass']=prior.Observable(np.linspace(170.,1800.,66),r'$m(t+j_1)$','GeV')
  specs['system_pt'].label=r'$p_T(t+j_1)$'
  specs['system_pt_soft'].label=r'$p_T(t+j_1)$ (soft region)'
  specs['system_y'].label=r'$y(t+j_1)$'
  specs['delta_phi'].label=r'$|\Delta\phi(t,j_1)|$'
  specs['delta_phi_deficit'].label=r'$\pi-|\Delta\phi(t,j_1)|$'
  specs['delta_y'].label=r'$y(t)-y(j_1)$'
  specs['object_2_pt'].label=r'Leading jet $p_T$ (spectator proxy)'
  specs['object_2_y'].label='Leading jet rapidity (spectator proxy)'
  specs['jet_pt'].label=r'Second jet $p_T$'
  specs['jet_pt_soft'].label=r'Second jet $p_T$ (soft region)'
  specs['jet_y'].label=r'Second jet rapidity ($p_T>20$ GeV)'
 if process=='wbb':
  specs['jet_pt'].label=r'Leading non-bottom radiation jet $p_T$'
  specs['jet_pt_soft'].label=r'Leading non-bottom radiation jet $p_T$ (soft region)'
  specs['jet_y'].label=r'Leading non-bottom radiation jet rapidity ($p_T>20$ GeV)'
  specs['system_pt'].label=r'$p_T(W^+b\bar b)$'
  specs['system_pt_soft'].label=r'$p_T(W^+b\bar b)$ (soft region)'
 if process=='wjet':
  specs['system_mass']=prior.Observable(np.linspace(80.,1800.,87),r'$m(W^++j_1)$','GeV')
  for name,label in dict(system_pt=r'$p_T(W^++j_1)$',system_pt_soft=r'$p_T(W^++j_1)$ (soft region)',
   system_y=r'$y(W^++j_1)$',delta_phi=r'$|\Delta\phi(W^+,j_1)|$',
   delta_phi_deficit=r'$\pi-|\Delta\phi(W^+,j_1)|$',delta_y=r'$y(W^+)-y(j_1)$',
   object_1_pt=r'$W^+$ transverse momentum',object_1_y=r'$W^+$ rapidity',
   object_2_pt=r'Leading fiducial jet $p_T$',object_2_y='Leading fiducial jet rapidity',
   jet_pt=r'Second central jet $p_T$',jet_pt_soft=r'Second central jet $p_T$ (soft region)',
   jet_y=r'Second jet rapidity ($p_T>25$ GeV)').items():specs[name].label=label
  specs['object_2_y'].edges=np.linspace(-4.5,4.5,46)
  specs['jet_y'].edges=np.linspace(-4.5,4.5,46)
 return specs

def analysis_plan():
 return dict(production=str(PRODUCTION),output=str(BASE),cpu_slots=CONFIG['cpu_slots'],
  levels=['first','full'],stages=list(STAGES),ratio_denominator='macnlops',
  no_hepmc=True,hadronisation=False,decays=False,mpi=False,primordialkt=False,qed=False,
  processes=[dict(key=p['key'],process=p['process'],born_partons=p['born_partons'],
   analysis_jets=p.get('analysis_jets',dict(algorithm='antikt',radius=.4)),
   observables={name:dict(edges=list(spec.edges),label=spec.label,unit=spec.unit)
                for name,spec in specs_for(p['analysis']).items()}) for p in CONFIG['processes']])

def build():
 directory=BASE/'build';directory.mkdir(parents=True,exist_ok=True)
 fj=ROOT/'Template/NLO/SubProcesses'
 inputs={str(path):mc.sha256(path) for path in [CONSUMER,fj/'fjcore.cc',fj/'fjcore.hh']}
 record=dict(inputs=inputs,pythia8=str(PYTHIA))
 binary=directory/'StreamHist';manifest=directory/'build.json'
 if binary.exists() and manifest.exists() and json.loads(manifest.read_text())==record:return binary
 command=['g++','-O2','-std=c++17','-I'+str(PYTHIA/'include'),'-I'+str(fj),str(CONSUMER),str(fj/'fjcore.cc'),
  '-L'+str(PYTHIA/'lib'),'-Wl,-rpath,'+str(PYTHIA/'lib'),'-lpythia8','-lz','-lcrypto','-ldl','-o',str(binary)+'.tmp']
 with (directory/'build.log').open('w') as stream:
  result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 if result.returncode:raise RuntimeError('Consumer build failed; see '+str(directory/'build.log'))
 Path(str(binary)+'.tmp').replace(binary);dump(manifest,record)
 return binary

def prepare(keys=None,modes=None):
 configs=json.loads((BASE/'configs.json').read_text()) if (BASE/'configs.json').exists() else []
 manifest=dict(schema='macnlops-million-streamed-validation-v1',created_utc=now(),inputs={},
  definitions=dict(first_reference='same first-emission proposals before removal/veto',full_reference='uninterrupted MC@NLO shower from original common source with normal global-recoil mode 2',
  full_input='unrejected first-emission endpoints restarted with the same local recoil and scale import as MAcNLOPS',
  full_macnlops='pre-existing accepted subset of those endpoint continuations; no decision is regenerated',
  errors='signed sumw/sumw2; all per-bin cross-covariances including direct/staged source identity pairing',
  jets='anti-kT R=0.4 on stable light quarks/gluons, excluding top; jet rapidity requires pT>20GeV',
  normalization='original signed common-source pb weights, no survivor or visible-range rescaling',
  no_hepmc=True,hadronisation=False,decays=False,mpi=False,primordialkt=False,qed=False),
  cpu_slots=CONFIG['cpu_slots'])
 if (BASE/'manifest.json').exists():manifest=json.loads((BASE/'manifest.json').read_text())
 manifest['definitions']['single_top_objects']='Stable top plus leading inclusive anti-kT R=0.4 jet (spectator proxy); jet_pt and jet_pt_soft are the second inclusive jet, zero if absent. No analysis acceptance cuts except pT>20GeV for the second-jet rapidity.'
 manifest['definitions']['wbb_objects']='Stable W+, b and bbar in the four-flavour scheme; angular observables use the b/bbar pair. Radiation jets exclude the stable bottom partons before anti-kT R=0.4 clustering; all_jet plots include them. Both Wbb-system and bottom-pair recoil, mass and rapidity are recorded.'
 manifest['definitions']['wjet_objects']='Stable W+ plus leading anti-kT R=0.5 jet with pT>25 GeV and |y|<4.5; require at least one such jet independently at each shower level. Second-jet pT includes softer jets within |y|<4.5, with zero if absent; its rapidity requires pT>25 GeV. Rates use the fiducial selection; raw event totals are retained for source checks.'
 manifest['fks_settings']=CONFIG['fks_settings']
 for key,(process,folder) in PROCESSES.items():
  if keys and key not in keys:continue
  for mode in ['ordinary','delta']:
   if modes and mode not in modes:continue
   p=next(p for p in CONFIG['processes'] if p['key']==key)
   checkpoint_path=PRODUCTION/(key+'_'+mode+'_complete.json')
   if not checkpoint_path.exists():continue
   checkpoint=json.loads(checkpoint_path.read_text())
   if checkpoint.get('contract')!=campaign.contract_id(CONFIG,p):raise RuntimeError('Stale or foreign source checkpoint')
   expected_policy=campaign.veto_policy(mode)
   if checkpoint.get('veto_prescription')!=expected_policy:raise RuntimeError('Source uses the old matching prescription')
   accepted=Path(checkpoint['macnlops']['path']);source=Path(checkpoint['source']['path']);job=BASE/'jobs'/(process+'_'+mode);job.mkdir(parents=True,exist_ok=True)
   for path,expected in [(source,checkpoint['source']['sha256']),(accepted,checkpoint['macnlops']['sha256'])]:
    if mc.sha256(path)!=expected:raise RuntimeError('Completed source/sample hash changed: '+str(path))
   with mc.LHEStream(accepted) as stream:
    profile=mc.read_shower_profile(stream.header)
    first=mc.Event(next(iter(stream)));first.require_continuation()
   if process=='wbb' and (profile['support']['flavour_count']!=4 or profile['support']['n_gluon_to_quark']!=4):
    raise RuntimeError('Wbb validation requires its measured four-flavour shower profile')
   endpoint=job/'endpoint.cmnd'
   environment=mc.write_continuation_settings(PRODUCTION/'processes'/folder,PYTHIA,accepted,dict(shower_profile=profile),endpoint)
   physical=endpoint.read_text()
   endpoint.write_text(physical+'\nTimeShower:globalRecoil = off\nTimeShower:limitPTmaxGlobal = on\nTimeShower:pTmaxMatch = 1\nSpaceShower:pTmaxMatch = 1\nTimeShower:pTmaxFudge = 1.\nSpaceShower:pTmaxFudge = 1.\n')
   direct=job/'source.cmnd'
   direct.write_text(physical+'\nTimeShower:globalRecoil = on\nTimeShower:globalRecoilMode = 2\nTimeShower:nMaxGlobalRecoil = 1\nTimeShower:nMaxGlobalBranch = 1\nTimeShower:nPartonsInBorn = %d\n'%p['born_partons'])
   reports=json.loads(accepted.with_name('summary.json').read_text());channels=[]
   for i,record in enumerate(reports):
    if record.get('status')!='complete':raise RuntimeError('Incomplete matching channel')
    channel=Path(record['output']).parent;report=json.loads((channel/'report.json').read_text())
    n=report['counts'].get('input_H',0)+report['counts'].get('input_S',0)
    channels.append(dict(input=str(channel/'proposals.lhe'),decisions=str(channel/'events.lhe.gz.decisions.jsonl'),events=n,seed=checkpoint['settings']['iseed']+900000+i))
   if sum(c['events'] for c in channels)!=CONFIG['events_per_source']:raise RuntimeError('Wrong source population')
   common=dict(process=process,mode=mode,delta=mode=='delta',source=str(source),source_settings=str(direct),endpoint_settings=str(endpoint),
    events=CONFIG['events_per_source'],accepted_events=checkpoint['macnlops']['counts']['events'],seed=checkpoint['settings']['iseed']+1800000,
    channels=channels,observables=[dict(name=name,edges=list(spec.edges),label=spec.label,unit=spec.unit,log=spec.logarithmic) for name,spec in specs_for(process).items()],
    expected_source=checkpoint['source'],expected_accepted=checkpoint['macnlops'],environment={k:environment[k] for k in ['LD_LIBRARY_PATH','LHAPDF_DATA_PATH','PYTHIA8DATA']},
    profile_sha256=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest(),
    source_contract=checkpoint['contract'],veto_prescription=expected_policy,
    analysis_jets=p.get('analysis_jets',dict(algorithm='antikt',radius=.4)),consumer_sha256=mc.sha256(CONSUMER))
   (job/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
   for level in ['first','full']:
    cfg=dict(common,level=level,output=str(job/(level+'.json')))
    path=job/(level+'_config.json')
    cfg['render_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if path.exists() and json.loads(path.read_text())!=cfg:raise RuntimeError('Analysis configuration changed; use a fresh analysis directory: '+str(path))
    dump(path,cfg)
    if str(path) not in configs:configs.append(str(path))
   manifest['inputs'][process+'_'+mode]=dict(source=str(source),source_sha256=checkpoint['source']['sha256'],accepted=str(accepted),accepted_sha256=checkpoint['macnlops']['sha256'],accepted_events=common['accepted_events'])
 dump(BASE/'manifest.json',manifest);dump(BASE/'configs.json',configs)
 print('Registered %d validation jobs from completed common sources.'%len(configs),flush=True)
 return configs

def load(config_path):
 config=json.loads(Path(config_path).read_text());data=json.loads(Path(config['output']).read_text())
 if config['consumer_sha256']!=mc.sha256(CONSUMER):raise RuntimeError('Consumer changed; use a fresh analysis directory')
 if not data.get('complete') or data['events']!=config['events'] or data['accepted']!=config['accepted_events']:raise RuntimeError('Incomplete histogram output')
 for stage,expected in [('reference',config['expected_source']),('input',config['expected_source']),('macnlops',config['expected_accepted'])]:
  actual=data['totals'][stage]
  if actual['events']!=expected['counts']['events']:raise RuntimeError('Event total mismatch for '+stage)
  if abs(actual['sumw']-expected['sumw'])>1.e-9*max(1.,expected['sumabs']):raise RuntimeError('Weight total mismatch for '+stage)
 return config,data

COLORS={'reference':'#111111','input':'#0072B2','h_only':'#D55E00','s_only':'#009E73','macnlops':'#AA3377'}
LINESTYLES={'reference':'-','input':':','h_only':'--','s_only':'-.','macnlops':'-'}

def render(config_path):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 config,data=load(config_path);process=config['process'];level=config['level'];mode=config['mode'];directory=BASE/level/(process+'_'+mode);directory.mkdir(parents=True,exist_ok=True)
 labels=dict(reference='MC@NLO'+('-Δ' if mode=='delta' else ''),input='Before removal/veto',h_only='H removal only',s_only='S veto only',macnlops='MAcNLOPS'+(' + Δ' if mode=='delta' else ''))
 if level=='full':labels['input']='Unrejected LHE restart'
 specs=specs_for(process);arrays={};flows={};figures=[]
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'savefig.dpi':145})
 for name,spec in specs.items():
  raw=data['histograms'][name];edges=np.asarray(raw['edges']);n=len(edges)-1
  sums=np.asarray(raw['sumw']).reshape(len(STAGES),n);vars=np.asarray(raw['sumw2']).reshape(len(STAGES),n);absw=np.asarray(raw['sumabsw']).reshape(len(STAGES),n);cov=np.asarray(raw['covariance']).reshape(len(STAGES),len(STAGES),n)
  for field,values in [('sumw',sums),('sumw2',vars),('sumabsw',absw),('covariance',cov),('count',np.asarray(raw['count']).reshape(len(STAGES),n)),('edges',edges)]:arrays[name+'_'+field]=values
  flows[name]=raw['flows'];centers=(edges[:-1]+edges[1:])/2;widths=np.diff(edges)
  shown=sums.copy();shown[np.abs(shown)<=8*np.finfo(float).eps*absw]=0.
  fig,(ax,ratio_ax)=plt.subplots(2,1,figsize=(8.2,7.1),sharex=True,gridspec_kw={'height_ratios':[3.1,1.2],'hspace':.05})
  roles=['reference','input','h_only','s_only','macnlops'] if level=='full' else ['reference','h_only','s_only','macnlops']
  visible=np.vstack([shown[STAGES.index(role)]/widths for role in roles])
  positive_only=not np.any(visible<0.)
  for role in roles:
   i=STAGES.index(role);color=COLORS[role];y=shown[i]/widths;error=np.sqrt(vars[i])/widths
   ax.stairs(y,edges,color=color,label=labels[role],linewidth=1.45,linestyle=LINESTYLES[role],baseline=None)
   band_mask=(y-error>0.) if spec.logarithmic and positive_only else np.isfinite(y)
   ax.fill_between(centers,y-error,y+error,where=band_mask,color=color,alpha=.12)
  if spec.logarithmic:
   positive=np.abs(visible[np.isfinite(visible)&(visible!=0)])
   if positive_only and positive.size:
    ax.set_yscale('log');ax.set_ylim(float(np.min(positive))*.55,float(np.max(positive))*1.8)
   else:
    threshold=max(1.e-12,float(np.min(positive))*.25) if positive.size else 1.e-6
    ax.set_yscale('symlog',linthresh=threshold)
   if name in ('system_pt','delta_phi_deficit'):
    ax.set_xscale('symlog',linthresh=edges[1])
  if process=='ttbar' and name in ('jet_pt_soft','system_pt_soft'):ax.set_xlim(edges[0],edges[-1])
  ax.axhline(0,color='.6',lw=.4)
  denominator=STAGES.index('macnlops')
  baseline=shown[denominator];baselinevar=vars[denominator];csv_ratios={}
  for role in roles:
   i=STAGES.index(role);ratio,var=prior.ratio_with_variance(shown[i],baseline,vars[i],baselinevar,cov[i,denominator]);csv_ratios[role]=(ratio,var)
   ratio_ax.stairs(ratio,edges,color=COLORS[role],linestyle=LINESTYLES[role],baseline=None,lw=1.2)
   error=np.sqrt(var);ratio_ax.fill_between(centers,ratio-error,ratio+error,color=COLORS[role],alpha=.14)
  ratio_ax.axhline(1,color='.4',lw=.6);ratio_ax.set_ylabel('/ MAcNLOPS');ratio_ax.set_xlabel(spec.label+(' ['+spec.unit+']' if spec.unit else ''))
  ax.set_ylabel(r'$d\sigma/dx$ [pb'+('/'+spec.unit if spec.unit else '')+']');ax.legend(frameon=False)
  ax.set_title('%s · %s · %s\n%s common source events; %s retained'%(PROCESS_LABELS[process], 'MC@NLO-Δ' if mode=='delta' else 'ordinary MC@NLO','first emission' if level=='first' else 'full parton shower',format(data['events'],','),format(data['accepted'],',')))
  for axis in [ax,ratio_ax]:axis.grid(alpha=.18)
  prior.autoscale_ratio_axis(ratio_ax);fig.text(.13,.02,'Original signed weights; ratio bands include paired hard-event covariance',fontsize=8)
  fig.subplots_adjust(left=.13,right=.97,top=.90,bottom=.13)
  for suffix in ['png','pdf']:fig.savefig(directory/(name+'.'+suffix))
  for role,(ratio,variance) in csv_ratios.items():
   below=np.isfinite(ratio)&(ratio<.8);above=np.isfinite(ratio)&(ratio>1.2)
   ratio_ax.plot(centers[below],np.full(np.sum(below),.8),linestyle='none',marker='v',markersize=4,color=COLORS[role],clip_on=False)
   ratio_ax.plot(centers[above],np.full(np.sum(above),1.2),linestyle='none',marker='^',markersize=4,color=COLORS[role],clip_on=False)
  ratio_ax.set_ylim(.8,1.2)
  ratio_ax.text(.01,.96,'Triangles mark ratios outside zoom; unrestricted ratios are in the companion figure',transform=ratio_ax.transAxes,fontsize=6.8,va='top',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
  for suffix in ['png','pdf']:fig.savefig(directory/(name+'_ratio_zoom.'+suffix))
  plt.close(fig);figures.append(name)
  with (directory/(name+'.csv')).open('w') as stream:
   writer=csv.writer(stream);header=['low','high']
   for role in STAGES:header.extend([role+'_sumw',role+'_sumw2',role+'_count'])
   for role in csv_ratios:header.extend([role+'_ratio',role+'_ratio_variance'])
   writer.writerow(header)
   counts=arrays[name+'_count']
   for b in range(n):
    row=[edges[b],edges[b+1]]
    for i in range(len(STAGES)):row.extend([sums[i,b],vars[i,b],counts[i,b]])
    for ratio,var in csv_ratios.values():row.extend([ratio[b],var[b]])
    writer.writerow(row)
  if name in ['jet_pt_soft','jet_pt','system_pt_soft','system_mass']:
   fig,axis=plt.subplots(figsize=(8.2,4.8))
   base=shown[denominator]
   for role in ['h_only','s_only','macnlops']:
    i=STAGES.index(role);r,v=prior.ratio_with_variance(shown[i],base,vars[i],vars[denominator],cov[i,denominator]);axis.stairs(r,edges,color=COLORS[role],label=labels[role],linestyle=LINESTYLES[role],baseline=None);err=np.sqrt(v);axis.fill_between(centers,r-err,r+err,color=COLORS[role],alpha=.12)
   axis.axhline(1,color='.5',lw=.6);axis.set_xlabel(spec.label+(' ['+spec.unit+']' if spec.unit else ''));axis.set_ylabel('/ MAcNLOPS');axis.legend(frameon=False);axis.grid(alpha=.2)
   axis.set_title('%s · %s · %s: removal and veto'%(process,mode,level));fig.tight_layout()
   for suffix in ['png','pdf']:fig.savefig(directory/(name+'_mechanisms.'+suffix))
   plt.close(fig)
 np.savez_compressed(directory/'histograms.npz',**arrays);dump(directory/'flows.json',flows)
 summary=dict(process=process,mode=mode,level=level,complete=True,events=data['events'],retained=data['accepted'],totals=data['fiducial_totals'],raw_totals=data['totals'],analysis_jets=config['analysis_jets'],reference_audit=data['reference_audit'],endpoint_audit=data['endpoint_audit'],plots=figures,config=str(config_path),ratio_denominator='macnlops')
 dump(directory/'summary.json',summary)
 text=['<!doctype html><meta charset="utf-8"><title>MAcNLOPS validation</title><style>body{font:16px system-ui;max-width:1150px;margin:2em auto}img{max-width:48%;vertical-align:top}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:.4em;border:1px solid #bbb}</style>',f'<h1>{process} · {mode} · {level}</h1>', '<p>One million common source events. Original signed cross-section weights; no normalization to survivors or plotted range. Jets: anti-kT R=0.4; no decays, hadronisation, MPI, primordial kT or QED radiation.</p>', '<p>At full shower, the black curve is uninterrupted MC@NLO; the blue curve restarts the unrejected first-step LHE; the magenta curve uses the frozen MAcNLOPS accepted subset of those same continued events. The orange dashed and green dash-dot curves apply only H removal and only S veto, respectively, to the same first-step sample. Every ratio uses the corresponding combined MAcNLOPS curve as denominator. Ratio errors include the paired covariance with that denominator; zero-denominator bins remain undefined.</p>','<p><a href="histograms.npz">Histograms and covariance</a> · <a href="flows.json">Underflow, overflow, missing and zero entries</a> · <a href="summary.json">Counts and shower checks</a></p>']
 selection=('At least one anti-kT R=0.5 jet with pT&gt;25 GeV and |y|&lt;4.5; all curves are fiducial. Additional-jet pT includes softer central jets.' if process=='wjet' else 'Jets: anti-kT R=0.4; jet rapidity requires pT&gt;20 GeV.')
 text[2]='<p>%s common source events. Original signed weights with no survivor rescaling. %s No decays, hadronisation, MPI, primordial kT or QED radiation.</p>'%(format(data['events'],','),selection)
 for name in figures:text.append('<div><a href="%s.pdf"><img src="%s.png" alt="%s"></a><a href="%s_ratio_zoom.pdf"><img src="%s_ratio_zoom.png" alt="ratio zoom"></a></div>'%(name,name,name,name,name))
 for name in ['jet_pt_soft','jet_pt','system_pt_soft','system_mass']:text.append('<a href="%s_mechanisms.pdf"><img src="%s_mechanisms.png"></a>'%(name,name))
 (directory/'index.html').write_text('\n'.join(text));return str(directory)

def index():
 rows=[]
 for key,(process,folder) in PROCESSES.items():
  for mode in ['ordinary','delta']:
   cells=[html.escape(key),mode]
   for level in ['first','full']:
    directory=BASE/level/(process+'_'+mode);path=directory/'summary.json';cfg=BASE/'jobs'/(process+'_'+mode)/(level+'_config.json');progress=BASE/'jobs'/(process+'_'+mode)/(level+'.json.progress.json')
    if path.exists():
     s=json.loads(path.read_text());cells.append('<a href="%s/%s_%s/index.html">Complete: %s source / %s retained</a>'%(level,process,mode,format(s['events'],','),format(s['retained'],',')))
    elif progress.exists():
     s=json.loads(progress.read_text());cells.append(html.escape('%s: %s/%s'%(s['phase'],format(s['events'],','),format(s['target'],','))))
    else:cells.append('Queued')
   rows.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
 (BASE/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MAcNLOPS validation</title><style>body{font:16px system-ui;margin:2em}table{border-collapse:collapse}td,th{padding:.6em;border:1px solid #aaa}</style><h1>MAcNLOPS validation: ZZ, DY, top pairs, single top, Wbb and W+jet</h1><p>Updated '+now()+' · '+format(CONFIG['events_per_source'],',')+' common hard-source events per process and source mode. Histograms are filled directly; no HepMC files are written.</p><table><tr><th>Process</th><th>Source</th><th>First emission</th><th>Full shower</th></tr>'+''.join(rows)+'</table><p><a href="manifest.json">Definitions and source provenance</a></p>')

def execute(path,cpus):
 config=json.loads(Path(path).read_text());out=Path(config['output']);job=out.parent;state=job/(config['level']+'_status.json');log=job/(config['level']+'.log')
 if out.exists():load(path);return dict(config=path,status='complete',cached=True)
 dump(state,dict(status='running',started_utc=now(),cpus=cpus,config=path,log=str(log)))
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');env.update(config['environment'])
 with log.open('w') as stream:result=subprocess.run(['nice','-n','19','taskset','-c',','.join(map(str,cpus)),str(BASE/'build/StreamHist'),str(path)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError('Histogram consumer failed for '+str(path)+'; see '+str(log))
 load(path);dump(state,dict(status='complete',finished_utc=now(),cpus=cpus,config=path,log=str(log)));return dict(config=path,status='complete')

def run():
 lock=(BASE/'analysis.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);(BASE/'analysis.pid').write_text(str(os.getpid())+'\n')
 configs=json.loads((BASE/'configs.json').read_text());results=[]
 if not configs:raise RuntimeError('No completed production samples registered; run prepare after production')
 build()
 for level in ['first','full']:
  pending=[p for p in configs if json.loads(Path(p).read_text())['level']==level]
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(CONFIG['cpu_slots'])) as pool:
   active={};free=list(CONFIG['cpu_slots'])
   while pending or active:
    while pending and free:
     path=pending.pop(0);cpu=free.pop(0);active[pool.submit(execute,path,cpu)]=(path,cpu)
    if not active:break
    done,_=concurrent.futures.wait(active,timeout=15,return_when=concurrent.futures.FIRST_COMPLETED)
    for future in done:
     path,cpu=active.pop(future);free.append(cpu)
     try:
      report=future.result();render(path);results.append(report);print(now(),'COMPLETE',path,flush=True)
     except Exception as exc:
      results.append(dict(config=path,status='failed',error=str(exc)));dump(Path(path).with_name(level+'_status.json'),dict(status='failed',error=str(exc)));traceback.print_exc()
    dump(BASE/'status.json',dict(updated_utc=now(),phase=level,active=[dict(config=p,cpu=c) for p,c in active.values()],queued=pending,results=results));index()
  if any(r['status']=='failed' for r in results):raise RuntimeError('A validation job failed; full results are not certified')
 dump(BASE/'status.json',dict(updated_utc=now(),phase='complete',active=[],queued=[],results=results));index()

def main(argv=None):
 global BASE,PRODUCTION,CONFIG,PYTHIA
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('action',choices=['plan','prepare','build','run','render','status'])
 parser.add_argument('config',nargs='?')
 parser.add_argument('--campaign-dir',type=Path,default=campaign.DEFAULT_BASE)
 parser.add_argument('--analysis-dir',type=Path)
 args=parser.parse_args(argv)
 PRODUCTION=args.campaign_dir.resolve();BASE=(args.analysis_dir or PRODUCTION/'validation').resolve()
 CONFIG=json.loads((PRODUCTION/'config.json').read_text());campaign.validate_config(CONFIG,PRODUCTION,check_physics=args.action!='status')
 PYTHIA=Path(CONFIG['pythia8']);BASE.mkdir(parents=True,exist_ok=True)
 if args.action=='plan':
  dump(BASE/'analysis_plan.json',analysis_plan());print(BASE/'analysis_plan.json')
 elif args.action=='prepare':prepare();index()
 elif args.action=='build':print(build())
 elif args.action=='render':print(render(args.config));index()
 elif args.action=='run':run()
 else:
  path=BASE/'status.json';print(path.read_text() if path.exists() else 'Not started');index()
 return 0

if __name__=='__main__':sys.exit(main())
