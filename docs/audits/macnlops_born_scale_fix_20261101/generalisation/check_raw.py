from pathlib import Path
import collections,hashlib,itertools,json,re,sys,numpy as np
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS');sys.path.insert(0,str(ROOT));from madgraph.various import macnlops
BASE=Path(__file__).resolve().parent
manifest=json.loads((BASE/'baseline_manifest.json').read_text());results=[];failures=[]
inv=['pt','matching_scale','r','d','k','sector','orientation','candidates','marking','total_k','alpha_s_shower','alpha_s_kernel','pdf_ratio','delta_source_mean','delta_shower_mean','starting_cdf']
def attrs(block):return macnlops.xml_attributes(re.search(r'<macnlops_decision\b([^>]*?)/>',block)[1],'macnlops_decision')
def clean(block):
 block=re.sub(r'<macnlops_decision\b[^>]*?/>','',block)
 return re.sub(r"(<wgt\s+id=['\"]macnlops['\"]>).*?(</wgt>)",r'\1WEIGHT\2',block)
for p in manifest['processes']:
 if not p['samples']:continue
 count=0;g=[];gc=[];gp=[];changed=collections.Counter();rule=collections.Counter();errors=collections.Counter();maxscale=0.;exactee=True;protected=None
 for sample in p['samples']:
  relative=Path(p['key'])/sample['provider']/sample['channel']/'events_macnlops.lhe';oldpath=BASE/'baseline'/relative;newpath=BASE/'corrected'/relative
  with macnlops.LHEStream(oldpath) as before,macnlops.LHEStream(newpath) as after:
   for number,(old,new) in enumerate(itertools.zip_longest(before,after),1):
    assert old is not None and new is not None
    a=attrs(old);b=attrs(new);count+=1;rule[b['rule']]+=1
    if clean(old)!=clean(new):errors['event_bytes_except_decision_and_match_weight']+=1
    for key in inv:
     if a.get(key)!=b.get(key):errors[key]+=1
    if a['keep']!=b['keep']:changed['keep']+=1
    if a['accepted']!=b['accepted']:changed['accepted']+=1
    if 'born_scale_factor' in b:
     G=float(b['born_scale_factor']);g.append(G);gc.append(float(b['born_coupling_factor']));gp.append(float(b['born_pdf_factor']))
     if G!=float(b['born_coupling_factor'])*float(b['born_pdf_factor']):errors['factor_product_not_bit_exact']+=1
     if 'kernel_scale' in a and G>0:
      r=float(b['kernel_scale'])/(float(a['kernel_scale'])*G);maxscale=max(maxscale,abs(r-1))
    if p['key'].startswith('EE'):
     if 'born_scale_factor' in b and any(float(b[k])!=1 for k in ['born_scale_factor','born_coupling_factor','born_pdf_factor']):exactee=False
     if a['keep']!=b['keep'] or a['accepted']!=b['accepted']:errors['fixed_ee_decision_changed']+=1
    if p['key']=='Wbb' and sample['channel']=='GF1.0_26' and number==5294:
     protected=dict(before=a,after=b,physical_event_bytes_unchanged=clean(old)==clean(new));assert a['rule']==b['rule']=='pdf_support_no_transfer' and b['keep']=='1'
 assert count==sum(s['events'] for s in p['samples'])
 result=dict(process=p['key'],events=count,changes=dict(changed),invariant_failures=dict(errors),rules=dict(rule),factors_recorded=len(g),
   G_quantiles=np.quantile(g,[0,.01,.5,.99,1]).tolist() if g else [],coupling_quantiles=np.quantile(gc,[0,.01,.5,.99,1]).tolist() if gc else [],pdf_quantiles=np.quantile(gp,[0,.01,.5,.99,1]).tolist() if gp else [],
   kernel_scale_equals_old_times_G_max_relative_difference=maxscale,fixed_ee_G_and_decisions_exact=exactee if p['key'].startswith('EE') else None,protected_pdf_point=protected)
 results.append(result);print(json.dumps(result),flush=True)
 if errors or maxscale>1e-13:failures.append(p['key'])
output=dict(passed=not failures,failed_processes=failures,processes=results,
  total_events=sum(r['events'] for r in results),method='Exact event bytes after masking only decision XML and matched weight; exact recorded R/D/K,A,CDF,emission/fixed-order couplings and PDF ratios, native first endpoints/source packets/continuation caps. G and decisions intentionally differ; fixed ee G=1 and every decision must match.',
  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(BASE/'raw_invariance.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n');assert not failures,failures
