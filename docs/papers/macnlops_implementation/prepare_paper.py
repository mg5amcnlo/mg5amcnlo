#!/usr/bin/env python3
"""Freeze the completed campaign, generate tables, and collect every plot.

This only reads completed artifacts. It neither generates events nor showers.
The LaTeX source bundle builds without the original event files or this script.
"""
from pathlib import Path
import collections,datetime,hashlib,json,shutil,subprocess
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
VALID=ROOT/'validation_1M_20261023'
PROD=ROOT/'production_1M_20261023'
ORDER=['zz','dy','ttbar','single_top','wbb']
KEYS={'zz':'ZZ','dy':'DY','ttbar':'ttbar','single_top':'single_top','wbb':'Wbb'}
LABELS={'zz':r'$ZZ$','dy':r'DY','ttbar':r'$t\bar t$','single_top':r'$tj$','wbb':r'$W^+b\bar b$'}
NAMES={'zz':'ZZ (5FS)','dy':'Drell-Yan (5FS)','ttbar':'Top pair (5FS)','single_top':'Single top (5FS)','wbb':'Wbb (4FS)'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def tex(value):
    result=str(value)
    for old,new in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:result=result.replace(old,new)
    return result
def table(path,columns,header,rows,caption,label):
    text=[r'\begin{table}[tbp]',r'\centering\small',r'\begin{tabular}{'+columns+'}',r'\toprule',header+r'\\',r'\midrule']
    text.extend(row+r'\\' for row in rows)
    text.extend([r'\bottomrule',r'\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{table}'])
    path.write_text('\n'.join(text)+'\n')

def main():
    generated=BASE/'generated';generated.mkdir(exist_ok=True)
    cohorts={};figures=[];claims=[]
    for level in ['first','full']:
        checks=json.loads((VALID/level/'checks.json').read_text())
        assert len(checks)==10 and all(x['all_rate_flow_selection_covariance_checks_pass'] for x in checks)
        for process in ORDER:
            for mode in ['ordinary','delta']:
                key=process+'_'+mode;directory=VALID/level/key
                summary=json.loads((directory/'summary.json').read_text())
                assert summary['complete'] and summary['events']==1000000 and summary['ratio_denominator']=='macnlops'
                checkpoint=json.loads((PROD/(KEYS[process]+'_'+mode+'_complete.json')).read_text())
                cfg=json.loads((VALID/'jobs'/key/(level+'_config.json')).read_text())
                native=json.loads(Path(cfg['output']).read_text())
                assert native['complete'] and native['events']==1000000
                assert native['accepted']==checkpoint['macnlops']['counts']['events']
                for stage,expected in [('reference',checkpoint['source']),('input',checkpoint['source']),('macnlops',checkpoint['macnlops'])]:
                    actual=native['totals'][stage]
                    assert actual['events']==expected['counts']['events']
                    assert abs(actual['sumw']-expected['sumw'])<1.e-8*max(1.,expected['sumabs'])
                if level=='first':
                    cohorts[key]=dict(process=process,mode=mode,checkpoint=checkpoint,
                        first_totals=native['totals'],first_config=cfg,
                        profile=json.loads((VALID/'jobs'/key/'profile.json').read_text()))
                else:
                    cohorts[key].update(full_totals=native['totals'],full_audits=dict(reference=native['reference_audit'],endpoint=native['endpoint_audit']))
                    for stage in native['totals']:
                        before=cohorts[key]['first_totals'][stage];after=native['totals'][stage]
                        assert before['events']==after['events']
                        assert abs(before['sumw']-after['sumw'])<1.e-8*max(1.,before['sumabsw'])
                expected_names=set()
                for observable in summary['plots']:
                    expected_names.update([observable+'.pdf',observable+'_ratio_zoom.pdf'])
                expected_names.update(name+'_mechanisms.pdf' for name in ['jet_pt_soft','jet_pt','system_pt_soft','system_mass'])
                assert {p.name for p in directory.glob('*.pdf')}==expected_names
                for name in sorted(expected_names):
                    source=directory/name;target=BASE/'figures'/level/key/name
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                    variant='ratio zoom' if '_ratio_zoom' in name else 'mechanisms' if '_mechanisms' in name else 'standard'
                    observable=name.removesuffix('.pdf').removesuffix('_ratio_zoom').removesuffix('_mechanisms')
                    figures.append(dict(level=level,process=process,mode=mode,observable=observable,variant=variant,
                        source=str(source.relative_to(ROOT)),file=str(target.relative_to(BASE)),sha256=sha(target),historical=False))
                claims.append(dict(cohort=key,level=level,events=summary['events'],retained=summary['retained'],count_weight_covariance_checks_passed=True))
    assert len(figures)==696
    historical=ROOT/'docs/papers/macnlops_common_source/figures'
    for source in sorted(historical.glob('*.pdf')):
        target=BASE/'figures/historical'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        figures.append(dict(file=str(target.relative_to(BASE)),source=str(source.relative_to(ROOT)),sha256=sha(target),historical=True,
                            title=source.stem.replace('_',' ')))
    assert len(figures)==703
    dump(BASE/'paper_data.json',cohorts)
    dump(BASE/'figure_manifest.json',dict(current_figures=696,historical_figures=7,figures=figures))
    rate_rows=[];negative_rows=[];effects=[]
    for process in ORDER:
        for mode in ['ordinary','delta']:
            item=cohorts[process+'_'+mode];d=item['checkpoint'];s=d['source'];m=d['macnlops'];t=item['first_totals']
            name=LABELS[process];mode_tex=r'$\Delta$' if mode=='delta' else 'ordinary'
            rate_rows.append('%s & %s & %s & %.6g & %.6g & %+.3f'%(name,mode_tex,f"{m['counts']['events']:,}",s['sumw'],m['sumw'],100*(m['sumw']/s['sumw']-1)))
            negative_rows.append('%s & %s & %s & %s & %s & %s'%(name,mode_tex,
                f"{s['counts'].get('negative_H',0):,}",f"{m['counts'].get('negative_H',0):,}",
                f"{s['counts'].get('negative_S',0):,}",f"{m['counts'].get('negative_S',0):,}"))
            effects.append('%s & %s & %+.5g & %+.5g & %+.5g'%(name,mode_tex,t['h_only']['sumw']-t['input']['sumw'],t['s_only']['sumw']-t['input']['sumw'],t['macnlops']['sumw']-t['input']['sumw']))
    table(generated/'rates.tex','llrrrr',r'Process & Source & Retained & $\sigma_{\rm MC}$ [pb] & $\sigma_{\rm MAc}$ [pb] & Change [\%]',rate_rows,
        r'Completed million-event common-source samples. Every source has $10^6$ events; retained events keep their original signed weights. Cross sections are the sums of the event weights, rather than a renormalization to the accepted count. The first-emission and full-shower sums agree to numerical precision.','tab:rates')
    table(generated/'negatives.tex','llrrrr',r'Process & Source & $N^-_{H,\rm source}$ & $N^-_{H,\rm retained}$ & $N^-_{S,\rm source}$ & $N^-_{S,\rm retained}$',negative_rows,
        r'Negative H and S event counts before and after the frozen decisions. These are counts, not integrals of negative densities. A zero in this finite sample does not prove that the residual vanishes throughout phase space.','tab:negatives')
    table(generated/'effects.tex','llrrr',r'Process & Source & H removal [pb] & S veto [pb] & Combined [pb]',effects,
        r'Changes of the signed cross section from H removal alone, S rejection alone, and their combination. The first two columns add to the last; rejected negative S weights contribute with the opposite sign to rejected positive S weights. No events are rescaled.','tab:effects')
    # Catalog every source figure once; overview PDFs are duplicates of these panels.
    atlas=[];last_group=None
    for i,item in enumerate(figures,1):
        if item['historical']:
            group='Historical 20 October 2026: patched-runtime study';title=item['title']
        else:
            group=('First emission' if item['level']=='first' else 'Full parton shower')+' / '+NAMES[item['process']]+' / '+item['mode']
            title=item['observable'].replace('_',' ')+' / '+item['variant']
        commands=[r'\thispagestyle{empty}']
        if item['historical']:commands.append(r'\HistoricalStamp')
        if group!=last_group:commands.append(r'\pdfbookmark[1]{'+tex(group)+'}{atlas-group-'+str(i)+'}');last_group=group
        commands.append(r'\pdfbookmark[2]{'+tex(title)+'}{atlas-figure-'+str(i)+'}')
        atlas.append(r'\includepdf[pages=1,fitpaper=true,pagecommand={'+''.join(commands)+r'}]{'+item['file']+'}')
    (generated/'all_figures.tex').write_text('\n'.join(atlas)+'\n')
    # A manageable main-text selection; the complete atlas contains every variant.
    selections=[]
    for process in ORDER:
        selections.extend([(process,'ordinary','first','jet_pt_soft'),(process,'ordinary','full','jet_pt_soft'),(process,'delta','full','jet_pt_soft')])
    for process,mode,level,observable in selections:
        name=observable+'_ratio_zoom' if process=='ttbar' else observable
        relative='figures/'+level+'/'+process+'_'+mode+'/'+name+'.pdf'
        caption=(NAMES[process]+', '+('ordinary MC@NLO' if mode=='ordinary' else r'MC@NLO-$\Delta$')+', '+
            ('first-emission proposal endpoint' if level=='first' else 'full parton shower')+'. ')
        if process=='wbb':caption+=r'The radiation jets exclude stable bottom partons before clustering. '
        if process=='single_top':caption+=r'The second inclusive jet is shown; the leading jet is the spectator proxy. '
        if process=='ttbar':caption+=r'The displayed soft range is $0$--$150\GeV$ with uniform $1\GeV$ bins. Triangles mark ratios outside the displayed zoom. '
        else:caption+=r'The soft range is $0$--$50\GeV$ with uniform $0.5\GeV$ bins. '
        if level=='first':caption+=r'The black curve is the complete unrejected first-step proposal control, not a separately initialized default-import shower. '
        else:caption+=r'The black curve is the uninterrupted reference; blue is the unrejected staged restart. '
        caption+=r'Orange and green isolate H removal and S veto. All ratios use the corresponding combined MAcNLOPS distribution as denominator. Original signed cross-section weights are preserved.'
        text='\n'.join([r'\begin{figure}[p]',r'\centering',r'\includegraphics[width=.96\textwidth,height=.72\textheight,keepaspectratio]{'+relative+'}',r'\caption{'+caption+'}',r'\label{fig:'+process+'-'+mode+'-'+level+'}',r'\end{figure}'])+'\n'
        (generated/('figure_'+process+'_'+mode+'_'+level+'.tex')).write_text(text)
    sources=['madgraph/various/macnlops.py','madgraph/various/banner.py','madgraph/interface/amcatnlo_run_interface.py',
        'Template/NLO/MCatNLO/srcPythia8/MacNLOPS.cc','Template/NLO/SubProcesses/driver_macnlops.f',
        'Template/NLO/SubProcesses/mc_matching_runtime.f','Template/NLO/SubProcesses/montecarlocounter.f',
        'Template/NLO/SubProcesses/genps_fks.f','Template/NLO/SubProcesses/genps_fks_radiation.f',
        'validation_1M_20261023/analysis.py','validation_1M_20261023/src/StreamHist.cc',
        'validation_1M_20261023/check_histograms.py','production_1M_20261023/campaign.py']
    provenance=dict(prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        working_tree_is_modified=True,source_hashes={name:sha(ROOT/name) for name in sources if (ROOT/name).exists()},
        checks=claims,figure_count=703,current_figure_count=696,historical_figure_count=7,
        scope='Completed 23--24 October 2026 million-event campaign plus explicitly historical seven figure panels',
        no_new_events_or_showers=True)
    dump(BASE/'evidence_manifest.json',provenance)
    (BASE/'implementation_changes.patch').write_bytes(subprocess.check_output(['git','diff','--',
        'Template/NLO','madgraph/various/macnlops.py','madgraph/various/banner.py','madgraph/interface/amcatnlo_run_interface.py'],cwd=ROOT))
    print('Frozen 10 completed common-source cohorts, 20 validated levels, 696 current plots and 7 historical figures.')

if __name__=='__main__':main()
