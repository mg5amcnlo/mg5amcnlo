#!/usr/bin/env python3
"""Independently audit exported histograms, shared covariance and signed residuals.

Reads existing analysis artifacts and original source weights. It does not
rerun showers, change events, or certify physical-distribution agreement.
"""
import argparse,csv,gzip,hashlib,json,math,re
from pathlib import Path
import numpy as np

TOL=2.e-11
LEGACY_STAGES=('reference','input','h_only','s_only','macnlops','H_positive','H_negative','S_positive','S_negative','H_total')
EXPECTED=LEGACY_STAGES+('H_negative_retained',)
FLOWS=('underflow','overflow','missing')
class AuditFailure(ValueError):pass

def require(condition,message):
    if not condition:raise AuditFailure(message)

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def source_weights(path):
    path=Path(path);before=sha256(path);weights=[];pending=False
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        for line in stream:
            text=line.strip()
            if re.match(r'^<event(?:\s[^>]*)?>$',text):pending=True;continue
            if pending and text and not text.startswith('#'):
                fields=text.split();require(len(fields)>=6,'Malformed source header: '+str(path))
                w=float(fields[2].replace('D','E').replace('d','e'))
                require(math.isfinite(w),'Nonfinite original weight');weights.append(w);pending=False
    require(not pending,'Truncated original source')
    require(sha256(path)==before,'Original source changed during audit')
    return dict(path=str(path.resolve()),sha256=before,events=len(weights),negative_events=sum(w<0 for w in weights),
                sumw=math.fsum(weights),sumw2=math.fsum(w*w for w in weights))

def original_sources(process,common,integration):
    roles=integration.get('processes',{}).get(process,{})
    needed=('primary',) if common else ('primary','reference')
    if common:
        require('reference' not in roles,process+': shared comparison declares a second integration source')
    for role in needed:
        require(isinstance(roles.get(role),dict) and roles[role].get('source'),process+'/'+role+': missing original source anchor')
    original={role:source_weights(roles[role]['source']) for role in needed}
    if common:
        original['reference']=original['primary']
    return original

def audit(directory,allow_incomplete=False):
    root=Path(directory).resolve();summary=json.loads((root/'summary.json').read_text())
    require(summary['schema']=='macnlops-validation-histograms-v1','Unknown histogram schema')
    final=summary.get('complete') is True and summary.get('smoke') is False
    if not allow_incomplete:
        require(final,'An incomplete preview cannot pass the final audit')
    if final:
        require(set(summary['processes'])=={'dy','zz','ttbar','wbb'},'Final report lacks a process')
        require(set(summary['levels'])=={'first','full'},'Final report lacks first/full levels')
        for process,record in summary['processes'].items():
            require(set(record.get('levels',{}))=={'first','full'},process+': final report lacks first/full levels')
    require(summary['normalization']=='unaltered original sum weights, pb','Normalization contract differs')
    require(summary['variance'] in (
        'sumw2; paired covariance within primary, independent reference',
        'sumw2; paired covariance including the shared ordinary MC@NLO baseline',
        'sumw2; reference covariance follows each process common_source_mode'),'Variance contract differs')
    modes=[record.get('configuration',{}).get('common_source_mode',False) for record in summary['processes'].values()]
    require(summary.get('common_source_mode',False)==(bool(modes) and all(modes)),'Global comparison mode differs from processes')
    if modes and all(modes):
        require(summary['variance']=='sumw2; paired covariance including the shared ordinary MC@NLO baseline','Shared baseline variance contract differs')
    result=dict(schema='macnlops-independent-histogram-audit-v1',passed=True,complete_validation=final,
        scope='Raw histogram/source-weight accounting only; no physical-distribution agreement or NLO-accuracy claim',
        summary_sha256=sha256(root/'summary.json'),tolerance=TOL,max_scaled_residual=0.,histograms=0,processes={})
    def close(a,b,scale,where):
        a,b,scale=np.broadcast_arrays(np.asarray(a,dtype=float),np.asarray(b,dtype=float),np.asarray(scale,dtype=float))
        require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)),where+': nonfinite accounting')
        residual=np.abs(a-b)/np.maximum(np.abs(scale),1.e-300)
        maximum=float(np.max(residual)) if residual.size else 0.
        result['max_scaled_residual']=max(result['max_scaled_residual'],maximum)
        require(maximum<=TOL,where+': closure residual '+repr(maximum))
    with (root/'stage_totals.csv').open() as stream:
        total_rows=list(csv.DictReader(stream))
    total_table={(row['process'],row['level'],row['stage']):row for row in total_rows}
    require(len(total_table)==len(total_rows),'Duplicate stage-total CSV row')
    expected_total_keys=set()
    integration=summary.get('native_source_integrals')
    if final:require(integration is not None,'Missing separate native integration metadata')
    if integration is not None:
        require(integration['schema']=='macnlops-source-integration-estimates-v1','Unknown integration metadata')
        with (root/'source_integrals.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        expected=[]
        for process,roles in integration['processes'].items():
            if process not in summary['processes']:continue
            for role,source in roles.items():
                for value in source['declared_init']:
                    expected.append((process,role,str(value['process_id']),float(value['cross_section_pb']),float(value['integration_error_pb'])))
        actual=[(r['process'],r['role'],r['source_process_id'],float(r['native_cross_section_pb']),float(r['native_integration_error_pb'])) for r in rows]
        require(sorted(actual)==sorted(expected),'Integration CSV differs from separate source metadata')
    for process,record in summary['processes'].items():
        configuration=record.get('configuration',{})
        common=configuration.get('common_source_mode',False)
        signed=configuration.get('signed_residuals',configuration.get('allow_signed_residuals',False))
        require(type(common) is bool and type(signed) is bool,process+': invalid comparison/policy flag')
        if common:
            require(configuration.get('source_construction')=='ordinary-mcatnlo-v1',process+': shared source is not ordinary MC@NLO')
            require(not configuration.get('reference'),process+': shared source declares an independent reference')
        if final:
            require(configuration.get('expected_events',200000)==200000,process+': source size is not 200000')
            require(configuration.get('flavours')==(4 if process=='wbb' else 5),process+': flavour scheme differs')
        result['processes'][process]=dict(common_source_mode=common,signed_residuals=signed,levels={},original_sources={})
        for level,data in record['levels'].items():
            where=process+'/'+level;folder=root/process/level
            require(data.get('common_source_mode',False)==common,where+': comparison mode differs from configuration')
            require(data.get('signed_residuals',signed)==signed,where+': signed-residual policy differs from configuration')
            if common:
                a,b=data['chunks']['primary'],data['chunks']['reference']
                require(len(a)==len(b) and len(a)>0,where+': missing derived baseline chunks')
                for original,derived in zip(a,b):
                    require(derived.get('derived_from_primary') is True,where+': baseline is not declared derived')
                    for key in ('path','events','weight_sequence_sha256','source_sequence_sha256','full_audit'):
                        require(original[key]==derived[key],where+': baseline chunk '+key+' differs')
            with np.load(folder/'histograms.npz',allow_pickle=False) as archive:
                raw={key:archive[key] for key in archive.files}
            stages=tuple(raw['stages'].tolist())
            require(stages in (LEGACY_STAGES,EXPECTED),where+': stage definitions differ')
            require(not signed or stages==EXPECTED,where+': signed residual histogram is missing')
            require(set(data['totals'])==set(stages),where+': total stage definitions differ')
            index={name:i for i,name in enumerate(stages)}
            flows=json.loads((folder/'flows.json').read_text())
            require(flows==data['flows'],where+': flow file differs from summary')
            with (folder/'range_flows.csv').open() as stream:
                range_rows=list(csv.DictReader(stream))
            range_table={(row['observable'],row['stage']):row for row in range_rows}
            require(len(range_table)==len(range_rows)==len(data['observables'])*len(stages),where+': range-flow CSV rows differ')
            if final:
                require(data['complete'] is True,where+': incomplete stage')
                require(data['totals']['input']['events']==data['totals']['reference']['events']==200000,where+': final count')
            for stage,total in data['totals'].items():
                key=(process,level,stage);expected_total_keys.add(key)
                require(key in total_table,where+'/'+stage+': missing total CSV row')
                row=total_table[key]
                for field in ('events','negative_events'):
                    require(int(row[field])==total[field],where+'/'+stage+': total CSV '+field)
                for field,column in (('sumw','sumw_pb'),('sumw2','sumw2_pb2'),('effective_events','effective_events')):
                    require(float(row[column])==total[field],where+'/'+stage+': total CSV '+field)
            bin_csv_sha256={}
            for name in data['observables']:
                prefix=name+'__';edges=raw[prefix+'edges'];w=raw[prefix+'sumw'];w2=raw[prefix+'sumw2'];n=raw[prefix+'count'];absolute=raw[prefix+'sumabsw'];cov=raw[prefix+'covariance']
                require(w.shape==w2.shape==n.shape==absolute.shape==(len(stages),len(edges)-1),where+'/'+name+': shape')
                require(np.all(np.diff(edges)>0) and np.all(n>=0) and np.all(n==n.astype(int)),where+'/'+name+': edges/counts')
                require(np.all(w2>=0) and np.all(absolute>=0),where+'/'+name+': negative variance/absolute sum')
                require(cov.shape==(len(stages),len(stages),len(edges)-1),where+'/'+name+': covariance shape')
                require(np.all(cov>=0),where+'/'+name+': negative shared-event covariance')
                displayed=w.copy();displayed[abs(w)<=8*np.finfo(float).eps*absolute]=0.
                require(np.array_equal(raw[prefix+'display_sumw'],displayed),where+'/'+name+': signed display differs from roundoff rule')
                csv_path=folder/(name+'.csv')
                with csv_path.open() as stream:csv_rows=list(csv.DictReader(stream))
                bin_csv_sha256[name]=sha256(csv_path)
                require(len(csv_rows)==len(stages)*(len(edges)-1),where+'/'+name+': bin CSV row count')
                for i,stage in enumerate(stages):
                    block=csv_rows[i*(len(edges)-1):(i+1)*(len(edges)-1)]
                    require(all(row['stage']==stage for row in block),where+'/'+name+': bin CSV stage')
                    require(all(float(row['bin_low'])==edges[j] and float(row['bin_high'])==edges[j+1] for j,row in enumerate(block)),where+'/'+name+': bin CSV edges')
                    require(all(int(row['count'])==n[i,j] for j,row in enumerate(block)),where+'/'+name+': bin CSV count')
                    for column,values,bound in (('sumw_pb',w[i],absolute[i]),('sumw2_pb2',w2[i],w2[i]),('sumabsw_pb',absolute[i],absolute[i])):
                        close([float(row[column]) for row in block],values,bound,where+'/'+name+'/'+stage+' bin CSV '+column)
                    widths=np.diff(edges)
                    close([float(row['density']) for row in block],w[i]/widths,absolute[i]/widths,where+'/'+name+'/'+stage+' CSV density')
                    close([float(row['density_variance']) for row in block],w2[i]/widths**2,w2[i]/widths**2,where+'/'+name+'/'+stage+' CSV density variance')
                for stage,i in index.items():
                    t=data['totals'][stage];f=flows[name][stage]
                    row=range_table.get((name,stage));require(row is not None,where+'/'+name+'/'+stage+': missing range-flow CSV row')
                    require(int(row['source_events'])==t['events'] and int(row['visible_events'])==int(n[i].sum()),where+'/'+name+'/'+stage+': range-flow CSV counts')
                    require(float(row['range_low'])==edges[0] and float(row['range_high'])==edges[-1],where+'/'+name+': range-flow CSV edges')
                    for column,value,bound in (('visible_sumw_pb',w[i].sum(),absolute[i].sum()),('visible_sumw2_pb2',w2[i].sum(),w2[i].sum())):
                        close(float(row[column]),value,bound,where+'/'+name+'/'+stage+' range-flow CSV '+column)
                    for flow in FLOWS:
                        require(int(row[flow+'_events'])==f[flow]['count'],where+'/'+name+'/'+stage+': range-flow CSV '+flow+' count')
                        for field,suffix in (('sumw','sumw_pb'),('sumw2','sumw2_pb2')):
                            close(float(row[flow+'_'+suffix]),f[flow][field],max(abs(f[flow][field]),1.e-300),where+'/'+name+'/'+stage+' range-flow CSV '+flow+' '+field)
                    require(int(n[i].sum())+sum(f[k]['count'] for k in FLOWS)==t['events'],where+'/'+name+'/'+stage+': count closure')
                    bound=float(absolute[i].sum())+sum(math.sqrt(f[k]['count']*f[k]['sumw2']) for k in FLOWS)
                    close(w[i].sum()+sum(f[k]['sumw'] for k in FLOWS),t['sumw'],bound,where+'/'+name+'/'+stage+' sumw')
                    close(w2[i].sum()+sum(f[k]['sumw2'] for k in FLOWS),t['sumw2'],t['sumw2'],where+'/'+name+'/'+stage+' sumw2')
                    close(cov[i,i],w2[i],w2[i],where+'/'+name+'/'+stage+' covariance diagonal')
                    neff=t['sumw']**2/t['sumw2'] if t['sumw2'] else 0.
                    close(neff,t['effective_events'],max(neff,1.),where+'/'+stage+' N_eff')
                close(cov,cov.swapaxes(0,1),np.maximum(np.abs(cov),np.abs(cov.swapaxes(0,1))),where+'/'+name+' symmetric covariance')
                if common:
                    for field,values in (('count',n),('sumw',w),('sumw2',w2),('sumabsw',absolute)):
                        close(values[0],values[1],np.maximum(absolute[1] if field=='sumw' else values[1],1.e-300),where+'/'+name+' identical baseline '+field)
                    close(cov[0],w2,np.maximum(w2,1.e-300),where+'/'+name+' shared reference covariance')
                else:
                    close(cov[0,1:],0.,np.maximum(cov[0,0],1.e-300),where+'/'+name+' independent reference covariance')
                for field,values in (('count',n),('sumw',w),('sumw2',w2)):
                    scale=absolute[index['input']] if field=='sumw' else np.maximum(values[index['input']],1.e-300)
                    close(values[index['input']],sum(values[index[s]] for s in ('H_positive','H_negative','S_positive','S_negative')),scale,where+'/'+name+' source '+field)
                    close(values[index['H_total']],values[index['H_positive']]+values[index['H_negative']],scale,where+'/'+name+' H '+field)
                    residual=values[index['H_negative_retained']] if 'H_negative_retained' in index else 0.
                    close(values[index['h_only']],values[index['input']]-values[index['H_negative']]+residual,scale,where+'/'+name+' H rejection '+field)
                    close(values[index['macnlops']],values[index['h_only']]+values[index['s_only']]-values[index['input']],scale,where+'/'+name+' mechanisms '+field)
                    for flow in FLOWS+('zero',):
                        v={s:flows[name][s][flow][field] for s in stages};bound=sum(abs(v[s]) for s in ('H_positive','H_negative','S_positive','S_negative'))
                        close(v['input'],sum(v[s] for s in ('H_positive','H_negative','S_positive','S_negative')),bound,where+'/'+name+'/'+flow+' source '+field)
                        close(v['H_total'],v['H_positive']+v['H_negative'],bound,where+'/'+name+'/'+flow+' H '+field)
                        residual=v.get('H_negative_retained',0.)
                        close(v['h_only'],v['input']-v['H_negative']+residual,bound,where+'/'+name+'/'+flow+' H rejection '+field)
                        close(v['macnlops'],v['h_only']+v['s_only']-v['input'],bound,where+'/'+name+'/'+flow+' mechanisms '+field)
                        if common:
                            close(v['reference'],v['input'],bound,where+'/'+name+'/'+flow+' identical baseline '+field)
                for stage,sign in (('H_positive',1),('S_positive',1),('H_negative',-1),('S_negative',-1),('H_negative_retained',-1)):
                    if stage not in index:continue
                    require(np.all(sign*w[index[stage]]>=-TOL*absolute[index[stage]]),where+'/'+name+': component sign')
                close(cov[index['h_only'],index['s_only']],w2[index['macnlops']],w2[index['macnlops']],where+'/'+name+' shared mask covariance')
                if 'H_negative_retained' in index:
                    r=index['H_negative_retained']
                    for stage in ('H_negative','h_only','macnlops'):
                        close(cov[r,index[stage]],w2[r],w2[r],where+'/'+name+' residual subset covariance')
                    for stage in ('H_positive','S_positive','S_negative'):
                        close(cov[r,index[stage]],0.,np.maximum(w2[r],1.e-300),where+'/'+name+' disjoint residual covariance')
                    require(np.all(n[r]<=n[index['H_negative']]),where+'/'+name+': residual count exceeds input negative H')
                # Recompute the exported ratio variance independently from
                # weighted event-mask moments, including baseline covariance.
                for numerator in ('input','h_only','s_only','macnlops'):
                    i=index[numerator];b=raw[prefix+'display_sumw'][0];a=raw[prefix+'display_sumw'][i]
                    valid=b!=0.;ratio=raw[prefix+'ratio_'+numerator+'_to_reference'];variance=raw[prefix+'ratio_variance_'+numerator+'_to_reference']
                    require(np.all(np.isnan(ratio[~valid])) and np.all(np.isnan(variance[~valid])),where+'/'+name+': undefined ratio is not masked')
                    close(ratio[valid],a[valid]/b[valid],np.maximum(abs(a[valid]/b[valid]),1.),where+'/'+name+' ratio '+numerator)
                    expected_var=(w2[i,valid]-2*(a[valid]/b[valid])*cov[i,0,valid]+(a[valid]/b[valid])**2*w2[0,valid])/b[valid]**2
                    if common and numerator=='input':
                        require(np.all(ratio[valid]==1.) and np.all(variance[valid]==0.),where+'/'+name+': identical baseline ratio has uncertainty')
                    close(variance[valid],np.maximum(0.,expected_var),np.maximum((w2[i,valid]+(a[valid]/b[valid])**2*w2[0,valid])/b[valid]**2,1.e-300),where+'/'+name+' ratio variance '+numerator)
                result['histograms']+=1
            veto=data.get('veto_diagnostic');require(veto is not None,where+': missing veto accounting')
            for group in ('total','H_positive','H_negative','S_positive','S_negative'):
                diagnostic=veto[group];source=data['totals']['input' if group=='total' else group]
                require(diagnostic['events']==source['events'],where+'/'+group+': veto source count')
                require(type(diagnostic['realized_rejected_events']) is int and 0<=diagnostic['realized_rejected_events']<=source['events'],where+'/'+group+': invalid rejected count')
                close(diagnostic['input_weight'],source['sumw'],math.sqrt(max(1,source['events'])*source['sumw2']),where+'/'+group+' veto source sumw')
                variance=diagnostic['conditional_bernoulli_variance']
                require(math.isfinite(variance) and variance>=0,where+'/'+group+': invalid conditional variance')
                delta=diagnostic['accepted_weight']-diagnostic['expected_accepted_weight']
                close(diagnostic['accepted_minus_expected'],delta,max(abs(delta),1.),where+'/'+group+' veto residual')
                if variance:
                    close(diagnostic['conditional_veto_z'],delta/math.sqrt(variance),max(abs(delta/math.sqrt(variance)),1.),where+'/'+group+' veto pull')
                else:
                    require(diagnostic['conditional_veto_z'] is None,where+'/'+group+': undefined veto pull')
            require(veto['H_positive']['realized_rejected_events']==0,where+': positive H rejected')
            negative_h=veto['H_negative']['events']-veto['H_negative']['realized_rejected_events']
            if not signed:
                require(negative_h==0,where+': undeclared negative H survived')
            residual=data['totals'].get('H_negative_retained',dict(events=0,negative_events=0,sumw=0.,sumw2=0.))
            require(residual['events']==residual['negative_events']==negative_h,where+': retained negative H count')
            close(veto['H_negative']['accepted_weight'],residual['sumw'],abs(veto['H_negative']['input_weight']),where+' retained H weight')
            negative_s=veto['S_negative']['events']-veto['S_negative']['realized_rejected_events']
            require(negative_s+negative_h==data['totals']['macnlops']['negative_events'],where+': retained negative H/S count')
            kept=sum(veto[s]['events']-veto[s]['realized_rejected_events'] for s in ('H_positive','H_negative','S_positive','S_negative'))
            require(kept==data['totals']['macnlops']['events'],where+': retained count')
            close(sum(veto[s]['accepted_weight'] for s in ('H_positive','H_negative','S_positive','S_negative')),
                  data['totals']['macnlops']['sumw'],math.sqrt(max(1,kept)*data['totals']['macnlops']['sumw2']),where+': veto signed weight')
            result['processes'][process]['levels'][level]=dict(histograms=len(data['observables']),source_events=data['totals']['input']['events'],reference_events=data['totals']['reference']['events'],retained_events=kept,negative_H_survivors=negative_h,negative_S_survivors=negative_s,
                retained_negative_H_sumw=residual['sumw'],
                histograms_sha256=sha256(folder/'histograms.npz'),flows_sha256=sha256(folder/'flows.json'),bin_csv_sha256=bin_csv_sha256)
            if common:
                require(data['totals']['reference']==data['totals']['input'],where+': shared baseline totals differ')
        if {'first','full'}<=set(record['levels']):
            a,b=(record['levels'][x]['totals'] for x in ('first','full'))
            require(set(a)==set(b),process+': first/full total stages differ')
            for stage in a:
                require(a[stage]['events']==b[stage]['events'] and a[stage]['negative_events']==b[stage]['negative_events'],process+'/'+stage+': first/full counts differ')
                for field in ('sumw','sumw2','effective_events'):close(a[stage][field],b[stage][field],max(abs(a[stage][field]),abs(b[stage][field]),1.e-300),process+'/'+stage+' first/full '+field)
        if final:
            originals=original_sources(process,common,integration)
            for role,stage in (('primary','input'),('reference','reference')):
                original=originals[role]
                target=next(iter(record['levels'].values()))['totals'][stage]
                require(original['events']==target['events'] and original['negative_events']==target['negative_events'],process+'/'+role+': original source counts differ')
                close(original['sumw'],target['sumw'],math.sqrt(original['events']*original['sumw2']),process+'/'+role+' original source sumw')
                close(original['sumw2'],target['sumw2'],original['sumw2'],process+'/'+role+' original source sumw2')
                result['processes'][process]['original_sources'][role]=original
    require(set(total_table)==expected_total_keys,'Unexpected stage-total CSV row')
    result['stage_totals_csv_sha256']=sha256(root/'stage_totals.csv')
    result['original_source_weights_checked']=bool(final)
    result['integration_errors_added_to_histogram_variance']=False
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-incomplete',action='store_true')
    args=parser.parse_args()
    report=audit(args.directory,args.allow_incomplete)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','complete_validation','histograms','max_scaled_residual','original_source_weights_checked')}))
if __name__=='__main__':main()
