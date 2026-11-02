"""Streaming signed-weight diagnostics; only histogram aggregates are saved."""
from collections import Counter
import ctypes
import gzip
import json
import math
import re
import subprocess

import numpy as np
from specification import ADDED_IDS, FIELDS, HISTOGRAMS, WEIGHTS

WGT = re.compile(r"<wgt\s+id=['\"]macnlops['\"]\s*>([^<]+)</wgt>")
ATTR = re.compile(r'''\b([A-Za-z_][A-Za-z0-9_]*)=["']([^"']*)["']''')


def number(x): return float(x.replace('D','E').replace('d','e'))


def compiled_observables(a,p):
    from run import HERE,sha
    source=HERE/'observables.cc'
    common=a.base/'processes'/p['directory']/'MCatNLO/srcCommon'
    inputs=[source,common/'fjcore.cc',common/'fjcore.hh']
    token=__import__('hashlib').sha256(''.join(sha(x) for x in inputs).encode()).hexdigest()[:20]
    library=a.base/'build'/('observables_'+token+'.so')
    if not library.exists():
        subprocess.run(['g++','-O2','-std=c++11','-shared','-fPIC','-I'+str(common),
                        str(source),str(common/'fjcore.cc'),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library));fun=lib.native_tc_observables_
    fun.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int)]
    def observe(final):
        ids=np.asarray([x[0] for x in final],dtype=np.int32)
        momenta=np.asarray([x[1:] for x in final],dtype=np.float64)
        values=np.zeros(19);n=ctypes.c_int(len(ids));code=ctypes.c_int(p['code']);status=ctypes.c_int()
        fun(ctypes.byref(n),ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            momenta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.byref(code),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.byref(status))
        if status.value:raise RuntimeError(f'Observable status {status.value}')
        return values
    return observe


class Histograms:
    def __init__(self,blocks):
        self.blocks=blocks
        self.sum=[np.zeros((n+2,6)) for _,n,*_ in HISTOGRAMS]
        self.cross=[np.zeros((n+2,4,4)) for _,n,*_ in HISTOGRAMS]
        self.block=[np.zeros((blocks,n+2,4)) for _,n,*_ in HISTOGRAMS]

    def fill(self,v,w,block):
        for i,(_,n,lo,hi,field,log,fid) in enumerate(HISTOGRAMS):
            if fid and v[8]<.5:continue
            x=.5 if field is None else v[field]
            if not math.isfinite(x):continue
            if log:
                if x<=0:continue
                x=math.log10(x)
            b=0 if x<lo else n+1 if x>=hi else min(n,1+int((x-lo)*n/(hi-lo)))
            self.sum[i][b]+=w
            self.cross[i][b]+=np.outer(w[:4],w[:4])
            self.block[i][block,b]+=w[:4]

    def serialise(self):
        return dict(weights=WEIGHTS,block_weights=WEIGHTS[:4],histograms={
            spec[0]:dict(edges=np.linspace(spec[2],spec[3],spec[1]+1).tolist(),
                sumw=self.sum[i].tolist(),sumww=self.cross[i].tolist(),blocks=self.block[i].tolist(),
                flow_bins='First/last rows are underflow/overflow; no range renormalization')
            for i,spec in enumerate(HISTOGRAMS)})


def diagnose(a,p):
    from run import dump,sha,epoch
    result=a.base/'results'/p['key']/a.run;result.mkdir(parents=True,exist_ok=True)
    source=a.base/'processes'/p['directory']/'Events'/a.run/'events.lhe.gz'
    summary=result/'first.json'
    if summary.exists():
        from resume_identity import verify_resume
        verified=verify_resume(source,summary,result/'encoding_audit.json')
        dump(result/'resume_identity.json',verified)
        print('First analysis already complete:',summary,flush=True);return
    observe=compiled_observables(a,p)
    blocks=50 if a.events>=10000 and a.events%50==0 else 4 if a.events%4==0 else 1
    hist=Histograms(blocks);counts=Counter();rules=Counter();totals=np.zeros(6)
    expected=variance=0.;missing=np.zeros(19,dtype=int)
    original_hash=sha(source);temporary=source.with_name(source.name+'.diagnostics.tmp')
    group="    <weightgroup name='native_tc_diagnostics' combine='none'>\n"+''.join(
        f"      <weight id='{name}'> {name} </weight>\n" for name in ADDED_IDS)+"    </weightgroup>\n"
    try:
        with gzip.open(source,'rt') as stream,gzip.open(temporary,'wt',compresslevel=1) as out:
            header=[]
            for line in stream:
                if line.lstrip().startswith('<event'):break
                header.append(line)
            else:raise RuntimeError('No LHE events')
            text=''.join(header)
            if any(name in text for name in ADDED_IDS):raise RuntimeError('Diagnostic weights already exist without summary')
            if text.count('</initrwgt>')!=1:raise RuntimeError('Expected one initrwgt')
            out.write(text.replace('</initrwgt>',group+'</initrwgt>'))
            while line:
                if not line.lstrip().startswith('<event'):
                    out.write(line);line=next(stream,'');continue
                event=[line,next(stream)];tokens=event[1].split();n=int(tokens[0]);nominal_text=tokens[2]
                w=number(nominal_text)
                if not math.isfinite(w) or w==0:raise RuntimeError('Invalid nominal weight')
                final=[]
                for _ in range(n):
                    row=next(stream);event.append(row);t=row.split()
                    if int(t[1])==1:final.append([int(t[0])]+[number(t[k]) for k in (6,7,8,9)])
                kind=packet=None;matched_text=None;valid_source=False
                for line in stream:
                    event.append(line)
                    if line.startswith('#aMCatNLO'):kind=int(line.split()[1])
                    if '<macnlops_hard_event ' in line:
                        attrs=dict(ATTR.findall(line));valid_source=attrs.get('source')=='delta_native_tc'
                    if '<macnlops_decision ' in line:packet=dict(ATTR.findall(line))
                    match=WGT.search(line)
                    if match:
                        if matched_text is not None:raise RuntimeError('Duplicate matching weight')
                        matched_text=match[1].strip()
                    if '</event>' in line:break
                if kind not in (1,2) or packet is None or matched_text is None or not valid_source:
                    raise RuntimeError('Missing source/decision/kind or incompatible source epoch')
                m=number(matched_text);keep=number(packet['keep'])
                if not (math.isfinite(m) and math.isfinite(keep) and 0<=keep<=1):raise RuntimeError('Invalid matching values')
                if m!=0 and not math.isclose(m,w,rel_tol=1.e-7):raise RuntimeError('Invalid matching draw')
                veto,removal=(matched_text,nominal_text) if kind==1 else (nominal_text,matched_text)
                extra=(veto,removal,nominal_text if kind==1 else '0',nominal_text if kind==2 else '0')
                weights=np.asarray([w,m,number(veto),number(removal),number(extra[2]),number(extra[3])])
                if not math.isclose(weights[2]+weights[3],w+m,rel_tol=1.e-14,abs_tol=1.e-14):raise RuntimeError('Diagnostic identity failed')
                v=observe(final);missing+=~np.isfinite(v)
                index=counts['events']
                if index>=a.events:raise RuntimeError('More events than --events')
                hist.fill(v,weights,min(blocks-1,index*blocks//a.events));totals+=weights
                expected+=w*keep;variance+=w*w*keep*(1-keep)
                label='S' if kind==1 else 'H';counts['events']+=1;counts[label]+=1
                counts['negative_'+label]+=w<0
                counts['retained_'+label]+=m!=0;counts['retained_negative_'+label]+=(m<0)
                counts['zero_matched']+=m==0;rules[label+':'+packet['rule']]+=1
                for row in event:
                    out.write(row)
                    if WGT.search(row):
                        out.writelines(f"    <wgt id='{name}'> {value} </wgt>\n" for name,value in zip(ADDED_IDS,extra))
                line=next(stream,'')
        if counts['events']!=a.events:raise RuntimeError('Event count differs from --events')
        # Keep a single canonical event copy. The unmodified hash records input provenance.
        from resume_identity import payload_sha
        modified_hash=sha(temporary);modified_payload_hash=payload_sha(temporary)
        temporary.replace(source)
        with gzip.open(result/'first_histograms.json.gz','wt') as out:json.dump(hist.serialise(),out,separators=(',',':'),allow_nan=False)
        dump(summary,dict(process=p['key'],run=a.run,events=a.events,counts=dict(counts),rules=dict(rules),
            totals_pb=dict(zip(WEIGHTS,totals.tolist())),expected_matched_pb=expected,
            matching_draw_variance_pb2=variance,blocks=blocks,
            missing_observables=dict(zip(FIELDS,missing.tolist())),
            original_input_sha256=original_hash,diagnostic_input_sha256=modified_hash,
            diagnostic_payload_sha256=modified_payload_hash,
            histogram_sha256=sha(result/'first_histograms.json.gz'),
            normalization='Original signed source weights; no survivor or acceptance renormalization',
            first_histograms='first_histograms.json.gz',provenance=epoch()))
    except BaseException:
        temporary.unlink(missing_ok=True);raise
    print('First analysis ready:',summary,flush=True)


def read_hwu(path):
    result={};names=None
    lines=iter(path.read_text().splitlines())
    for line in lines:
        if line.startswith('##&'):names=[x.strip() for x in line[3:].split('&')]
        if not line.startswith('<histogram>'):continue
        label=line.split('"')[1].strip().split(' |')[0];rows=[]
        for line in lines:
            if line.startswith('<\\histogram>'):break
            if line.strip():rows.append([number(x) for x in line.split()])
        if label in result:raise RuntimeError('Duplicate histogram '+label)
        result[label]=np.asarray(rows)
    if names is None or not result:raise RuntimeError('Malformed HwU '+str(path))
    return names,result


def analyse_full(a,p):
    from run import dump,sha
    process=a.base/'processes'/p['directory'];result=a.base/'results'/p['key']/a.run
    candidates=[]
    for folder in (process/'MCatNLO').glob('RUN_PYTHIA8_*'):
        files=sorted(folder.glob('run_*/MADatNLO.HwU'),key=lambda x:int(x.parent.name.split('_')[-1]))
        if not files and (folder/'MADatNLO.HwU').is_file():files=[folder/'MADatNLO.HwU']
        if files and all((x.parent/'native_tc_counts.dat').is_file() for x in files):candidates.append((folder,files))
    if not candidates:raise RuntimeError('No completed native-tc shower analysis found')
    folder,files=max(candidates,key=lambda pair:pair[0].stat().st_mtime)
    blocks=[];bindings=[];counts=np.zeros(9,dtype=np.int64);names=None
    for path in files:
        labels,rows=read_hwu(path)
        if names is not None and names!=labels:raise RuntimeError('Different HwU weight labels')
        names=labels
        local=np.asarray([int(x) for x in (path.parent/'native_tc_counts.dat').read_text().split()])
        if local.shape!=(9,):raise RuntimeError('Invalid native count file')
        counts+=local;blocks.append(rows);bindings.append(dict(path=str(path),sha256=sha(path),events=int(local[0])))
    if counts[0]!=a.events:raise RuntimeError(f'Full shower processed {counts[0]}, expected {a.events}')
    # Native tables include an error column after the nominal value.
    weight_indices=[names.index(x) for x in ('central value','macnlops','macnlops_veto_only',
        'macnlops_removal_only','native_tc_source_S','native_tc_source_H')]
    data={}
    for label in blocks[0]:
        parts=np.asarray([b[label][:,weight_indices] for b in blocks])
        data[label]=dict(edges=blocks[0][label][:,:2].tolist(),sumw=parts.sum(axis=0).tolist(),
                         blocks=parts[:,:,:4].tolist())
    with gzip.open(result/'full_histograms.json.gz','wt') as output:
        json.dump(dict(weights=WEIGHTS,histograms=data),output,separators=(',',':'),allow_nan=False)
    total=np.asarray(data['total']['sumw'])[0]
    report=dict(process=p['key'],run=a.run,events=a.events,shower_directory=str(folder),jobs=len(files),
        counts=dict(zip(('events','S','H','negative_S','negative_H','zero_matched','retained_negative_S',
                        'retained_negative_H','fiducial_pass'),counts.tolist())),
        totals_pb=dict(zip(WEIGHTS,total.tolist())),histogram_sha256=sha(result/'full_histograms.json.gz'),
        histogram='full_histograms.json.gz',bindings=bindings,
        normalization='Native HwU signed weights; no survivor or acceptance renormalization',
        checkpoint_policy='HwU and aggregate counts replaced at checkpoints; no per-event output streams')
    dump(result/'full.json',report)
    make_report(result)
    print('Full analysis ready:',result/'full.json',flush=True)


def make_report(result):
    """Paired per-job uncertainty, retaining all signed contributions."""
    from run import dump
    stages={}
    for stage in ('first','full'):
        path=result/(stage+'_histograms.json.gz')
        if not path.exists():continue
        with gzip.open(path,'rt') as stream:data=json.load(stream)
        selected={}
        for name in ('total','primary_0_100','primary_0_500','fiducial40_total','fiducial40_primary_0_100'):
            h=data['histograms'][name];s=np.asarray(h['sumw']);b=np.asarray(h['blocks'])
            if stage=='first':s=s[1:-1];b=b[:,1:-1]
            transfer_s=s[:,0]-s[:,2];transfer_h=s[:,3]-s[:,0]
            net=s[:,1]-s[:,0];n=len(b)
            delta=b[:,:,1]-b[:,:,0]
            error=np.sqrt(n/(n-1)*((delta-delta.mean(axis=0))**2).sum(axis=0)) if n>1 else np.zeros_like(net)
            selected[name]=dict(source=s[:,0].tolist(),matched=s[:,1].tolist(),
                S_veto=transfer_s.tolist(),H_removal=transfer_h.tolist(),net=net.tolist(),
                net_paired_block_error=error.tolist())
        stages[stage]=selected
    dump(result/'comparison.json',dict(stages=stages,
        errors='Paired contiguous source chunks/native shower jobs; source and transfer share the same events',
        S_veto='source - veto_only',H_removal='removal_only - source',net='matched - source'))
