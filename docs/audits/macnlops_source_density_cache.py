#!/usr/bin/env python3
"""Qualify all source H signs before showering and reuse their exact densities.

The ordinary written particle body may differ from its saved hard point.
Keys therefore bind the preferred hard annotation and original event header,
never the remassed body. This module neither rewrites inputs nor selects a
subset of H events. H-/K>1 is diagnostic under signed-residual matching.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from madgraph.various import macnlops as mc

SCHEMA='macnlops-source-H-density-gate-v1'
RECORD_SCHEMA='macnlops-source-H-density-v1'
CACHE_NAME='source_h_densities.jsonl'
REPORT_NAME='gate_report.json'


def require(condition,message):
    if not condition:raise mc.MatchingError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def artifact(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=mc.sha256(path))


@dataclass(frozen=True)
class HardFingerprint:
    """Canonical physical hard point plus the original MC@NLO header.

    Float.hex preserves parsed double precision, including signed zero.
    No normal-body multiplicity or coordinates enter this key. Mothers,
    colours, spin fields and all native partition labels are retained from
    the annotation, so changing source provenance cannot produce a hit.
    """
    version: int
    kind: str
    process: int
    header: tuple
    source: str
    partition: str
    metadata: tuple
    hard_rows: tuple

    @classmethod
    def from_event(cls,event):
        require(event.hard is not None,'H density cache requires the saved hard annotation')
        require(event.hard.partition=='global_sum','H density cache cannot use sampled source partitions')
        require(event.endpoint_origin!='shower',
                'H density cache cannot replace an evolved shower point by its source annotation')
        packet=re.search(r'<macnlops_hard_event\b[^>]*>(.*?)</macnlops_hard_event\s*>',event.block,re.S)
        require(packet is not None,'Missing canonical hard packet')
        rows=[]
        for row in packet[1].strip().splitlines()[1:]:
            fields=row.split()
            # Event/HardEvent already validates this packet. Parse every
            # column to retain physical and latent provenance exactly.
            rows.append(tuple(map(int,fields[:6]))+tuple(mc.number(x).hex() for x in fields[6:]))
        header=tuple(float(x).hex() for x in (event.weight,event.source_scale,event.aqed,event.aqcd))
        return cls(1,event.kind,event.process,header,event.hard.source,
                   event.hard.partition,event.hard.metadata,tuple(rows))

    def sha256(self):
        return digest(asdict(self))


def physical_fingerprint(event):
    return HardFingerprint.from_event(event).sha256()


def point_checks(weight,density):
    """Fatal density/sign checks; never impose or clip an H-/K bound."""
    values={key:float(getattr(density,key)) for key in ('real','subtraction','shower')}
    errors=[]
    if not all(math.isfinite(value) for value in values.values()):
        return ['nonfinite_density'],dict(densities={k:v if math.isfinite(v) else str(v) for k,v in values.items()})
    if values['real']<0 or values['shower']<0:
        errors.append('negative_real_or_shower_density')
    hard=values['real']-values['subtraction']
    if not math.isfinite(hard):errors.append('nonfinite_global_H')
    elif weight!=0:
        if hard==0:errors.append('zero_global_H_with_nonzero_source_weight')
        elif (weight>0)!=(hard>0):errors.append('source_H_sign_disagrees_with_global_R_minus_D')
    negative=max(-hard,0.) if math.isfinite(hard) else 0.
    shower=values['shower']
    ratio=negative/shower if shower>0 else (None if negative else 0.)
    if ratio is not None and not math.isfinite(ratio):ratio=None
    return errors,dict(densities=values,global_H=hard if math.isfinite(hard) else str(hard),
        negative_H_over_K=ratio,requires_signed_residual=negative>max(shower,0.),
        negative_H_with_zero_K=negative>0 and shower==0,
        required_H_survival=max(1.-shower/negative,0.) if negative and shower>=0 else 0.)


def _inventory(process):
    paths=[]
    for directory in sorted((process/'SubProcesses').glob('P*')):
        if (directory/'leshouche_info.dat').is_file():
            binary=directory/'macnlops_evaluator'
            require(binary.is_file(),'Precompile all native evaluators before the source gate: '+str(binary))
            paths.append(binary)
    require(paths,'No compiled native evaluators for the source gate')
    driver=process/'SubProcesses/driver_macnlops.f'
    if driver.is_file():paths.append(driver)
    paths.extend(sorted((process/'lib').glob('*.so')))
    return {str(path.relative_to(process)):mc.sha256(path) for path in paths}


def _bindings(source,process,configuration):
    profile_path=process/'Cards/macnlops_profile.json'
    profile=mc.generation_profile(process)
    require(profile['construction']==mc.ORDINARY_SOURCE,'Source H cache requires ordinary MC@NLO construction')
    for key,profile_key in (('support','support'),('pdf','pdf'),('alpha','alpha_s_mz'),
                            ('order','alpha_s_order'),('pdf_info_sha256','pdf_info_sha256'),
                            ('pdf_member_sha256','pdf_member_sha256')):
        require(configuration.get(key)==profile[profile_key],'Source gate configuration differs from profile: '+key)
    require(mc.sha256(configuration['settings'])==profile['settings_sha256'],
            'Source gate settings differ from the generation profile')
    pin=configuration.get('source_manifest')
    require(isinstance(pin,dict) and {'path','sha256'}<=set(pin),'Missing full-source manifest artifact for chunk gate')
    manifest_path=Path(pin['path']).resolve()
    require(mc.sha256(manifest_path)==pin['sha256'],'Source generation manifest changed')
    manifest=json.loads(manifest_path.read_text())
    require(manifest.get('schema')=='macnlops-source-v2' and isinstance(manifest.get('sources'),dict),
            'Unknown source generation manifest')
    require(bool(manifest['sources']),'Source generation manifest has no source bindings')
    if manifest_path in (Path(str(source)+'.macnlops.json'),Path(str(source)+'.gz.macnlops.json')):
        require(mc.lhe_sha256(source)==manifest.get('lhe_sha256'),
                'Generation manifest does not describe the original source bytes')
    for name,sha in manifest['sources'].items():
        # This is the same explicitly allowed evaluator-only exception as
        # check_manifest. Its actual current bytes/binary are pinned below.
        if Path(name).name=='driver_macnlops.f':continue
        require(mc.sha256(process/name)==sha,'Generation source changed before H gate: '+name)
    return dict(source=str(source),source_chunk_sha256=mc.sha256(source),process=str(process),
        configuration=configuration,configuration_sha256=digest(configuration),
        profile=artifact(profile_path),source_manifest=artifact(manifest_path),
        evaluators=_inventory(process),cache_module=artifact(Path(__file__)))


def _return_report(path,report):
    return dict(report,report_path=str(path.resolve()),report_sha256=mc.sha256(path))


def _load_complete(cache_path,expected_report_sha256=None,bindings=None):
    cache_path=Path(cache_path).resolve();report_path=cache_path.parent/REPORT_NAME
    require(report_path.is_file(),'Missing completed source-H gate report')
    if expected_report_sha256 is not None:
        require(mc.sha256(report_path)==expected_report_sha256,'Source-H gate report changed after qualification')
    report=json.loads(report_path.read_text())
    require(report.get('schema')==SCHEMA and report.get('passed') is True and
            report.get('complete_source') is True,'Source-H density gate has not passed completely')
    require(report.get('cache_path')==str(cache_path) and cache_path.is_file(),
            'Source-H gate cache path mismatch or missing cache')
    require(mc.sha256(cache_path)==report.get('cache_sha256'),'Source-H density cache bytes changed')
    if bindings is not None:
        require(report.get('bindings')==bindings,'Source-H gate source/profile/evaluator bindings changed')
    records=[];seen=set();points={}
    with cache_path.open() as stream:
        for line in stream:
            record=json.loads(line);sequence=record.get('source_sequence')
            require(record.get('schema')==RECORD_SCHEMA and type(sequence) is int and sequence>0 and sequence not in seen,
                    'Invalid or duplicate source-H cache record')
            seen.add(sequence)
            key=record.get('physical_fingerprint')
            require(isinstance(key,str) and re.fullmatch('[0-9a-f]{64}',key) is not None,'Invalid source-H physical fingerprint')
            density=mc.Densities(**record['densities'])
            errors,_=point_checks(record['source_weight'],density)
            require(not errors,'Invalid cached source-H density: '+','.join(errors))
            if key in points:
                require(points[key]==density,'Identical source-H points have inconsistent cached densities')
            points[key]=density;records.append(record)
    require(len(records)==report['counts']['source_H']==report['counts']['evaluated_H'],
            'Source-H cache does not cover every source H record')
    return report,points


def verify_gate(source,process,configuration,output_directory):
    """Validate a durable gate without starting a native evaluator."""
    source,process=Path(source).resolve(),Path(process).resolve()
    directory=Path(output_directory).resolve()
    report,_=_load_complete(directory/CACHE_NAME,bindings=_bindings(source,process,configuration))
    return _return_report(directory/REPORT_NAME,report)


def gate_chunk(source,process,configuration,output_directory):
    """Evaluate every H sign before any native shower; reuse a verified gate.

    Caller precompiles the evaluators once. configuration is the prepared
    native configuration plus source_manifest={path,sha256}. An incomplete
    or failed gate never qualifies a cache and no existing checkpoint is
    overwritten. Errors preserve the exact first failing event separately.
    """
    source,process=Path(source).resolve(),Path(process).resolve()
    directory=Path(output_directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.gate.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        binding=_bindings(source,process,configuration)
        cache_path,report_path=directory/CACHE_NAME,directory/REPORT_NAME
        if report_path.exists():
            report,_=_load_complete(cache_path,bindings=binding)
            return _return_report(report_path,report)
        require(not cache_path.exists(),'Uncheckpointed source-H cache exists; preserve it before restarting the gate')
        descriptor,temporary_name=tempfile.mkstemp(prefix='.source_h_densities.',dir=directory,text=True)
        temporary=Path(temporary_name)
        logs=Path(tempfile.mkdtemp(prefix='evaluator_logs.',dir=directory))
        counts=Counter(source_events=0,source_S=0,source_H=0,evaluated_H=0,
                       positive_H=0,negative_H=0,zero_weight_H=0,failed_H=0)
        statistics=dict(requires_signed_residual_H=0,negative_H_with_zero_K=0,
            max_finite_negative_H_over_K=0.,max_required_H_survival=0.,unbounded_negative_H_over_K=0)
        first_failure=None;fatal=None;complete=False
        environment=dict(os.environ,**configuration.get('environment',{}))
        try:
            with os.fdopen(descriptor,'w') as cache,mc.NativeEvaluator(process,logs,environment=environment) as evaluator:
                support=configuration['support']
                evaluator.configure(support['beam_xmax'],support['shower_cutoffs'])
                with mc.LHEStream(source) as stream:
                    for sequence,block in enumerate(stream,1):
                        event=mc.Event(block);counts['source_events']+=1;counts['source_'+event.kind]+=1
                        require(event.hard is not None and event.hard.source=='ordinary' and
                                event.hard.partition=='global_sum' and event.endpoint_origin is None,
                                'Source gate needs original ordinary globally summed hard annotations on every event')
                        if event.kind!='H':continue
                        sign='positive_H' if event.weight>0 else 'negative_H' if event.weight<0 else 'zero_weight_H'
                        counts[sign]+=1
                        try:
                            density=evaluator.evaluate(event,sequence)
                        except Exception as error:
                            counts['failed_H']+=1
                            first_failure=dict(sequence=sequence,errors=['native_evaluator_error'],error=str(error),source_weight=event.weight)
                            (directory/'first_failure.lhe').write_text(stream.header+block+'</LesHouchesEvents>\n')
                            raise
                        counts['evaluated_H']+=1
                        errors,details=point_checks(event.weight,density)
                        record=dict(schema=RECORD_SCHEMA,source_sequence=sequence,
                            physical_fingerprint=physical_fingerprint(event),source_weight=event.weight,**details)
                        cache.write(canonical(record)+'\n')
                        statistics['requires_signed_residual_H']+=details.get('requires_signed_residual',False)
                        statistics['negative_H_with_zero_K']+=details.get('negative_H_with_zero_K',False)
                        ratio=details.get('negative_H_over_K')
                        if ratio is not None:statistics['max_finite_negative_H_over_K']=max(statistics['max_finite_negative_H_over_K'],ratio)
                        elif details.get('requires_signed_residual'):statistics['unbounded_negative_H_over_K']+=1
                        statistics['max_required_H_survival']=max(statistics['max_required_H_survival'],details.get('required_H_survival',0.))
                        if errors:
                            counts['failed_H']+=1
                            if first_failure is None:
                                first_failure=dict(sequence=sequence,errors=errors,source_weight=event.weight,**details)
                                (directory/'first_failure.lhe').write_text(stream.header+block+'</LesHouchesEvents>\n')
                cache.flush();os.fsync(cache.fileno());complete=True
        except Exception as error:
            fatal=str(error)
        try:
            require(_bindings(source,process,configuration)==binding,'Source/profile/evaluators changed during H gate')
        except Exception as error:
            fatal=str(error);complete=False
        passed=complete and not fatal and counts['failed_H']==0 and counts['source_events']>0
        report=dict(schema=SCHEMA,passed=passed,complete_source=complete,
            scope='all original source H signs; H-/K bound informational, signed residuals allowed; no event rejection',
            source=str(source),source_chunk_sha256=binding['source_chunk_sha256'],
            profile_sha256=binding['profile']['sha256'],bindings=binding,counts=dict(counts),
            informational=statistics,first_failure=first_failure,error=fatal,
            cache_path=str(cache_path),evaluator_log_directory=str(logs),
            source_unchanged=mc.sha256(source)==binding['source_chunk_sha256'])
        report['informational']['max_negative_H_over_K']=(None if statistics['unbounded_negative_H_over_K'] else statistics['max_finite_negative_H_over_K'])
        # Publish the cache first and the report last. A crash between these
        # operations leaves an explicit uncheckpointed cache, never a pass.
        temporary.replace(cache_path);report['cache_sha256']=mc.sha256(cache_path)
        report_temporary=directory/'.gate_report.tmp'
        with report_temporary.open('w') as stream:
            stream.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        report_temporary.replace(report_path)
        require(passed,'Source-H density gate failed: '+str(report_path))
        _load_complete(cache_path,bindings=binding)
        return _return_report(report_path,report)


class CachedSourceEvaluator:
    """Borrow a live evaluator for S; serve only qualified, matching H keys."""
    def __init__(self,native,cache_path,expected_report_sha256=None):
        self.native=native
        cache_path=Path(cache_path).resolve()
        report,points=_load_complete(cache_path,expected_report_sha256)
        binding=report['bindings'];process=Path(binding['process'])
        require(Path(native.process_dir).resolve()==process,'H cache belongs to another generated process')
        current=_bindings(Path(binding['source']),process,binding['configuration'])
        require(current==binding,'H cache source/profile/evaluator bindings changed')
        environment=os.environ if native.environment is None else native.environment
        for key,value in binding['configuration'].get('environment',{}).items():
            require(environment.get(key)==value,'Live S evaluator environment differs from H gate: '+key)
        self.report=report;self.points=points;self.cached_H_calls=0;self.live_S_calls=0
        support=binding['configuration']['support']
        self.support=tuple(support['beam_xmax'])+tuple(support['shower_cutoffs'])

    def configure(self,beam_xmax,shower_cutoffs):
        require(tuple(beam_xmax)+tuple(shower_cutoffs)==self.support,'H cache shower support differs from native endpoints')
        return self.native.configure(beam_xmax,shower_cutoffs)

    def evaluate(self,event,request_id):
        require(self.native.configuration==self.support,'Configure cached evaluator before evaluating events')
        if event.kind=='S':
            self.live_S_calls+=1
            return self.native.evaluate(event,request_id)
        require(event.kind=='H','Unknown source kind for density cache')
        key=physical_fingerprint(event)
        require(key in self.points,'H event hard annotation/header does not match its qualified source-density cache')
        self.cached_H_calls+=1
        return self.points[key]

    def __getattr__(self,name):
        return getattr(self.native,name)
