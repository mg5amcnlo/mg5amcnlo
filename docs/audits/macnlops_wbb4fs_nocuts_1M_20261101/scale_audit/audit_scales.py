#!/usr/bin/env python3
"""All-event written-cap checks and stratified native-import sample for Wbb4FS."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'docs/audits'))
import check_macnlops_continuation_scales as audit


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='run_1M_wbb4fs_nocuts')
    p.add_argument('--per-bin',type=int,default=80)
    p.add_argument('--settings',type=Path)
    p.add_argument('--output',type=Path,default=BASE/'scale_audit')
    p.add_argument('--events',type=int,default=1000000)
    p.add_argument('--seed',type=int,default=4010601)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    source=BASE/'backup'/(a.run+'_events.lhe.gz')
    first=BASE/'results'/(a.run+'_lhe.npz')
    with np.load(first) as data:
        coordinate=data['pt'].copy()
    assert len(coordinate)==a.events
    assert np.all(np.isfinite(coordinate)) and np.all(coordinate>=0)
    audit.scan(source,a.output,'Wbb',a.per_bin,sample_coordinate=coordinate,seed=a.seed)
    path=a.output/'Wbb_written_scales.json'
    result=json.loads(path.read_text())
    assert result['events']==a.events
    result.update(sample_coordinate='pT(W+b+bbar)',sample_coordinate_units='GeV',
                  reservoir_seed=a.seed,per_bin_limit=a.per_bin,
                  sampling='Separate S/H reservoirs in fixed primary-recoil bins; source weights are signed expected transfers. Probe trials are diagnostics of cap transport, not additional production events.')
    record=dict(source=str(source),source_sha256=digest(source),
                first_arrays=str(first),first_arrays_sha256=digest(first),
                scan_script_sha256=digest(ROOT/'docs/audits/check_macnlops_continuation_scales.py'),
                probe_source_sha256=digest(ROOT/'docs/audits/macnlops_continuation_scale_probe.cc'),
                probe_executable_sha256=digest(BASE/'scale_audit/scale_probe'),
                pythia_library_sha256=digest(Path('/nfs/home/rikkert/Downloads/pythia8/lib/libpythia8.so')))
    if a.settings:
        settings=a.settings.resolve();record['settings']=str(settings);record['settings_sha256']=digest(settings)
        text=settings.read_text()
        for needle in ['SPACESHOWER:NQUARKIN = 4','TIMESHOWER:NGLUONTOQUARK = 4','5:M0 = 4.7']:
            assert needle in text.upper(),needle
        env=dict(os.environ,LD_LIBRARY_PATH='/nfs/home/rikkert/LHAPDF/lib:/nfs/home/rikkert/Downloads/pythia8/lib',
                 LHAPDF_DATA_PATH='/nfs/home/rikkert/LHAPDF/share/LHAPDF',
                 PYTHIA8DATA='/nfs/home/rikkert/Downloads/pythia8/share/Pythia8/xmldoc')
        command=[str(BASE/'scale_audit/scale_probe'),str(a.output/'Wbb_sample.lhe'),str(settings),
                 str(a.output/'Wbb_native.csv'),'full','qed-off']
        with (a.output/'native_probe.log').open('w') as log:
            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=a.output)
        assert run.returncode==0,run.returncode
        arr=np.genfromtxt(a.output/'Wbb_native.csv',delimiter=',',names=True)
        arr=np.atleast_1d(arr)
        assert len(arr)==result['sampled_events']
        assert all(np.max(arr[name])<2e-12 for name in ['global_error','production_error','dipole_error'])
        assert np.count_nonzero(arr['missing_fsr_pairs'])==0
        assert np.all(arr['max_emission']<=arr['global']*(1+1e-9))
        assert np.count_nonzero(arr['photons'])==0
        result['native_probe']=dict(events=len(arr),mode='full',command=command,
            max_global_error=float(np.max(arr['global_error'])),
            max_production_error=float(np.max(arr['production_error'])),
            max_dipole_error=float(np.max(arr['dipole_error'])),
            missing_fsr_pairs=int(np.sum(arr['missing_fsr_pairs'])),
            accepted_emissions_above_global=int(np.count_nonzero(arr['max_emission']>arr['global']*(1+1e-9))),
            photons=int(np.sum(arr['photons'])),passed=True,
            limits='Cap import and continuation checks. The unchanged probe final_pt excludes bottom particles and is not the Wbb primary observable; it is not used for Wbb shape analysis.')
        record['native_csv_sha256']=digest(a.output/'Wbb_native.csv')
        record['native_log_sha256']=digest(a.output/'native_probe.log')
    path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (a.output/'provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='bins'},indent=2))

if __name__=='__main__':main()
