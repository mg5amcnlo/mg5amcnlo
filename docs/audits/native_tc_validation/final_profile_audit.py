#!/usr/bin/env python3
"""Read-only audit of recorded native profiles, physical masses and switches."""
import argparse
import json
from pathlib import Path

from run import DEFAULT_BASE,PROCESSES,PYTHIA,dump,sha

OFF=('partonlevel:mpi','hadronlevel:hadronize','beamremnants:primordialkt',
    'spaceshower:qedshowerbyq','spaceshower:qedshowerbyl',
    'timeshower:qedshowerbyq','timeshower:qedshowerbyl','timeshower:qedshowerbygamma',
    'timeshower:qedshowerbyother','spaceshower:mecorrections','timeshower:mecorrections',
    'timeshower:globalrecoil','6:maydecay','24:maydecay','25:maydecay')
ON=('beams:setproductionscalesfromlhef','beams:setdipoleshowerstartingscalesfromlhef',
    'timeshower:limitptmaxglobal','main:spareflag1')


def audit(base,p,run,stock_sha):
    result=base/'results'/p['key']/run;path=result/'native_accounting.json'
    if not path.exists():return dict(status='pending',reason='native_accounting.json missing')
    data=json.loads(path.read_text());first=data['first_profile']['values'];support=first['support']
    ee=p['key'].startswith('EE');nf=p['flavours']
    pdf='none' if ee else 'LHAPDF6:NNPDF40_nlo_as_01180'+('_nf_4' if nf==4 else '')+'/0'
    checks=dict(native_accounting=data.get('passed') is True and data.get('native_event_accounting_passed') is True,
        first_source=first['construction']=='mcatnlo-delta-native-tc-v1',
        first_alpha=first['alpha_s_mz']==.1365 and first['alpha_s_order']==1,
        first_pdf=first['pdf']==pdf,
        first_beams=support['beam_ids']==([-11,11] if ee else [2212,2212]),
        first_energies=all(x==(45.594 if ee else 6500) for x in support['beam_energies_cm']),
        first_species=support['flavour_count']==nf and support['n_quark_in']==nf and support['n_gluon_to_quark']==nf,
        first_nfmax=support['alpha_s_nfmax']==5,
        first_QCD_only=support['qcd_only'] is True and not any(support['qcd_only_off_switches'].values()),
        first_scales=support['production_scales_from_lhef'] is True and support['dipole_scales_from_lhef'] is True,
        first_cutoffs=all(abs(x-.5)<1.e-14 for x in support['shower_cutoffs']),
        first_stock_library=first['shower_libraries']['libpythia8.so']==stock_sha)
    from madgraph.various import macnlops
    masses=macnlops.hard_model_masses(base/'processes'/p['directory'])
    checks['first_physical_masses']=all(support['particle_masses'].get(str(pid))==mass for pid,mass in masses.items())
    jobs=[]
    for row in data['full']['jobs']:
        s=row['effective_settings'];good={}
        for key in OFF:good[key]=s.get(key) in ('off','false','0')
        for key in ON:good[key]=s.get(key) in ('on','true','1')
        for prefix in ('spaceshower','timeshower'):
            for suffix,value in (('alphasvalue',.118),('alphasorder',1.),('ptmaxmatch',1.),('ptmaxfudge',1.),('ptmin',.5)):
                good[prefix+':'+suffix]=abs(float(s.get(prefix+':'+suffix,'nan'))-value)<1.e-14
        expected_pdf='13' if ee else 'lhapdf6:nnpdf40_nlo_as_01180'+('_nf_4' if nf==4 else '')
        good['pdf:pset']=s.get('pdf:pset')==expected_pdf;good['pdf:lepton']=s.get('pdf:lepton') in ('off','false','0')
        good['physical_species']=s.get('spaceshower:nquarkin')==str(nf) and s.get('timeshower:ngluontoquark')==str(nf)
        good['stock_nfmax']=s.get('standardmodel:alphasnfmax')=='5'
        good['physical_masses']=all(float(s.get(str(pid)+':m0','nan'))==mass for pid,mass in masses.items())
        jobs.append(dict(job=row['job'],card_sha256=row['card_sha256'],passed=all(good.values()),checks=good))
    checks['full_profiles']=bool(jobs) and all(x['passed'] for x in jobs)
    report=dict(status='complete' if all(checks.values()) else 'failed',passed=all(checks.values()),
        process=p['key'],run=run,checks=checks,first_pdf=first['pdf'],first_alpha_s_mz=.1365,
        full_alpha_s_mz=.118,full_pdf='Pythia PDF:pSet='+expected_pdf,physical_flavours=nf,stock_alpha_s_nfmax=5,
        hard_model_masses=masses,jobs=jobs,stock_pythia_library_sha256=stock_sha,
        native_accounting_sha256=sha(path),notes='Warnings/errors are audited separately from settings. Recorded support/mass values are actual first-emission initialization; full settings are the last assignment of each case-insensitive card key.')
    dump(result/'final_profile_audit.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE);parser.add_argument('--run',default='run_1M_native_tc')
    a=parser.parse_args();stock=sha(PYTHIA/'lib/libpythia8.so')
    reports={p['key']:audit(a.base,p,a.run,stock) for p in PROCESSES}
    complete=[key for key,r in reports.items() if r['status']=='complete'];failed=[key for key,r in reports.items() if r['status']=='failed']
    dump(a.base/'validation/final_physical_profiles.json',dict(run=a.run,processes=reports,
        complete_processes=complete,failed_processes=failed,all_eight_complete=len(complete)==8,
        available_profiles_passed=not failed,pythia_modified=False,
        historical_qualification='Fresh full showers use explicit hard-model masses; archived DY full retained Pythia charm/bottom defaults. Both fresh and archived hadronic continuations use the corresponding NLO LHAPDF set; bare-lepton cards contain inactive PDF13 with PDF:lepton off.'))
    print('Profiles',len(complete),'/8 complete; failures',failed,flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
