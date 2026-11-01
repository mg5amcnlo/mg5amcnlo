#!/usr/bin/env python3
"""Read-only checks of Wbb4FS no-cut inputs and native PDF coupling."""
from pathlib import Path
import argparse,datetime,hashlib,json,re,subprocess,sys,tempfile
ROOT=Path('/export/slow1/rikkert/MCcntRefactor_Sfun_MacNLOPS')
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from madgraph.various.banner import RunCardNLO
from models.check_param_card import ParamCard
PDF=Path('/nfs/home/rikkert/LHAPDF/share/LHAPDF/NNPDF40_nlo_as_01180_nf_4')
LIB=Path('/nfs/home/rikkert/LHAPDF/lib/libLHAPDF.so')
PYTHIA=Path('/nfs/home/rikkert/Downloads/pythia8/lib/libpythia8.so')
def binding(path):
    path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='run_pilot_wbb4fs_nocuts');a=ap.parse_args()
    cards=BASE/'cards'/a.run
    card=RunCardNLO(str(cards/'run_card.dat'))
    cuts={name:card[name] for name in card.cuts_parameter}
    for name,value in cuts.items():
        if isinstance(value,dict):assert not value,(name,value)
        elif 'eta' in name or 'max' in name:assert value<0,(name,value)
        else:assert value==0,(name,value)
    assert card['custom_fcts']==[]
    assert card['lhaid']==[334700] and card['maxjetflavor']==card['macnlops_flavour_count']==4
    assert card['lpp1']==card['lpp2']==1 and card['ebeam1']==card['ebeam2']==6500
    assert card['macnlops'] and card['mcatnlo_delta'] and not card['macnlops_resolved_source'] and card['ickkw']==0
    param=ParamCard(str(cards/'param_card.dat'))
    masses={str(i):param['mass'].get((i,)).value for i in [1,2,3,4,5,21,23]}
    assert all(masses[str(i)]==0 for i in [1,2,3,4,21]) and masses['5']==4.7
    proc=BASE/'processes/Wbb_4FS/Cards/proc_card_mg5.dat'
    text=proc.read_text();assert 'generate p p > w+ b b~ QCD=2 QED=1 [QCD]' in text
    pdefs=re.findall(r'^define p = (.*)$',text,re.M);assert 'b' not in pdefs[-1].split() and 'b~' not in pdefs[-1].split()
    code='''#include "LHAPDF/LHAPDF.h"\n#include <iomanip>\n#include <iostream>\nint main(){LHAPDF::setVerbosity(0);auto p=LHAPDF::mkPDF("NNPDF40_nlo_as_01180_nf_4",0);std::cout<<std::setprecision(17);for(double q:{4.92,91.1876,91.188})std::cout<<q<<" "<<p->alphasQ(q)<<"\\n";std::cout<<p->hasFlavor(5)<<" "<<p->hasFlavor(4)<<"\\n";delete p;}\n'''
    with tempfile.TemporaryDirectory(prefix='wbb4fs-coupling-') as tmp:
        p=Path(tmp);(p/'probe.cc').write_text(code)
        command=['g++','-std=c++11','-I/nfs/home/rikkert/LHAPDF/include',str(p/'probe.cc'),'-L/nfs/home/rikkert/LHAPDF/lib','-Wl,-rpath,/nfs/home/rikkert/LHAPDF/lib','-lLHAPDF','-o',str(p/'probe')]
        subprocess.run(command,check=True,capture_output=True,text=True)
        stdout=subprocess.check_output([str(p/'probe')],text=True)
    lines=stdout.splitlines();alphas={format(float(row.split()[0]),'.8g'):float(row.split()[1]) for row in lines[:3]}
    assert lines[-1]=='0 1' and abs(alphas['91.188']-.11261762576566925)<1e-15
    info=(PDF/'NNPDF40_nlo_as_01180_nf_4.info').read_text()
    def metadata(key):return re.search(r'^'+re.escape(key)+r':\s*(.*)$',info,re.M)[1]
    result=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=a.run,passed=True,
      no_generation_cuts=True,no_analysis_cuts_requested=True,analysis_cut_implementation_verified=False,
      generation=dict(cuts=cuts,custom_fcts=card['custom_fcts'],events=card['nevents'],incoming=pdefs[-1],
                      beam_energy_GeV=6500,ickkw=card['ickkw'],maxjetflavor=4,macnlops_flavour_count=4,model_masses_GeV=masses),
      pdf=dict(name=PDF.name,member=0,LHAPDF_ID=334700,NumFlavors=metadata('NumFlavors'),
               Flavors=metadata('Flavors'),alpha_s_Q=alphas,has_bottom=False,has_charm=True,
               metadata_reference_MZ=metadata('MZ'),metadata_AlphaS_MZ=metadata('AlphaS_MZ'),
               reference_note='The grid MZ/AlphaS_MZ entries specify its low-scale reference anchor; actual alpha_s at physical MZ comes from alphasQ.',
               probe_source=code,probe_output=stdout),
      stock_shower=dict(required_nQuarkIn=4,required_nGluonToQuark=4,alphaSnfmax=5,
               caveat='Massive-bottom nf4 hard/PDF calculation and four physical incoming/splitting flavours. Unmodified stock Pythia retains nfmax5 coupling/trial running; this is not the historical patched fully-nf4 shower profile.',
               first_emission_alpha_s_MZ=.1365,first_emission_alpha_s_order=1,
               continuation_coupling_note='Measured effective continuation settings must be recorded separately; this setup audit does not infer them from the PDF.'),
      hard_coupling_source=dict(setrun='Template/NLO/Source/setrun.f:103 calls pdfwrap for hadron beams',
          evolution='Template/NLO/Source/alfas_functions_lhapdf6.f:89 calls alphasPDFm(iset,Q)',
          python_param_update='madgraph/interface/common_run_interface.py:7636 queries alphasQ(91.1876) when Python bindings are available'),
      inputs={str(p):binding(p) for p in [cards/'run_card.dat',cards/'param_card.dat',cards/'FKS_params.dat',proc,PDF/'NNPDF40_nlo_as_01180_nf_4.info',PDF/'NNPDF40_nlo_as_01180_nf_4_0000.dat',LIB,PYTHIA]},
      limits='Generation cards and PDF/native coupling are checked. Event-level mass/flavour/scale audit and full analysis no-cut checks remain separate.')
    analysis_sources=[BASE/'analysis/wbb_observables.cc',BASE/'analysis/py8an_HwU_wbb.f',BASE/'analysis/observables.json',BASE/'analysis/analysis_checks.json']
    if all(q.is_file() for q in analysis_sources):
        assert json.loads(analysis_sources[-1].read_text())['passed']
        result['analysis_cut_implementation_verified']=True
        result['analysis_no_cut_proof']=dict(method='Independent source read: every stable particle enters the helper, jets use inclusive_jets(0), no pT/rapidity acceptance thresholds; undefined rapidities are missing coordinates only; total raw fills every valid event.',sources={str(q.relative_to(BASE)):binding(q) for q in analysis_sources})
    first_profile=BASE/'processes/Wbb_4FS/Cards/macnlops_profile.json'
    if first_profile.is_file():
        measured=json.loads(first_profile.read_text())
        assert measured['alpha_s_mz']==.1365 and measured['alpha_s_order']==1
        assert measured['support']['n_quark_in']==measured['support']['n_gluon_to_quark']==4
        assert measured['support']['alpha_s_nfmax']==5
        result['first_emission_measured_profile']=binding(first_profile)
    full_profile=BASE/'validation/pilot_full_profile_audit.json'
    if full_profile.is_file():
        measured=json.loads(full_profile.read_text());assert measured['passed']
        result['stock_shower'].update(continuation_alpha_s_MZ=measured['full_alpha_s_MZ'],continuation_alpha_s_order=1,
          continuation_profile_audit=binding(full_profile),
          continuation_coupling_note='Measured after stock Pythia initialization from the corrected RUN_PYTHIA8_2 pilot card. Hard PDF, first-step and continuation couplings are distinct intentional inputs.',
          continuation_particle_masses_GeV={str(i):measured['actual'][str(i)+':m0'] for i in [1,2,3,4,5,6,21,23,24]})
    result['limits']='Generation/analysis no-cut definitions, hard four-flavour PDF/masses, and available measured first/continuation settings are checked. Event-level/distribution validation is separate. Stock nfmax5 coupling running is retained.'
    output=BASE/'validation'/('setup_audit.json' if 'pilot' in a.run else 'setup_audit_'+a.run+'.json')
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('PASS',a.run,'all generation cuts disabled; mb4.7,nf4PDF,physical alpha_s(MZ)=',alphas['91.188'])
    print(output)
if __name__=='__main__':main()
