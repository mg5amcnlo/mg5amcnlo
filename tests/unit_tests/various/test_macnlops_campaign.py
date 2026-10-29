"""Lightweight campaign/card/analysis checks: no event generation or showering."""
import copy
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('macnlops_campaign_tested',
    ROOT / 'docs/audits/run_macnlops_campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class TestCampaign(unittest.TestCase):
    def test_cpu_slots(self):
        self.assertEqual(campaign.cpu_slots(range(12)), [list(range(6)), list(range(6, 12))])
        self.assertEqual(campaign.cpu_slots([3, 4, 7]), [[3], [4]])
        self.assertEqual(campaign.cpu_slots([9]), [[9]])
        self.assertEqual(campaign.cpu_slots(range(40)), [list(range(6)), list(range(6, 12))])
        for parallel, cores in [(3, 6), (2, 7), (0, 6), (2, 0)]:
            with self.assertRaises(ValueError):
                campaign.cpu_slots(range(12), parallel, cores)

    def test_queue_reuses_only_free_slots(self):
        live, seen, completed = {}, [], []

        class Child:
            def __init__(self, key):
                self.key, self.polls = key, 0

            def poll(self):
                self.polls += 1
                return None if self.polls < 3 else int(self.key == 'Wbb')

        def start(p, cpus):
            self.assertLess(len(live), 2)
            self.assertTrue(all(set(cpus).isdisjoint(other) for other in live.values()))
            live[p['key']] = cpus
            seen.append(p['key'])
            return Child(p['key'])

        def finished(p, cpus, child):
            self.assertEqual(live.pop(p['key']), cpus)
            completed.append(p['key'])

        with mock.patch.object(campaign.time, 'sleep'):
            failed = campaign.run_queue(campaign.PROCESSES,
                campaign.cpu_slots(range(12)), start, finished)
        self.assertEqual(seen, [p['key'] for p in campaign.PROCESSES])
        self.assertEqual(completed, seen)
        self.assertEqual([p['key'] for p in failed], ['Wbb'])
        self.assertFalse(live)

    def test_all_source_cards_use_sampled_components_and_fixed_flow(self):
        from madgraph.various.banner import RunCardNLO
        from madgraph.various.shower_card import ShowerCard
        with tempfile.TemporaryDirectory() as temporary:
            process = Path(temporary)
            (process / 'Cards').mkdir()
            (process / 'SubProcesses').mkdir()
            config = campaign.new_config(process / 'campaign')
            for p in campaign.PROCESSES:
                for delta in (False, True):
                    with self.subTest(process=p['key'], delta=delta):
                        RunCardNLO().write(str(process / 'Cards/run_card.dat'))
                        ShowerCard().write(str(process / 'Cards/shower_card.dat'),
                            template=str(ROOT / 'Template/NLO/Cards/shower_card.dat'))
                        shutil.copyfile(ROOT / 'Template/NLO/Cards/FKS_params.dat',
                                        process / 'SubProcesses/FKS_params.dat')
                        campaign.configure(process, p, delta, config)
                        card = RunCardNLO(str(process / 'Cards/run_card.dat'))
                        self.assertEqual(card['mcatnlo_delta'], delta)
                        self.assertTrue(card['macnlops'])
                        self.assertFalse(card['macnlops_resolved_source'])
                        self.assertEqual(card['nevents'], 1000000)
                        text = (process / 'SubProcesses/FKS_params.dat').read_text()
                        for key, value in campaign.FKS_SETTINGS.items():
                            self.assertIn('#' + key + '\n' + ('.true.' if value else '.false.'), text)
                        if p['key'] == 'Wj':
                            for key, value in dict(ptj=10., etaj=-1., jetalgo=-1., jetradius=.5).items():
                                self.assertEqual(card[key], value)
                            self.assertEqual(card['ptl'], 0.)
                            self.assertEqual(p['born_partons'], 1)
                        if p['key'] == 'single_top':
                            self.assertEqual(card['ptj'], 0.)
                            self.assertEqual(card['etaj'], -1.)

    def test_prepare_writes_no_exports_and_rejects_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / 'new'
            with mock.patch.object(campaign.subprocess, 'Popen', side_effect=AssertionError('No jobs allowed')), \
                    mock.patch.object(campaign.subprocess, 'check_output', return_value=b'test-revision\n'):
                config = campaign.prepare(base)
            campaign.validate_config(config, base)
            self.assertEqual(list((base / 'processes').iterdir()), [])
            self.assertEqual(len(config['processes']), 6)
            self.assertEqual(config['fks_settings'], dict(
                MCExplicitKLSum=False, MCSubtractionAtFixedFlow=True, FKSExplicitSum=False))
            legacy = copy.deepcopy(config)
            legacy['fks_settings']['MCExplicitKLSum'] = True
            with self.assertRaisesRegex(RuntimeError, 'fresh sampled'):
                campaign.validate_config(legacy, base)
            changed = copy.deepcopy(config)
            changed['physics_sha256']['madgraph/various/macnlops.py'] = 'old'
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                campaign.validate_config(changed, base)

    def test_sample_packets_and_signed_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'events.lhe.gz'
            with gzip.open(path, 'wt') as stream:
                for kind, weight in [(1, 2.), (2, -1.)]:
                    stream.write('<event>\n2 1 %s 10 .01 .1\n#aMCatNLO %s\n'
                        '<macnlops_hard_event version="1" partition="sampled">\n'
                        '<macnlops_component version="1">\n</event>\n' % (weight, kind))
            sample = campaign.scan_lhe(path)
            campaign.check_sample(sample, 2)
            self.assertEqual(sample['sumw'], 1.)
            self.assertEqual(sample['sumabs'], 3.)
            self.assertEqual(sample['counts']['negative_H'], 1)
            sample['counts']['partition_sampled'] = 1
            with self.assertRaisesRegex(RuntimeError, 'sampled kl/ij/flow'):
                campaign.check_sample(sample, 2)

    def test_status_reads_saved_campaign_after_a_matching_fix(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = campaign.new_config(base)
            config['physics_sha256']['madgraph/various/macnlops.py'] = 'old'
            campaign.dump(base / 'config.json', config)
            campaign.dump(base / 'status_DY.json', dict(status='failed', stage='delta_validate_matching'))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(campaign.main(['status', '--campaign-dir', str(base)]), 0)
            self.assertIn('original matching code', stream.getvalue())
            self.assertIn('delta_validate_matching', stream.getvalue())
            with self.assertRaisesRegex(RuntimeError, 'source changed'):
                campaign.validate_config(config, base)


class TestAnalysisPreparation(unittest.TestCase):
    def test_wjet_both_modes_first_and_full_use_current_source(self):
        spec = importlib.util.spec_from_file_location('sampled_analysis_tested',
            ROOT / 'docs/audits/analyse_macnlops_campaign.py')
        analysis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(analysis)

        class Source:
            header = 'test profile'

            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def __iter__(self):
                return iter(['test event'])

        def continuation(process, pythia, accepted, contract, output):
            output.write_text('! common measured parton-only settings\n')
            return dict(LD_LIBRARY_PATH='', LHAPDF_DATA_PATH='', PYTHIA8DATA='')

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            config = campaign.new_config(base, events=7)
            p = next(p for p in config['processes'] if p['key'] == 'Wj')
            for mode in ('ordinary', 'delta'):
                run = base / 'processes' / p['directory'] / 'Events' / mode
                accepted = run / 'macnlops/events.lhe.gz'
                accepted.parent.mkdir(parents=True)
                source = run / 'events.lhe.gz'
                source.write_text('source fixture')
                accepted.write_text('accepted fixture')
                channel = accepted.parent / 'channel'
                channel.mkdir()
                campaign.dump(channel / 'report.json', dict(counts=dict(input_H=3, input_S=4)))
                campaign.dump(accepted.with_name('summary.json'), [dict(status='complete', output=str(channel / 'events.lhe.gz'))])
                sample = lambda path, n: dict(path=str(path), sha256=campaign.sha(path),
                                              counts=dict(events=n), sumw=1., sumabs=3.)
                checkpoint = dict(contract=campaign.contract_id(config, p),
                    source=sample(source, 7), macnlops=sample(accepted, 4), settings=dict(iseed=12),
                    veto_prescription=campaign.veto_policy(mode))
                campaign.dump(base / ('Wj_' + mode + '_complete.json'), checkpoint)
            with mock.patch.multiple(analysis, BASE=base / 'validation', PRODUCTION=base, CONFIG=config), \
                    mock.patch.object(analysis.mc, 'LHEStream', Source), \
                    mock.patch.object(analysis.mc, 'Event'), \
                    mock.patch.object(analysis.mc, 'read_shower_profile', return_value={'support': {'flavour_count': 5}}), \
                    mock.patch.object(analysis.mc, 'write_continuation_settings', side_effect=continuation), \
                    contextlib.redirect_stdout(io.StringIO()):
                paths = analysis.prepare(keys=['Wj'])
                self.assertEqual(len(paths), 4)
                for path in paths:
                    job = json.loads(Path(path).read_text())
                    self.assertEqual(job['events'], 7)
                    self.assertEqual(job['analysis_jets']['pt_min_GeV'], 25.)
                    self.assertEqual(job['analysis_jets']['radius'], .5)
                    self.assertIn('TimeShower:nPartonsInBorn = 1', Path(job['source_settings']).read_text())
                    self.assertIn('TimeShower:globalRecoil = off', Path(job['endpoint_settings']).read_text())
                    self.assertIn('system_pt_soft', [o['name'] for o in job['observables']])
                old = base / 'Wj_delta_complete.json'
                checkpoint = json.loads(old.read_text())
                checkpoint['veto_prescription'] = 'global-source-v1'
                campaign.dump(old, checkpoint)
                with self.assertRaisesRegex(RuntimeError, 'old matching'):
                    analysis.prepare(keys=['Wj'])


class TestWJetNative(unittest.TestCase):
    def test_clustering_selection_signed_weights_and_migrations(self):
        """Compile only histogram/observable code; never initialize Pythia."""
        pythia = Path('/export/tmp/rikkert/Pythia/pythia8318')
        if not shutil.which('g++') or not (pythia / 'include/Pythia8/Pythia.h').exists():
            self.skipTest('Requires g++ and the campaign Pythia headers')
        native = (ROOT / 'docs/audits/macnlops_stream_hist.cc').read_text()
        prefix = native[:native.index('class CountReader:')]
        fixture = r'''
Particle part(double pt,double y=0.,double phi=0.){
 return Particle{21,1,pt*std::cos(phi),pt*std::sin(phi),pt*std::sinh(y),pt*std::cosh(y),0.};
}
std::vector<Particle> event(double pt,double y=0.){
 return {Particle{24,1,-30.,0.,0.,std::sqrt(80.*80.+900.),80.},part(pt,y)};
}
int main(){
 fjcore::ClusterSequence::set_fastjet_banner_stream(nullptr);
 require(observables(event(25.),"wjet").empty(),"25 GeV boundary must fail");
 require(observables(event(25.01),"wjet").size()==14,"Jet above 25 must pass");
 require(observables(event(30.,4.5001),"wjet").empty(),"Forward jet must fail");
 require(!observables(event(30.,4.4999),"wjet").empty(),"Central jet must pass");
 auto merged=event(20.);merged.push_back(part(20.,0.,.45));
 require(observables(merged,"wjet").at("object_2_pt")>38.,"R=0.5 must merge these subthreshold partons");
 auto additional=event(30.);additional.push_back(part(10.,0.,2.));additional.push_back(part(70.,5.,1.));
 auto values=observables(additional,"wjet");
 require(std::abs(values.at("object_2_pt")-30.)<1.e-9,"Leading jet must satisfy rapidity selection");
 require(std::abs(values.at("jet_pt")-10.)<1.e-9,"Additional-jet soft diagnostic lost");
 require(std::isnan(values.at("jet_y")),"Soft additional jet rapidity should be missing");
 ptree tree,edges;tree.put("name","system_pt");tree.put("label","");tree.put("unit","GeV");tree.put("log",false);
 for(double edge:{0.,50.,200.}){ptree child;child.put_value(edge);edges.push_back(std::make_pair("",child));}
 tree.add_child("edges",edges);std::vector<Hist> hist;hist.emplace_back(tree);
 std::array<Sum,NS> totals,fiducial;Pairs pairs(3);RawEvent raw;Mask reference{};reference[0]=true;
 Mask endpoint{};endpoint[1]=endpoint[4]=true;
 // Migration into the fiducial region: raw totals still include the source,
 // while the reference contributes neither a histogram nor a covariance.
 raw.weight=-2.;auto failed=fillSelected(raw,event(20.),"wjet",reference,nullptr,hist,totals,fiducial);
 pairs.add(Key{1,2},raw.weight,hist,failed);auto paired=pairs.take(Key{1,2},raw.weight);
 fillSelected(raw,event(30.),"wjet",endpoint,&paired,hist,totals,fiducial);
 require(totals[0].count==1&&fiducial[0].count==0,"Raw and fiducial totals confused");
 require(fiducial[4].w==-2.&&hist[0].w[4*2]==-2.&&hist[0].w2[4*2]==4.,"Signed weights changed");
 require(hist[0].cov[4*NS*2]==0.,"Failed reference generated spurious covariance");
 // Both pass: retain the full same-event covariance, including its transpose.
 raw.weight=3.;values=fillSelected(raw,event(30.),"wjet",reference,nullptr,hist,totals,fiducial);
 pairs.add(Key{2,3},raw.weight,hist,values);paired=pairs.take(Key{2,3},raw.weight);
 fillSelected(raw,event(30.),"wjet",endpoint,&paired,hist,totals,fiducial);
 require(hist[0].cov[4*NS*2]==9.&&hist[0].cov[4*2]==9.,"Paired covariance incorrect");
 // Migration out: consume the pair without adding an endpoint histogram.
 raw.weight=1.;values=fillSelected(raw,event(30.),"wjet",reference,nullptr,hist,totals,fiducial);
 pairs.add(Key{3,4},raw.weight,hist,values);paired=pairs.take(Key{3,4},raw.weight);
 fillSelected(raw,event(20.),"wjet",endpoint,&paired,hist,totals,fiducial);
 require(totals[4].count==3&&fiducial[4].count==2&&hist[0].count[4*2]==2,"Failed endpoint contaminated histograms");
 require(pairs.consumed==3,"Fiducial rejection lost a source identity");
 // Other processes retain their existing inclusive observable definitions.
 auto dy=std::vector<Particle>{{11,1,30.,0.,0.,30.,0.},{-11,1,-30.,0.,0.,30.,0.}};
 require(observables(dy,"dy").at("system_mass")==60.,"DY observable changed");
 std::cout<<"WJET_SELECTION_PASS\n";
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'fixture.cc'
            source.write_text(prefix + fixture)
            binary = directory / 'check'
            command = ['g++', '-O0', '-std=c++17', '-I' + str(pythia / 'include'),
                '-I' + str(ROOT / 'Template/NLO/SubProcesses'), str(source),
                str(ROOT / 'Template/NLO/SubProcesses/fjcore.cc'),
                '-L' + str(pythia / 'lib'), '-Wl,-rpath,' + str(pythia / 'lib'),
                '-lpythia8', '-lcrypto', '-lz', '-o', str(binary)]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('WJET_SELECTION_PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
