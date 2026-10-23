"""All-H qualification and hard-coordinate cache identity/resume controls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from madgraph.various import macnlops as mc
from tests.unit_tests.various.test_macnlops import HEADER,event,hard_annotation

ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location('macnlops_source_density_cache',
    ROOT/'docs/audits/macnlops_source_density_cache.py')
import sys
CACHE=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=CACHE;SPEC.loader.exec_module(CACHE)


def block(kind,weight):
    return event(kind,weight).replace('</event>',hard_annotation(kind)+'</event>')


class Native:
    instances=[]
    def __init__(self,process_dir,log_dir,environment=None):
        self.process_dir=Path(process_dir);self.environment=environment
        self.configuration=None;self.calls=[];self.__class__.instances.append(self)
    def configure(self,beam_xmax,shower_cutoffs):
        self.configuration=tuple(beam_xmax)+tuple(shower_cutoffs)
    def evaluate(self,event,request_id):
        self.calls.append((event.kind,event.weight,request_id))
        if event.kind=='S':return mc.Densities(30.,2.,5.)
        if event.weight>0:return mc.Densities(9.,2.,3.)
        if event.weight<0:return mc.Densities(1.,6.,2.)
        return mc.Densities(2.,2.,0.)
    def __enter__(self):return self
    def __exit__(self,*args):pass


def fixture(folder):
    process=folder/'process';cards=process/'Cards';cards.mkdir(parents=True)
    context=process/'SubProcesses/P0';context.mkdir(parents=True)
    (context/'leshouche_info.dat').write_text('I 1 4 2 -2 11 -11\n')
    (context/'macnlops_evaluator').write_text('TEST ONLY precompiled fake evaluator\n')
    (process/'SubProcesses/driver_macnlops.f').write_text('TEST ONLY driver\n')
    (cards/'run_card.dat').write_text('False = macnlops_resolved_source\n')
    settings=cards/'macnlops_shower.cmnd';settings.write_text('TEST ONLY settings\n')
    support=dict(beam_xmax=[.999,.998],shower_cutoffs=[.5,.6])
    profile=dict(schema='macnlops-generation-profile-v1',construction=mc.ORDINARY_SOURCE,
        support=support,pdf='TEST/0',alpha_s_mz=.118,alpha_s_order=2,
        pdf_info_sha256='1'*64,pdf_member_sha256='2'*64,settings_sha256=mc.sha256(settings))
    (cards/'macnlops_profile.json').write_text(json.dumps(profile))
    source=folder/'source.lhe'
    source.write_text(HEADER+''.join(block(k,w) for k,w in [('S',2.),('H',3.),('H',-4.),('H',0.)])+'</LesHouchesEvents>\n')
    manifest=Path(str(source)+'.macnlops.json')
    pins={str(path.relative_to(process)):mc.sha256(path) for path in cards.iterdir()}
    pins['SubProcesses/driver_macnlops.f']=mc.sha256(process/'SubProcesses/driver_macnlops.f')
    manifest.write_text(json.dumps(dict(schema='macnlops-source-v2',sources=pins,lhe_sha256=mc.lhe_sha256(source))))
    configuration=dict(support=support,pdf=profile['pdf'],alpha=profile['alpha_s_mz'],order=profile['alpha_s_order'],
        settings=str(settings),pdf_info_sha256=profile['pdf_info_sha256'],pdf_member_sha256=profile['pdf_member_sha256'],
        environment=dict(OMP_NUM_THREADS='1'),source_manifest=CACHE.artifact(manifest))
    return source,process,configuration


class TestSourceDensityCache(unittest.TestCase):
    def setUp(self):
        Native.instances=[]
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.source,self.process,self.configuration=fixture(self.folder)
        self.output=self.folder/'gate'
        self.patch=mock.patch.object(CACHE.mc,'NativeEvaluator',Native);self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def gate(self):return CACHE.gate_chunk(self.source,self.process,self.configuration,self.output)
    def cached(self,report=None):
        report=report or self.gate()
        native=Native(self.process,self.folder,environment=self.configuration['environment'])
        cached=CACHE.CachedSourceEvaluator(native,report['cache_path'],report['report_sha256'])
        cached.configure(**{key:self.configuration['support'][key] for key in ('beam_xmax','shower_cutoffs')})
        return native,cached

    def test_every_h_both_signs_and_zero_weight_is_gated_without_bound_clamp(self):
        before=mc.sha256(self.source);report=self.gate()
        self.assertTrue(report['passed']);self.assertTrue(report['complete_source'])
        self.assertEqual(report['counts']['source_events'],4)
        self.assertEqual(report['counts']['source_H'],report['counts']['evaluated_H'])
        self.assertEqual(report['counts']['evaluated_H'],3)
        self.assertEqual([call[1] for call in Native.instances[0].calls],[3.,-4.,0.])
        self.assertEqual(report['informational']['requires_signed_residual_H'],1)
        self.assertEqual(report['informational']['max_negative_H_over_K'],2.5)
        self.assertEqual(mc.sha256(self.source),before)
        rows=[json.loads(line) for line in Path(report['cache_path']).read_text().splitlines()]
        self.assertEqual(rows[1]['densities'],dict(real=1.,subtraction=6.,shower=2.))

    def test_resume_and_verify_reuse_complete_checkpoint_without_any_h_rpc(self):
        a=self.gate();b=self.gate();c=CACHE.verify_gate(self.source,self.process,self.configuration,self.output)
        self.assertEqual(a,b);self.assertEqual(a,c);self.assertEqual(len(Native.instances),1)

    def test_h_reuses_hard_point_despite_different_body_and_s_delegates_live(self):
        report=self.gate();native,cached=self.cached(report)
        body=event('H',-4.).replace('100 0 0 100 0 0 9','98 0 0 100 0.2 0 9')
        endpoint=body.replace('</event>',hard_annotation('H')+'<macnlops_endpoint version="1" origin="hard_annotation" stage="first"/>\n</event>')
        self.assertEqual(cached.evaluate(mc.Event(endpoint),99),mc.Densities(1.,6.,2.))
        self.assertEqual(native.calls,[])
        self.assertEqual(cached.evaluate(mc.Event(block('S',2.)),100),mc.Densities(30.,2.,5.))
        self.assertEqual(native.calls,[('S',2.,100)])

    def test_changed_header_hard_geometry_or_latent_labels_never_falls_back_to_rpc(self):
        _,cached=self.cached();original=block('H',-4.)
        changes=[block('H',-5.),original.replace('4 1 -4.000000000000e+00','4 2 -4.000000000000e+00'),
            original.replace('100 0.007297 0.118','101 0.007297 0.118'),
            original.replace('100 0.007297 0.118','100 0.007298 0.118'),
            original.replace('100 0.007297 0.118','100 0.007297 0.119'),
            original.replace('4 2 1 1 1 1 1 0 1 0 0 1 1 1','4 2 2 1 1 1 1 0 1 0 0 1 1 1'),
            original.replace(hard_annotation('H'),hard_annotation('H').replace('100 0 0 100 0 0 9','101 0 0 100 0 0 9'))]
        for changed in changes:
            self.assertNotEqual(original,changed)
            with self.subTest(changed=changed),self.assertRaisesRegex(mc.MatchingError,'does not match'):
                cached.evaluate(mc.Event(changed),1)
        self.assertEqual(cached.native.calls,[])

    def test_evolved_h_endpoint_cannot_borrow_its_original_h_annotation(self):
        _,cached=self.cached()
        changed=block('H',-4.).replace('</event>',
            '<macnlops_endpoint version="1" origin="shower" stage="first"/>\n</event>')
        with self.assertRaisesRegex(mc.MatchingError,'evolved shower point'):
            cached.evaluate(mc.Event(changed),1)

    def test_zero_k_negative_h_requires_signed_residual_and_is_not_failure(self):
        with mock.patch.object(Native,'evaluate',return_value=mc.Densities(1.,6.,0.)):
            # Retain only the negative H; positive H would correctly fail.
            self.source.write_text(HEADER+block('H',-4.)+'</LesHouchesEvents>\n')
            manifest=Path(self.configuration['source_manifest']['path']);record=json.loads(manifest.read_text())
            record['lhe_sha256']=mc.lhe_sha256(self.source);manifest.write_text(json.dumps(record))
            self.configuration['source_manifest']=CACHE.artifact(manifest)
            report=self.gate()
        self.assertTrue(report['passed'])
        self.assertIsNone(report['informational']['max_negative_H_over_K'])
        self.assertEqual(report['informational']['negative_H_with_zero_K'],1)
        self.assertEqual(report['informational']['max_required_H_survival'],1.)

    def test_wrong_sign_and_zero_global_h_with_nonzero_weight_fail_closed(self):
        for index,density in enumerate((mc.Densities(1.,6.,2.),mc.Densities(2.,2.,2.),
                                       mc.Densities(-1.,2.,2.),mc.Densities(1.,2.,-1.),
                                       mc.Densities(float('nan'),2.,2.))):
            output=self.folder/('invalid%d'%index)
            with mock.patch.object(Native,'evaluate',return_value=density),self.assertRaisesRegex(mc.MatchingError,'gate failed'):
                CACHE.gate_chunk(self.source,self.process,self.configuration,output)
            report=json.loads((output/CACHE.REPORT_NAME).read_text())
            self.assertFalse(report['passed']);self.assertTrue(report['complete_source'])
            self.assertEqual(report['counts']['evaluated_H'],3)
            self.assertTrue((output/'first_failure.lhe').is_file())
            with self.assertRaises(mc.MatchingError):CACHE.CachedSourceEvaluator(Native(self.process,self.folder),output/CACHE.CACHE_NAME)

    def test_stale_chunk_profile_configuration_manifest_or_binary_cannot_resume(self):
        self.gate()
        paths=[self.source,self.process/'Cards/macnlops_profile.json',
            Path(self.configuration['source_manifest']['path']),self.process/'SubProcesses/P0/macnlops_evaluator',
            self.process/'SubProcesses/driver_macnlops.f']
        for path in paths:
            old=path.read_bytes();path.write_bytes(old+b'\n')
            with self.subTest(path=path),self.assertRaises(mc.MatchingError):self.gate()
            path.write_bytes(old)
        changed=dict(self.configuration,alpha=.119)
        with self.assertRaisesRegex(mc.MatchingError,'differs from profile'):
            CACHE.gate_chunk(self.source,self.process,changed,self.output)

    def test_cache_and_external_report_hash_pins_detect_coherent_tampering(self):
        report=self.gate();path=Path(report['cache_path']);original=path.read_text()
        rows=[json.loads(line) for line in original.splitlines()];rows[0]['densities']['real']+=1.
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(mc.MatchingError,'cache bytes changed'):self.cached(report)
        report_path=self.output/CACHE.REPORT_NAME;stored=json.loads(report_path.read_text())
        stored['cache_sha256']=mc.sha256(path);report_path.write_text(json.dumps(stored))
        with self.assertRaisesRegex(mc.MatchingError,'report changed after qualification'):self.cached(report)

    def test_wrong_live_support_environment_and_process_are_rejected(self):
        report=self.gate();native,cached=self.cached(report)
        with self.assertRaisesRegex(mc.MatchingError,'support differs'):cached.configure([1.,1.],[.5,.6])
        native.environment['OMP_NUM_THREADS']='2'
        with self.assertRaisesRegex(mc.MatchingError,'environment differs'):CACHE.CachedSourceEvaluator(native,report['cache_path'])
        native.environment['OMP_NUM_THREADS']='1';native.process_dir=self.folder
        with self.assertRaisesRegex(mc.MatchingError,'another generated process'):CACHE.CachedSourceEvaluator(native,report['cache_path'])

    def test_orphan_cache_is_preserved_not_overwritten_on_restart(self):
        self.output.mkdir();path=self.output/CACHE.CACHE_NAME;path.write_text('orphan evidence\n')
        with self.assertRaisesRegex(mc.MatchingError,'Uncheckpointed'):self.gate()
        self.assertEqual(path.read_text(),'orphan evidence\n')

    def test_native_error_preserves_the_exact_event_and_never_qualifies_partial_cache(self):
        before=self.source.read_bytes()
        with mock.patch.object(Native,'evaluate',side_effect=mc.MatchingError('native test failure')):
            with self.assertRaisesRegex(mc.MatchingError,'gate failed'):self.gate()
        report=json.loads((self.output/CACHE.REPORT_NAME).read_text())
        self.assertFalse(report['passed']);self.assertFalse(report['complete_source'])
        self.assertEqual(report['first_failure']['sequence'],2)
        self.assertEqual(report['counts']['evaluated_H'],0)
        self.assertIn(block('H',3.),(self.output/'first_failure.lhe').read_text())
        self.assertEqual(self.source.read_bytes(),before)


if __name__=='__main__':unittest.main()
