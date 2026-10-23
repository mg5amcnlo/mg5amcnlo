"""Independent exported-data checks for shared ordinary MC@NLO comparisons."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.unit_tests.various.test_macnlops_plots import (
    ROOT,PLOTS,HAVE_FASTJET,fixture,common_fixture)

SPEC=importlib.util.spec_from_file_location('macnlops_histogram_audit',
    ROOT/'docs/audits/audit_macnlops_validation.py')
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


class TestMacNLOPSOriginalWeightAnchors(unittest.TestCase):
    def test_common_source_anchors_both_curves_with_one_original_read(self):
        integral=dict(processes=dict(dy=dict(primary=dict(source='/original/ordinary.lhe.gz'))))
        weights=dict(events=200000,sumw=5.,sumw2=7.)
        with mock.patch.object(AUDIT,'source_weights',return_value=weights) as reader:
            result=AUDIT.original_sources('dy',True,integral)
        reader.assert_called_once_with('/original/ordinary.lhe.gz')
        self.assertIs(result['primary'],result['reference'])
        integral['processes']['dy']['reference']=dict(source='/independent/reference.lhe.gz')
        with mock.patch.object(AUDIT,'source_weights',return_value=weights),self.assertRaisesRegex(AUDIT.AuditFailure,'second integration source'):
            AUDIT.original_sources('dy',True,integral)

    def test_original_reader_uses_central_header_and_keeps_signed_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'ordinary.lhe'
            path.write_text('<LesHouchesEvents>\n'+''.join('<event>\n# header comment\n2 1 %sD+00 91.2 .007 .118\n<rwgt><wgt id="other">999</wgt></rwgt>\n</event>\n'%w for w in (2.,-3.,4.))+'</LesHouchesEvents>\n')
            result=AUDIT.source_weights(path)
            self.assertEqual((result['events'],result['negative_events']),(3,1))
            self.assertEqual((result['sumw'],result['sumw2']),(3.,29.))


@unittest.skipUnless(HAVE_FASTJET,'analysis venv with awkward, vector and fastjet is required')
class TestMacNLOPSHistogramAudit(unittest.TestCase):
    def output(self,folder,common=True):
        config=(common_fixture if common else fixture)(folder)
        datasets=folder/'datasets.json';datasets.write_text(json.dumps(dict(processes=dict(dy=config))))
        output=folder/'plots'
        PLOTS.run_analysis(datasets,output,allow_incomplete=True,render=False)
        return output

    def rewrite_npz(self,output,mutate):
        file=output/'dy/first/histograms.npz'
        with np.load(file,allow_pickle=False) as raw:arrays={key:raw[key].copy() for key in raw.files}
        mutate(arrays);np.savez_compressed(file,**arrays)

    def test_common_source_and_signed_residuals_close_without_final_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary));report=AUDIT.audit(output,allow_incomplete=True)
            self.assertTrue(report['passed']);self.assertFalse(report['complete_validation'])
            self.assertFalse(report['original_source_weights_checked'])
            for level,record in report['processes']['dy']['levels'].items():
                self.assertEqual(record['negative_H_survivors'],1)
                self.assertEqual(record['negative_S_survivors'],1)
                self.assertEqual(record['retained_negative_H_sumw'],-1.)
                self.assertEqual(record['source_events'],record['reference_events'])
            with self.assertRaisesRegex(AUDIT.AuditFailure,'incomplete preview'):
                AUDIT.audit(output)

    def test_independent_cohorts_still_close(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary),False);report=AUDIT.audit(output,allow_incomplete=True)
            self.assertFalse(report['processes']['dy']['common_source_mode'])
            self.assertEqual(report['processes']['dy']['levels']['first']['negative_H_survivors'],0)

    def test_shared_reference_cannot_claim_independent_covariance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary))
            def mutate(arrays):
                covariance=arrays['jet_pt__covariance'];covariance[0,1:]=0.;covariance[1:,0]=0.
            self.rewrite_npz(output,mutate)
            with self.assertRaisesRegex(AUDIT.AuditFailure,'shared reference covariance'):
                AUDIT.audit(output,allow_incomplete=True)

    def test_common_baseline_chunk_binding_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary));path=output/'summary.json';summary=json.loads(path.read_text())
            summary['processes']['dy']['levels']['first']['chunks']['reference'][0]['path']+='different'
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(AUDIT.AuditFailure,'baseline chunk path differs'):
                AUDIT.audit(output,allow_incomplete=True)

    def test_ratio_error_cannot_double_count_the_same_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary))
            self.rewrite_npz(output,lambda arrays:arrays['jet_pt__ratio_variance_input_to_reference'].__setitem__(0,.1))
            with self.assertRaisesRegex(AUDIT.AuditFailure,'identical baseline ratio has uncertainty'):
                AUDIT.audit(output,allow_incomplete=True)

    def test_negative_h_residual_cannot_be_reported_as_all_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary));path=output/'summary.json';summary=json.loads(path.read_text())
            summary['processes']['dy']['configuration']['signed_residuals']=False
            for record in summary['processes']['dy']['levels'].values():record['signed_residuals']=False
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(AUDIT.AuditFailure,'undeclared negative H survived'):
                AUDIT.audit(output,allow_incomplete=True)

    def test_full_publication_requires_both_endpoints_in_each_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary));path=output/'summary.json';summary=json.loads(path.read_text())
            summary['complete']=True;summary['smoke']=False
            source=summary['processes']['dy']
            summary['processes']={name:copy.deepcopy(source) for name in ('dy','zz','ttbar','wbb')}
            del summary['processes']['ttbar']['levels']['full'];path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(AUDIT.AuditFailure,'ttbar: final report lacks first/full'):
                AUDIT.audit(output)

    def test_portable_flow_table_cannot_hide_missing_soft_jets(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=self.output(Path(temporary));path=output/'dy/first/range_flows.csv'
            with path.open() as stream:rows=list(csv.DictReader(stream))
            for row in rows:
                if row['observable']=='jet_y' and row['stage']=='input':row['missing_events']='0'
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(AUDIT.AuditFailure,'range-flow CSV missing count'):
                AUDIT.audit(output,allow_incomplete=True)


if __name__=='__main__':unittest.main()
