"""Small metadata regression: gzip encoding changes do not change identity."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from resume_identity import file_sha,payload_sha,verify_resume


class ResumeIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.base=Path(self.temporary.name)
        self.source=self.base/'events.lhe.gz';self.summary=self.base/'first.json'
        self.audit=self.base/'encoding_audit.json'

    def tearDown(self):self.temporary.cleanup()

    def write(self,payload,mtime,level):
        with self.source.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=mtime,compresslevel=level) as out:
                out.write(payload)

    def initialise(self):
        self.write(b'<event>same signed event payload</event>\n',1,1)
        value=dict(diagnostic_input_sha256=file_sha(self.source),
            diagnostic_payload_sha256=payload_sha(self.source),counts=dict(events=1),totals_pb=dict(source=1.))
        self.summary.write_text(json.dumps(value));return value

    def test_identical_reencoding_is_accepted(self):
        original=self.initialise();self.write(b'<event>same signed event payload</event>\n',2,9)
        self.assertNotEqual(file_sha(self.source),original['diagnostic_input_sha256'])
        self.assertTrue(verify_resume(self.source,self.summary,self.audit)['verified'])

    def test_changed_payload_is_rejected(self):
        self.initialise();self.write(b'<event>changed physical payload</event>\n',2,9)
        with self.assertRaisesRegex(RuntimeError,'payload changed'):
            verify_resume(self.source,self.summary,self.audit)

    def test_legacy_adoption_requires_exact_audit_binding(self):
        original=self.initialise();original.pop('diagnostic_payload_sha256')
        self.summary.write_text(json.dumps(original))
        with self.assertRaisesRegex(RuntimeError,'Legacy diagnostics have no payload digest'):
            verify_resume(self.source,self.summary,self.audit)
        self.audit.write_text(json.dumps(dict(schema='native-tc-encoding-audit-v2',first_summary_sha256=file_sha(self.summary),
            first_diagnostic_gzip_sha256=original['diagnostic_input_sha256'],
            current_uncompressed_sha256=payload_sha(self.source),
            weight_sequence_audit=dict(passed=True,events=1,weight_sequence_sha256='a'*64,
                per_event_identities_passed=True,count_checks=dict(events=dict(first=1,current=1,passed=True)),
                weight_sum_checks=dict(source=dict(first=1.,current=1.,passed=True))))))
        self.assertTrue(verify_resume(self.source,self.summary,self.audit)['verified'])
        self.summary.write_text(self.summary.read_text()+'\n')
        with self.assertRaisesRegex(RuntimeError,'different first.json'):
            verify_resume(self.source,self.summary,self.audit)


if __name__=='__main__':unittest.main()
