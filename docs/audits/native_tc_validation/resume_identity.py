"""Payload-based resume identities, independent of gzip container encoding."""
import gzip
import hashlib
import json
import math
from pathlib import Path


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def payload_sha(path):
    h=hashlib.sha256()
    with gzip.open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def verify_resume(source,summary,audit_path):
    """Verify a new payload digest or a tightly bound legacy encoding audit.

    Historical first.json files stay immutable. A legacy adoption is returned
    as a separate verification record rather than rewriting their old epoch.
    """
    source,summary,audit_path=map(Path,(source,summary,audit_path))
    old=json.loads(summary.read_text());expected=old.get('diagnostic_payload_sha256')
    current=payload_sha(source);audit_binding=None
    if expected is None:
        if not audit_path.is_file():
            raise RuntimeError('Legacy diagnostics have no payload digest. Run encoding_audit.py to verify the current weight sequence/counts and bind its payload digest to this exact first.json; no event regeneration is needed.')
        audit=json.loads(audit_path.read_text())
        checks=audit.get('weight_sequence_audit',{})
        sequence=checks.get('weight_sequence_sha256','')
        count_checks=checks.get('count_checks',{})
        sum_checks=checks.get('weight_sum_checks',{})
        counts_ok=all(count_checks.get(key,{}).get('passed')
            and count_checks[key].get('first')==value
            and count_checks[key].get('current')==value for key,value in old['counts'].items())
        sums_ok=all(sum_checks.get(key,{}).get('passed')
            and sum_checks[key].get('first')==value
            and math.isclose(sum_checks[key].get('current',float('nan')),value,rel_tol=1.e-11,abs_tol=1.e-11)
            for key,value in old.get('totals_pb',{}).items())
        if (audit.get('schema')!='native-tc-encoding-audit-v2'
            or audit.get('first_summary_sha256')!=file_sha(summary)
            or audit.get('first_diagnostic_gzip_sha256')!=old.get('diagnostic_input_sha256')
            or not checks.get('passed')
            or not checks.get('per_event_identities_passed') or not counts_ok or not sums_ok
            or len(sequence)!=64 or any(c not in '0123456789abcdef' for c in sequence)
            or checks.get('events')!=old['counts']['events']):
            raise RuntimeError('Legacy encoding audit is missing verified weight/count checks or is bound to a different first.json. Re-run encoding_audit.py against the unchanged completed sample.')
        expected=audit['current_uncompressed_sha256']
        audit_binding=dict(path=str(audit_path),sha256=file_sha(audit_path))
    if current!=expected:
        raise RuntimeError('Canonical LHE payload changed since the verified diagnostic epoch; gzip re-encoding alone is allowed, event-payload changes are rejected.')
    return dict(verified=True,payload_sha256=current,
        original_gzip_encoding_sha256=old['diagnostic_input_sha256'],
        current_gzip_encoding_sha256=file_sha(source),legacy_audit=audit_binding)
