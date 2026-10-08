"""Record stream provenance in an isolated physics replay, without RNG changes.

Usage: python instrument_streams.py /tmp/export/SubProcesses/ampli_mint_adapter.f90
The diagnostic is deliberately not part of production source or the pool format.
"""

from pathlib import Path
import argparse
import difflib
import hashlib
import json


def instrument(path):
    path = Path(path).resolve()
    repository = Path(__file__).resolve().parents[3]
    if path == repository / 'Template/NLO/SubProcesses/ampli_mint_adapter.f90':
        raise ValueError('Refusing to instrument production template')
    before = path.read_text()
    after = before
    replacements = [
        ('  integer(kind=8),save :: production_points=0_8',
         '  integer,save :: stream_trace_unit=0\n'
         '  integer(kind=8),save :: production_points=0_8'),
        ('    production_points=0_8\n',
         "    open(newunit=stream_trace_unit,file='ampli_stream_trials.dat',status='replace',action='write')\n"
         "    write(stream_trace_unit,'(a)') '# trial epoch stream probability abs_weight signed_weight stored candidate'\n"
         '    production_points=0_8\n'),
        ('    call sampler%native_consider(point,x(1:ndim),to_write,iteration_done)',
         '    call sampler%native_consider(point,x(1:ndim),to_write,iteration_done)\n'
         "    write(stream_trace_unit,'(i12,1x,i6,1x,a4,3(1x,es25.16),1x,l1,1x,i12)') &\n"
         '         production_points,sampler%native_iteration,abrv,probability,point,to_write,sampler%ncandidates'),
        ('    call sampler%write_native_pool(iu)',
         '    call sampler%write_native_pool(iu)\n'
         '    close(stream_trace_unit)'),
    ]
    for old, new in replacements:
        if after.count(old) != 1:
            raise ValueError((path, old, after.count(old)))
        after = after.replace(old, new)
    path.write_text(after)
    return {
        'path': str(path),
        'source_sha256': hashlib.sha256(before.encode()).hexdigest(),
        'instrumented_sha256': hashlib.sha256(after.encode()).hexdigest(),
        'patch': ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                              fromfile=path.name, tofile=path.name)),
        'columns': ['trial', 'epoch', 'stream', 'probability', 'abs_weight',
                    'signed_weight', 'stored', 'candidate'],
        'note': 'Weights already include inverse stream probability and sampling Jacobian. '
                'Multiply weights by probability for conditional-stream weights. '
                'Candidate is the running count, only identifies a candidate when stored=T.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('adapter', type=Path)
    parser.add_argument('--provenance', type=Path)
    args = parser.parse_args()
    result = instrument(args.adapter)
    if args.provenance:
        args.provenance.write_text(json.dumps(result, indent=2) + '\n')
    else:
        print(json.dumps(result, indent=2))
