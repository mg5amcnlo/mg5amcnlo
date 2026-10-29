#!/usr/bin/env python3
"""Check actual flavour-resolved paired H/S weights in a compiled source.

Usage: python3 docs/audits/check_macnlops_source_pairs.py SUBPROCESS_DIR

The directory must already contain a compiled macnlops_evaluator and use
macnlops=True with its probed shower profile. This audit compiles a temporary
driver and checks R_H+R_S=R and MC_H+MC_S=0 before outer H repartitioning.
It exercises every subprocess flavour at the supplied physical real points.
The default retains ordinary G/P. --resolved-source checks the explicit
legacy convex construction and its additional negative-H envelope bound.
"""
import argparse
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from docs.audits.check_macnlops_source_h import ROOT, evaluator_source, link_audit_object, points


def check_source_pairs(subprocess_dir, point_records=None, profile=None,
                       resolved_source=False):
    directory = Path(subprocess_dir).resolve()
    source = evaluator_source(directory).replace(
        'program macnlops_evaluator', 'program macnlops_source_pairs_audit', 1)
    initialize = '      call setrun\n'
    profile_code = ('      macnlops_resolved_source=%s\n' %
                    ('.true.' if resolved_source else '.false.'))
    if profile is not None:
        if len(profile) != 6 or not all(map(math.isfinite, profile)):
            raise ValueError('profile needs six finite cutoffs, masses and beam bounds')
        names = ('macnlops_isr_cutoff', 'macnlops_fsr_cutoff',
                 'macnlops_charm_mass', 'macnlops_bottom_mass',
                 'macnlops_beam_xmax(1)', 'macnlops_beam_xmax(2)')
        profile_code += '      macnlops=.true.\n' + ''.join(
            '      %s=%s\n' % (name, ('%.17e' % value).replace('e', 'd'))
            for name, value in zip(names, profile))
    source = source.replace(initialize, initialize + profile_code +
        "      if(.not.macnlops)stop 'Source-pair audit needs macnlops'\n", 1)
    declaration = '      double precision lab(0:3,nexternal),real_density,'
    if source.count(declaration) != 1:
        raise RuntimeError('Evaluator declarations changed: update the audit adapter')
    source = source.replace(declaration,
        '      double precision source_pairs(6),lab(0:3,nexternal),real_density,', 1)
    output = "         write(*,'(a,2(1x,i12),3(1x,es26.17e3))')"
    if source.count(output) != 1:
        raise RuntimeError('Evaluator output changed: update the audit adapter')
    source = source.replace(output,
        '         call source_pairs_audit(owner,ids,lab,source_pairs)\n'
        "         write(*,'(a,1x,i5,6(1x,es26.17e3))')\n"
        "     $     'SOURCE_PAIRS_AUDIT',request,source_pairs\n" + output, 1)
    source += '\n' + (ROOT / 'tests/input_files/macnlops_source_pairs_audit.f').read_text()
    dry_run = subprocess.run(['make', '-n', '-W', 'driver_macnlops.o',
                             'macnlops_evaluator'], cwd=directory, check=True,
                             capture_output=True, text=True).stdout
    links = [line for line in dry_run.splitlines() if ' -o macnlops_evaluator ' in line]
    if len(links) != 1:
        raise RuntimeError('Build the evaluator before running its source-pair audit')
    command = shlex.split(links[0])
    requests = points() if point_records is None else point_records
    # The exported LHAPDF wrapper locates lib/PDFsets from argv[0]. Keep
    # the temporary executable inside this export, as the native driver is.
    with tempfile.TemporaryDirectory(prefix='.macnlops-source-pairs-',
                                     dir=directory) as temporary:
        temporary = Path(temporary)
        src, obj, executable = (temporary / name for name in ('audit.f', 'audit.o', 'audit'))
        src.write_text(source)
        subprocess.run(['gfortran', '-O', '-fPIC', '-ffixed-line-length-none',
                        '-fno-common', '-I.', '-I../../lib', '-c', str(src),
                        '-o', str(obj)], cwd=directory, check=True,
                       capture_output=True, text=True)
        command[command.index('-o') + 1] = str(executable)
        link_audit_object(command, obj)
        command += ['-Wl,-rpath,' + str(directory.parent.parent / 'lib')]
        subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True)
        result = subprocess.run([str(executable)], cwd=directory,
                                input=''.join(requests), capture_output=True,
                                text=True, timeout=120, check=True)
    records, densities = [], []
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and fields[0] == 'SOURCE_PAIRS_AUDIT':
            records.append(tuple(float(x) for x in fields[2:]))
        if fields and fields[0] == 'MACNLOPS_DENSITY_V1':
            densities.append((int(fields[2]), *[float(x) for x in fields[3:]]))
    if len(records) != len(requests) or len(densities) != len(requests):
        raise AssertionError('Source audit did not evaluate every point:\n' + result.stdout[-12000:])
    real_error = mc_error = 0.
    transferred = resolved = 0
    probne_values, hard_real_fractions = [], []
    for hard_real, soft_real, soft_mc, hard_mc, real, probne in records:
        if real <= 0. or not all(map(math.isfinite,
                                     (hard_real, soft_real, soft_mc, hard_mc, real))):
            raise AssertionError('Nonfinite paired source contribution')
        real_error = max(real_error, abs(hard_real + soft_real - real) / max(abs(real), 1.))
        mc_error = max(mc_error, abs(soft_mc + hard_mc) / max(abs(soft_mc), abs(hard_mc), 1.))
        transferred += soft_real != 0.
        resolved += hard_real != 0.
        probne_values.append(probne)
        hard_real_fractions.append(hard_real / real)
        if not resolved_source and abs(hard_real / real - probne) > 1.e-11:
            raise AssertionError('Ordinary real H/S split lost its history P')
    if real_error > 1.e-11 or mc_error > 1.e-11:
        raise AssertionError('Source pairing failed: R %.17g, MC %.17g' % (real_error, mc_error))
    bound_error = 0.
    for status, real, subtraction, kernel in densities:
        if status or not all(map(math.isfinite, (real, subtraction, kernel))):
            raise AssertionError('Invalid complete density in paired source audit')
        bound_error = max(bound_error,
            (max(subtraction - real, 0.) - kernel) / max(abs(real), abs(subtraction), 1.))
    if resolved_source and bound_error > 1.e-11:
        raise AssertionError('Pure source negative H exceeds its shower kernel')
    return {'points': len(records), 'transferred_real_points': transferred,
            'resolved_real_points': resolved, 'max_real_pair_error': real_error,
            'max_mc_pair_error': mc_error, 'max_bound_excess': bound_error,
            'resolved_source': resolved_source,
            'probne_values': probne_values,
            'hard_real_fractions': hard_real_fractions,
            'densities': densities}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subprocess_dir', type=Path)
    parser.add_argument('--resolved-source', action='store_true')
    args = parser.parse_args()
    print(check_source_pairs(args.subprocess_dir,
                             resolved_source=args.resolved_source))
