#!/usr/bin/env python3
"""Interference (squared split orders) on the GPU backend of madmatrix, and in mg7 on it.

Run by interference_gpu.sh (job interference of gpu_runner_ci.yml) in its run directory.
Any CPU backend (scalar, simd_128, ...) works too, to try the script without a GPU.

1. standalone ``u u~ > u u~`` with QED^2==0, ==2, ==4 and <=4, FPTYPE=d, at the classic
   RAMBO point (1000 GeV) of check_sa.exe:
   - QED^2==2, the QCD x EW interference alone, is the Fortran split-order value
     -5.5828746494657265e-02 (an unmasked colour sum would give the total, +2.776);
   - the three components add up to the QED^2<=4 total of the same build: this pins the
     pair loop of the colour sum (all ORDERED pairs of amplitude orders, not a triangle);
   and QED^2==2 again with FPTYPE=m (colour algebra in single precision).
2. the helicity drawn for an interference |M|^2 through umami (event generation), on
   ``u u~ > t t~ g QED^2==2``, whose helicities have both signs at most points, with
   check_sa.exe perf on a grid of random numbers for each RAMBO point (--momenta,
   --rnd-hel, --dump-me, --dump-hel): every draw has |M|^2 = sum|T|, a helicity always
   comes with the same sign, the average over the uniform grid is the helicity sum (what
   umami returns without the random number) and the points with mixed signs give both
   signs; with the umami parameter interference_helicity_summed (run card
   interference_helicity = "summed", --param) every draw returns the helicity sum.
3. mg7 ``p p > u u~ QCD^2==2`` on the backend: the negative cross section of the CPU
   reference, the LHE <init> declaring the negative weights (IDWTUP = -4, XMAXUP the
   unit weight sigma_abs) and events of both signs. It runs with the PDF set of the
   reference, whatever the workflow's pdf_set (--print-reference-pdf names it, for
   the driver to download it).

Writes summary.txt (key=value lines) in the current directory; exit status 1 if a
check fails.
"""

import argparse
import array
import glob
import gzip
import json
import math
import os
import re
import subprocess
import sys
import time

GPU_BACKENDS = ('cuda', 'hip')

# u u~ > u u~ QED^2==2 at the check_sa RAMBO point, from the Fortran split-order
# standalone (madmatrix CPU: -5.5828746494657258e-02)
UUX_INTERFERENCE = -5.5828746494657265e-02
# mg7 p p > u u~ QCD^2==2, fixed scale 91.188 GeV, NNPDF23_lo_as_0130_qed, default cuts:
# three CPU runs of 200k events (tests/acceptance_tests/test_mg7_interference.py)
UUX_MG7_CROSS, UUX_MG7_ERROR = -12253., 10.
UUX_MG7_PDF = 'NNPDF23_lo_as_0130_qed'

summary = {}
failures = []


def section(title):
    print('\n=================== %s (%s)' % (title, time.strftime('%H:%M:%S')), flush=True)


def check(name, ok, detail):
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)
    summary[name] = ('ok' if ok else 'FAILED') + ' (%s)' % detail
    if not ok:
        failures.append(name)


def run(cmd, cwd=None, log=None):
    with open(log or os.devnull, 'w') as out:
        return subprocess.call(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)


def madgraph(repo, name, lines):
    """Run bin/madgraph on these commands; name.mg5 and name.log in the run directory."""
    with open(name + '.mg5', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if run([sys.executable, os.path.join(repo, 'bin', 'madgraph'), name + '.mg5'],
           log=name + '.log') != 0:
        raise RuntimeError('madgraph failed, see %s.log' % name)


def standalone(repo, backend, name, process, fptype):
    """output standalone + make: the P* directory."""
    madgraph(repo, name, ['import model sm', 'generate ' + process,
                          'output standalone %s -f' % name])
    proc_dirs = sorted(glob.glob(os.path.join(name, 'SubProcesses', 'P*')))
    if not proc_dirs:
        raise RuntimeError('no subprocess directory for %s' % process)
    jobs = os.environ.get('SLURM_CPUS_PER_TASK', '4')
    if run(['make', '-j', jobs, 'BACKEND=' + backend, 'FPTYPE=' + fptype],
           cwd=proc_dirs[0], log=name + '_make.log') != 0:
        raise RuntimeError('make failed, see %s_make.log' % name)
    return proc_dirs[0]


def matrix_element(proc_dir):
    """|M|^2 of check_sa.exe at its classic RAMBO point (1000 GeV)."""
    out = subprocess.run(['./check_sa.exe', '1000'], cwd=proc_dir, capture_output=True,
                         text=True).stdout
    found = re.findall(r'Matrix element\s*=\s*([\d.eE+-]+)\s*GeV', out)
    if not found:
        raise RuntimeError('no matrix element from %s/check_sa.exe:\n%s' % (proc_dir, out))
    return float(found[0])


# ---------------------------------------------------------------------------
# 1. standalone squared-order components
# ---------------------------------------------------------------------------

def check_standalone(repo, backend):
    section('standalone u u~ > u u~ squared-order components (%s)' % backend)
    values = {}
    for tag, constraint in (('qed2', 'QED^2==2'), ('qed0', 'QED^2==0'),
                            ('qed4', 'QED^2==4'), ('total', 'QED^2<=4')):
        proc = standalone(repo, backend, 'uux_%s_d' % tag, 'u u~ > u u~ ' + constraint, 'd')
        values[tag] = matrix_element(proc)
        print('  %-9s FPTYPE=d  %.16e' % (constraint, values[tag]))
    rel = abs(values['qed2'] - UUX_INTERFERENCE) / abs(UUX_INTERFERENCE)
    check('uux_interference_d', rel < 1e-9,
          '%.16e vs Fortran %.16e, rel %.1e' % (values['qed2'], UUX_INTERFERENCE, rel))
    parts = values['qed0'] + values['qed2'] + values['qed4']
    rel = abs(parts - values['total']) / abs(values['total'])
    check('uux_sum_rule_d', rel < 1e-12,
          'QED=0+2+4 %.16e vs QED^2<=4 %.16e, rel %.1e' % (parts, values['total'], rel))

    proc = standalone(repo, backend, 'uux_qed2_m', 'u u~ > u u~ QED^2==2', 'm')
    value = matrix_element(proc)
    rel = abs(value - UUX_INTERFERENCE) / abs(UUX_INTERFERENCE)
    check('uux_interference_m', rel < 1e-4,
          'FPTYPE=m %.8e vs %.8e, rel %.1e' % (value, UUX_INTERFERENCE, rel))


# ---------------------------------------------------------------------------
# 2. helicity choice through umami
# ---------------------------------------------------------------------------

def check_helicity_choice(repo, backend, npoints=16, ngrid=4000):
    section('helicity choice of an interference |M|^2 through umami (%s)' % backend)
    proc_dir = standalone(repo, backend, 'uuxttxg_qed2_d', 'u u~ > t t~ g QED^2==2', 'd')
    work = os.path.abspath('uuxttxg_qed2_d')

    def check_sa(args, log):
        status = run(['./check_sa.exe', 'perf'] + [str(a) for a in args], cwd=proc_dir,
                     log=log)
        if status != 0:
            raise RuntimeError('check_sa.exe perf %s failed, see %s'
                               % (' '.join(str(a) for a in args), log))

    def doubles(path):
        values = array.array('d')
        with open(path, 'rb') as f:
            values.frombytes(f.read())
        return values

    # npoints RAMBO points at 1000 GeV, each repeated ngrid times, with a uniform grid of
    # helicity random numbers (k + 0.5) / ngrid for each point: one batch of 64000 events
    points = os.path.join(work, 'points.bin')
    check_sa(['--energy', 1000, '--dump-momenta', points, 1, npoints, 1], 'uuxttxg_points.log')
    with open(points, 'rb') as f:
        raw = f.read()
    size = len(raw) // npoints
    grid = os.path.join(work, 'grid.bin')
    with open(grid, 'wb') as f:
        for ipoint in range(npoints):
            f.write(raw[ipoint * size:(ipoint + 1) * size] * ngrid)
    rnd = os.path.join(work, 'rnd.bin')
    with open(rnd, 'wb') as f:
        array.array('d', [(k + 0.5) / ngrid for k in range(ngrid)] * npoints).tofile(f)
    batch = [npoints * ngrid // 256, 256, 1]
    outputs = {}
    for tag, extra in (('sum', []), ('draw', ['--rnd-hel', rnd]),
                       ('summed', ['--rnd-hel', rnd, '--param', 'interference_helicity_summed=1'])):
        me, hel = os.path.join(work, 'me_%s.bin' % tag), os.path.join(work, 'hel_%s.bin' % tag)
        check_sa(['--momenta', grid, '--dump-me', me, '--dump-hel', hel] + extra + batch,
                 'uuxttxg_%s.log' % tag)
        helicities = array.array('i')
        with open(hel, 'rb') as f:
            helicities.frombytes(f.read())
        outputs[tag] = (doubles(me), helicities)

    mixed = 0
    problems = []
    for ipoint in range(npoints):
        events = slice(ipoint * ngrid, (ipoint + 1) * ngrid)
        helicity_sum = outputs['sum'][0][ipoint * ngrid]
        me, hel = outputs['draw'][0][events], outputs['draw'][1][events]
        abs_sum = abs(me[0])
        signs = {}
        for value, ihel in zip(me, hel):
            if abs(abs(value) - abs_sum) > 1e-12 * abs_sum:
                problems.append('point %d: |M|^2 %.6e != sum|T| %.6e' % (ipoint, value, abs_sum))
                break
            if signs.setdefault(ihel, value > 0) != (value > 0):
                problems.append('point %d: helicity %d with both signs' % (ipoint, ihel))
                break
        mean = sum(me) / ngrid
        if abs(mean - helicity_sum) > len(signs) * abs_sum / ngrid:
            problems.append('point %d: mean %.6e != helicity sum %.6e'
                            % (ipoint, mean, helicity_sum))
        if len(set(signs.values())) == 2:
            mixed += 1
            if not abs_sum > abs(helicity_sum) * (1 + 1e-6):
                problems.append('point %d: mixed signs but sum|T| == |sum T|' % ipoint)
        print('  point %2d  sum T % .6e  mean % .6e  sum|T| %.6e  helicities %2d  signs %s'
              % (ipoint, helicity_sum, mean, abs_sum, len(signs),
                 ''.join('+' if s else '-' for _h, s in sorted(signs.items()))))
    if mixed <= npoints // 2:
        problems.append('only %d of %d points with mixed signs' % (mixed, npoints))
    check('helicity_choice', not problems,
          '; '.join(problems[:3]) or '%d of %d points with mixed signs' % (mixed, npoints))

    # run card [generation] interference_helicity = "summed": the instance keeps the
    # helicity sum as the |M|^2 of every draw (mg7 then writes the helicities as 9)
    worst = max(abs(v - s) / abs(s) for v, s in zip(outputs['summed'][0], outputs['sum'][0]))
    check('helicity_summed', worst < 1e-12,
          'max |M|^2 / helicity sum - 1 = %.1e over %d draws' % (worst, npoints * ngrid))


# ---------------------------------------------------------------------------
# 3. mg7
# ---------------------------------------------------------------------------

def read_lhe(path):
    init, weights = [], []
    with gzip.open(path, 'rt') as f:
        in_init = in_event = False
        for line in f:
            if in_event:
                weights.append(float(line.split()[2]))
                in_event = False
            elif line.startswith('<event'):
                in_event = True
            elif line.startswith('<init>'):
                in_init = True
            elif line.startswith('</init>'):
                in_init = False
            elif in_init:
                init.append(line.split())
    return init, weights


def check_mg7(repo, backend, events):
    section('mg7 p p > u u~ QCD^2==2 (%s)' % backend)
    madgraph(repo, 'mg7_uux', ['generate p p > u u~ QCD^2==2', 'output mg7 PROC_uux'])
    card_path = os.path.join('PROC_uux', 'Cards', 'run_card.toml')
    card = open(card_path).read()
    device = backend if backend in GPU_BACKENDS else 'cpu'
    for pattern, value in (
            (r'(?m)^device = .*', 'device = ["%s"]' % device),
            (r'(?m)^seed = .*', 'seed = 31'),
            (r'(?m)^events = \d+', 'events = %d' % events),
            (r'(?m)^output_format = \S+', 'output_format = "lhe"'),
            (r'(?m)^fixed_ren_scale = false', 'fixed_ren_scale = true'),
            (r'(?m)^fixed_fact_scale = false', 'fixed_fact_scale = true'),
            (r'(?m)^pdf1 = ".*"$', 'pdf1 = "%s"' % UUX_MG7_PDF),
            (r'(?m)^pdf2 = ".*"$', 'pdf2 = "%s"' % UUX_MG7_PDF),
            (r'(?m)^(\[systematics\]\n(?:#.*\n)*)enable = \S+', r'\1enable = false')):
        card, count = re.subn(pattern, value, card, count=1)
        if count != 1:
            raise RuntimeError('cannot set %r in %s' % (value, card_path))
    open(card_path, 'w').write(card)
    if run([sys.executable, 'bin/generate_events', '-f'], cwd='PROC_uux',
           log='mg7_uux_generate_events.log') != 0:
        raise RuntimeError('mg7 generate_events failed, see mg7_uux_generate_events.log')
    info = sorted(glob.glob(os.path.join('PROC_uux', 'Events', '*', 'info.json')))[-1]
    status = json.load(open(info))['process']
    cross, error, abs_cross = status['mean'], status['error'], status['mean_abs']
    summary['mg7_xsec'] = '%.6g +- %.3g pb' % (cross, error)
    sigma = math.hypot(error, UUX_MG7_ERROR)
    allowed = 0.01 * abs(UUX_MG7_CROSS) + 3 * sigma
    check('mg7_cross_section', cross < 0 and abs(cross - UUX_MG7_CROSS) <= allowed,
          '%.6g +- %.3g pb vs reference %.6g +- %.3g pb (allowed %.3g)'
          % (cross, error, UUX_MG7_CROSS, UUX_MG7_ERROR, allowed))
    init, weights = read_lhe(os.path.join(os.path.dirname(info), 'events.lhe.gz'))
    idwtup = int(init[0][8])
    xsecup, _xerrup, xmaxup = (float(v) for v in init[1][:3])
    check('mg7_lhe_init', idwtup == -4 and abs(xsecup - cross) <= 1e-8 * abs(cross)
          and abs(xmaxup - abs_cross) <= 1e-8 * abs_cross,
          'IDWTUP %d, XSECUP %.6g, XMAXUP %.6g (sigma_abs %.6g)'
          % (idwtup, xsecup, xmaxup, abs_cross))
    negative = sum(1 for w in weights if w < 0)
    mean = sum(weights) / len(weights) if weights else 0.
    ratio = cross / abs_cross
    spread = 2 * abs_cross * math.sqrt((1 - ratio ** 2) / 4 / max(1, len(weights)))
    check('mg7_signed_events', len(weights) == events and 0 < negative < len(weights)
          and abs(mean - cross) <= 5 * spread + 3 * error,
          '%d events, %d negative, mean weight %.6g' % (len(weights), negative, mean))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--print-reference-pdf', action='store_true',
                        help='print the LHAPDF set the mg7 check runs with, and exit')
    parser.add_argument('--repo')
    parser.add_argument('--backend')
    parser.add_argument('--events', type=int, default=5000)
    args = parser.parse_args()
    if args.print_reference_pdf:
        print(UUX_MG7_PDF)
        return 0
    if not args.repo or not args.backend:
        parser.error('--repo and --backend are required')
    start = time.time()
    for name, step in (('standalone', lambda: check_standalone(args.repo, args.backend)),
                       ('helicity_choice', lambda: check_helicity_choice(args.repo, args.backend)),
                       ('mg7', lambda: check_mg7(args.repo, args.backend, args.events))):
        try:
            step()
        except Exception as error:  # one broken step must not hide the others
            check(name, False, 'error: %s' % error)
    summary['node'] = os.uname().nodename
    summary['gpu'] = os.environ.get('GPU_NAME', 'none')
    summary['backend'] = args.backend
    summary['walltime'] = '%ds' % (time.time() - start)
    with open('summary.txt', 'w') as f:
        for key, value in summary.items():
            f.write('%s=%s\n' % (key, value))
    print('\n%d check(s) failed: %s' % (len(failures), ', '.join(failures)) if failures
          else '\nall checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
