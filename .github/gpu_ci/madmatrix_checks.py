#!/usr/bin/env python3
"""The GPU backend of madmatrix against its scalar CPU backend, with check_sa.exe.

Run by madmatrix_gpu.sh (job madmatrix of gpu_runner_ci.yml) in its run directory.
Every evaluation is a ``check_sa.exe perf`` run of the standalone output, one executable
per backend (check_sa_<backend>.exe), on the same events: the momenta, alpha_s, flavour
and random numbers of every event go in as raw binary files (--momenta, --alphas,
--flavors, --rnd-hel, --rnd-col) and the |M|^2, helicities and colours come out the
same way (--dump-me, --dump-hel, --dump-col). An abort of the GPU runtime fails the run
of check_sa.exe, so only the check it belongs to.

1. mssm_gpu_vs_cpu: MSSM ``p p > go go`` (both subprocesses), whose squark-quark-gluino
   couplings are running flavour couplings (nDPF > 0, gathered per event into
   dpf_value), on RAMBO points with a random alpha_s and flavour per event: the GPU
   |M|^2 equal the CPU ones event by event (the GPU gather used to address its one-event
   dpf_value with the offset of the event in the all-events coupling buffer, out of
   bounds past the first event page), and scale as alpha_s^2 (--alphas 0.10 and 0.12).
2. zero_me: ``b b~ > ta+ ta- / z a`` (s-channel Higgs only): |M|^2 > 0 with the default
   param card, then the tau Yukawa coupling set to 0, so that no helicity survives the
   helicity filtering: both backends return |M|^2 = 0 and no helicity (-1), and for the
   colour either none (-1) or the only colour flow (0, what a -ffast-math build makes of
   0/0), never stale memory; the GPU used to abort on the invalid launch configuration
   (gridDim.y = 0) of the helicity loop.
3. nonblocking_stream: ``g g > t t~ g``, 4 calls of 16384 events, with umami handed a
   non-blocking stream (--nonblocking-stream, as a torch side stream, which does not
   synchronise with the default stream) gives the same |M|^2 and helicities as on the
   default stream: every memset, kernel and copy of the matrix element must go to the
   stream umami is given, the helicity filtering of the first call included.
4. helicity_index: ``g g > t t~ z`` (massless and massive vectors, a longitudinal Z,
   massive fermions) evaluated for one given helicity per event (UMAMI_IN_HELICITY_INDEX,
   --hel-index), on both backends: the |M|^2 of the ncomb helicities, each given to every
   event, add up to the helicity-summed |M|^2, the GPU ones equal the CPU ones helicity by
   helicity, and with a random helicity per event (the threads of a warp, or the lanes of
   a SIMD vector, on different helicities in lock-step) every event gets the |M|^2 of its
   helicity and returns it as the chosen helicity.

Any CPU backend but scalar (simd_128, ...) works too, to try the script without a GPU;
check 3 is then skipped. Writes summary.txt (key=value lines) in the current directory;
exit status 1 if a check fails.
"""

import argparse
import array
import glob
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time

GPU_BACKENDS = ('cuda', 'hip')
CPU_BACKEND = 'scalar'
CHECKS = ('mssm_gpu_vs_cpu', 'zero_me', 'nonblocking_stream', 'helicity_index')

results = {}


def section(title):
    print('\n=================== %s (%s)' % (title, time.strftime('%H:%M:%S')), flush=True)


def check(name, ok, detail):
    print('%s %s: %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)
    results[name] = (ok, detail)


def run(cmd, cwd=None, log=None):
    with open(log or os.devnull, 'w') as out:
        return subprocess.call(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT)


def write(path, typecode, values):
    """A raw binary file of check_sa.exe: 'd' double, 'I'/'i' 32-bit unsigned/signed int."""
    with open(path, 'wb') as f:
        array.array(typecode, values).tofile(f)
    return os.path.abspath(path)


def read(path, typecode):
    """A file written by check_sa.exe: 'd' (--dump-me), 'i' (--dump-hel, --dump-col)."""
    values = array.array(typecode)
    with open(path, 'rb') as f:
        values.frombytes(f.read())
    return values


def output(repo, name, model, process, backends):
    """output standalone + make for these backends in every subprocess, keeping the
    check_sa.exe of each as check_sa_<backend>.exe: the subprocess directories."""
    with open(name + '.mg5', 'w') as f:
        f.write('import model %s\ngenerate %s\noutput standalone %s -f\n' % (model, process, name))
    if run([sys.executable, os.path.join(repo, 'bin', 'madgraph'), name + '.mg5'],
           log=name + '.log') != 0:
        raise RuntimeError('madgraph failed, see %s.log' % name)
    proc_dirs = sorted(glob.glob(os.path.join(name, 'SubProcesses', 'P*')))
    if not proc_dirs:
        raise RuntimeError('no subprocess directory for %s' % process)
    jobs = os.environ.get('SLURM_CPUS_PER_TASK', '4')
    for proc_dir in proc_dirs:
        for backend in backends:
            log = '%s_%s_%s_make.log' % (name, os.path.basename(proc_dir), backend)
            # check_sa.exe is relinked for the backend of each make (its libraries carry
            # the backend in their name)
            if run(['make', '-j', jobs, 'BACKEND=' + backend, 'FPTYPE=d'],
                   cwd=proc_dir, log=log) != 0:
                raise RuntimeError('make failed, see %s' % log)
            shutil.copy(os.path.join(proc_dir, 'check_sa.exe'),
                        os.path.join(proc_dir, 'check_sa_%s.exe' % backend))
    return proc_dirs


def check_sa(proc_dir, backend, args, log):
    """check_sa_<backend>.exe perf <args> in proc_dir (it reads ../../Cards/param_card.dat)."""
    cmd = ['./check_sa_%s.exe' % backend, 'perf'] + [str(a) for a in args]
    status = run(cmd, cwd=proc_dir, log=log)
    if status != 0:
        raise RuntimeError('%s failed (exit status %d), see %s' % (' '.join(cmd), status, log))


def dump(prefix, what):
    return os.path.abspath('%s_%s.bin' % (prefix, what))


# ---------------------------------------------------------------------------
# 1. running flavour couplings (MSSM p p > go go): GPU against CPU
# ---------------------------------------------------------------------------

def check_mssm_gpu_vs_cpu(backend, repo, nevt=1024):
    section('MSSM p p > go go, running flavour couplings: %s against %s' % (backend, CPU_BACKEND))
    name = 'mssm_gogo'
    proc_dirs = output(repo, name, 'MSSM_SLHA2', 'p p > go go', (CPU_BACKEND, backend))
    rng = random.Random(11)
    problems, details = [], []
    for proc_dir in proc_dirs:
        sub = os.path.basename(proc_dir)
        data = open(os.path.join(proc_dir, 'ProcessData.h')).read()
        ndpf = int(re.search(r'constexpr int nDPF = (\d+);', data).group(1))
        nflavor = int(re.search(r'constexpr int nmaxflavor = (\d+);', data).group(1))
        if ndpf == 0:
            problems.append('%s: nDPF = 0, no running flavour coupling to check' % sub)
        prefix = '%s_%s' % (name, sub)
        # one batch of nevt events, the same on both backends
        batch = [1, nevt, 1]
        momenta = dump(prefix, 'momenta')
        check_sa(proc_dir, CPU_BACKEND, ['--energy', 3000, '--dump-momenta', momenta] + batch,
                 prefix + '_rambo.log')
        alphas = write(prefix + '_alphas.bin', 'd', [rng.uniform(0.09, 0.14) for _ in range(nevt)])
        flavors = write(prefix + '_flavors.bin', 'I', [rng.randrange(nflavor) for _ in range(nevt)])
        me = {}
        for tag, run_backend, alphas_arg in (('cpu', CPU_BACKEND, alphas), ('gpu', backend, alphas),
                                             ('gpu010', backend, '0.10'), ('gpu012', backend, '0.12')):
            out = dump(prefix, 'me_' + tag)
            check_sa(proc_dir, run_backend, ['--momenta', momenta, '--alphas', alphas_arg,
                                             '--flavors', flavors, '--dump-me', out] + batch,
                     '%s_%s.log' % (prefix, tag))
            me[tag] = read(out, 'd')
        if not len(me['cpu']) == len(me['gpu']) == nevt:
            problems.append('%s: %d CPU and %d GPU |M|^2 for %d events'
                            % (sub, len(me['cpu']), len(me['gpu']), nevt))
        rels = [abs(a - b) / abs(a) if a != 0 else (0. if b == 0 else float('inf'))
                for a, b in zip(me['cpu'], me['gpu'])]
        # every event that is not within the tolerance, a nan included (it fails any
        # comparison, so it must not go through a running maximum, which would drop it)
        bad = [ievt for ievt, rel in enumerate(rels) if not rel < 1e-9]
        worst = max((rel for rel in rels if not math.isnan(rel)), default=0.)
        if bad:
            first = bad[0]
            problems.append('%s: %d of %d events differ, first event %d %s %.10e vs %s %.10e (rel %.1e)'
                            % (sub, len(bad), nevt, first, backend, me['gpu'][first],
                               CPU_BACKEND, me['cpu'][first], rels[first]))
        if not any(v != 0 for v in me['cpu']):
            problems.append('%s: all |M|^2 vanish' % sub)
        # |M|^2 ~ alpha_s^2, event by event
        ratio = (0.12 / 0.10) ** 2
        scaling = [b / a / ratio - 1 if a != 0 else float('inf')
                   for a, b in zip(me['gpu010'], me['gpu012'])]
        bad = [ievt for ievt, s in enumerate(scaling) if not abs(s) < 1e-9]
        if bad or len(scaling) != nevt:
            problems.append('%s: |M|^2 not ~ alpha_s^2 in %d of %d events (first: ratio - 1 = %.1e)'
                            % (sub, len(bad), nevt, scaling[bad[0]] if bad else float('nan')))
        details.append('%s nDPF %d, %d flavours, max rel diff %.1e' % (sub, ndpf, nflavor, worst))
        print('  ' + details[-1])
    check('mssm_gpu_vs_cpu', not problems, '; '.join(problems[:3]) or '; '.join(details))


# ---------------------------------------------------------------------------
# 2. a subprocess with no good helicity
# ---------------------------------------------------------------------------

def check_zero_me(backend, repo, nevt=64):
    section('no good helicity: b b~ > ta+ ta- / z a with ymtau = 0 (%s, %s)' % (backend, CPU_BACKEND))
    name = 'zero_me'
    (proc_dir,) = output(repo, name, 'sm', 'b b~ > ta+ ta- / z a', (CPU_BACKEND, backend))
    card = os.path.join(name, 'Cards', 'param_card.dat')
    text = open(card).read()
    zero_card, count = re.subn(r'(?mi)^(\s*15\s+)\S+(\s*#\s*ymtau)', r'\g<1>0.000000e+00\2', text)
    if count != 1:
        raise RuntimeError('no ymtau in %s' % card)
    rng = random.Random(5)
    batch = [1, nevt, 1]
    momenta = dump(name, 'momenta')
    check_sa(proc_dir, CPU_BACKEND, ['--energy', 500, '--dump-momenta', momenta,
                                     '--dump-me', dump(name, 'me_default')] + batch,
             name + '_default.log')
    me_default = read(dump(name, 'me_default'), 'd')
    rnd = write(name + '_rnd.bin', 'd', [rng.random() for _ in range(nevt)])
    # check_sa.exe reads the param card at every run: from here on ymtau = 0 (the
    # helicity filtering runs at the first call of each run)
    open(card, 'w').write(zero_card)
    problems, seen = [], []
    for run_backend in (backend, CPU_BACKEND):
        prefix = '%s_%s' % (name, run_backend)
        check_sa(proc_dir, run_backend, ['--momenta', momenta, '--rnd-hel', rnd, '--rnd-col', rnd,
                                         '--dump-me', dump(prefix, 'me'),
                                         '--dump-hel', dump(prefix, 'hel'),
                                         '--dump-col', dump(prefix, 'col')] + batch,
                 prefix + '.log')
        me, hel, col = (read(dump(prefix, 'me'), 'd'), read(dump(prefix, 'hel'), 'i'),
                        read(dump(prefix, 'col'), 'i'))
        if len(me) != nevt or any(v != 0 for v in me):
            problems.append('%s: |M|^2 not 0: %s' % (run_backend, list(me[:4])))
        if set(hel) != {-1} or not set(col) <= {-1, 0}:
            problems.append('%s: helicities %s, colours %s (expected -1, and -1 or 0)'
                            % (run_backend, sorted(set(hel))[:4], sorted(set(col))[:4]))
        seen.append('%s colour %s' % (run_backend, sorted(set(col))))
    if len(me_default) != nevt or not all(v > 0 for v in me_default):
        problems.append('|M|^2 with the default card not > 0: %s' % list(me_default[:4]))
    check('zero_me', not problems, '; '.join(problems) or
          '%d events: |M|^2 = 0, helicity -1, %s (default card: %.3e)'
          % (nevt, ', '.join(seen), me_default[0]))


# ---------------------------------------------------------------------------
# 3. a non-blocking stream
# ---------------------------------------------------------------------------

def check_nonblocking_stream(backend, repo, ncalls=4):
    section('g g > t t~ g on a non-blocking stream (%s)' % backend)
    if backend not in GPU_BACKENDS:
        check('nonblocking_stream', True, 'skipped: %s is not a GPU backend' % backend)
        return
    name = 'ggttxg'
    (proc_dir,) = output(repo, name, 'sm', 'g g > t t~ g', (backend,))
    # ncalls calls (iterations) of 64 x 256 = 16384 events, the RAMBO points of check_sa.exe
    # (the same in both runs: one seed per iteration)
    batch = [64, 256, ncalls]
    nevt = 64 * 256 * ncalls
    rng = random.Random(3)
    rnd = write(name + '_rnd.bin', 'd', [rng.random() for _ in range(nevt)])
    me, hel = {}, {}
    for tag, extra in (('default', []), ('nonblocking', ['--nonblocking-stream'])):
        prefix = '%s_%s' % (name, tag)
        check_sa(proc_dir, backend, extra + ['--energy', 1000, '--rnd-hel', rnd,
                                             '--dump-me', dump(prefix, 'me'),
                                             '--dump-hel', dump(prefix, 'hel')] + batch,
                 prefix + '.log')
        me[tag], hel[tag] = read(dump(prefix, 'me'), 'd'), read(dump(prefix, 'hel'), 'i')
    problems = []
    if not len(me['default']) == len(me['nonblocking']) == nevt:
        problems.append('%d and %d |M|^2 for %d events' % (len(me['default']), len(me['nonblocking']), nevt))
    for icall in range(ncalls):
        events = slice(icall * nevt // ncalls, (icall + 1) * nevt // ncalls)
        pairs = list(zip(me['nonblocking'][events], me['default'][events]))
        bad_me = sum(1 for a, b in pairs if not abs(a - b) <= 1e-12 * abs(b))
        bad_hel = sum(1 for a, b in zip(hel['nonblocking'][events], hel['default'][events]) if a != b)
        if bad_me or bad_hel:
            problems.append('call %d: %d |M|^2 and %d helicities differ' % (icall, bad_me, bad_hel))
    if not all(v > 0 for v in me['default']) or min(hel['default'], default=-1) < 0:
        problems.append('default stream: |M|^2 <= 0 or no helicity')
    check('nonblocking_stream', not problems, '; '.join(problems[:3]) or
          '%d calls of %d events identical to the default stream' % (ncalls, nevt // ncalls))


# ---------------------------------------------------------------------------
# 4. one given helicity per event
# ---------------------------------------------------------------------------

def check_helicity_index(backend, repo, nevt=512):
    section('g g > t t~ z, one given helicity per event (%s, %s)' % (backend, CPU_BACKEND))
    name = 'ggttxz_hel'
    (proc_dir,) = output(repo, name, 'sm', 'g g > t t~ z', (CPU_BACKEND, backend))
    data = open(os.path.join(proc_dir, 'ProcessData.h')).read()
    ncomb = int(re.search(r'constexpr int ncomb = (\d+);', data).group(1))
    batch = [1, nevt, 1]
    momenta = dump(name, 'momenta')
    check_sa(proc_dir, CPU_BACKEND, ['--energy', 2000, '--dump-momenta', momenta] + batch,
             name + '_rambo.log')
    rng = random.Random(17)
    mixed = [rng.randrange(ncomb) for _ in range(nevt)]
    problems, details = [], []
    perhel = {}
    for run_backend in (CPU_BACKEND, backend):
        prefix = '%s_%s' % (name, run_backend)
        check_sa(proc_dir, run_backend, ['--momenta', momenta, '--dump-me', dump(prefix, 'me')] + batch,
                 prefix + '.log')
        summed = read(dump(prefix, 'me'), 'd')
        perhel[run_backend] = []
        for ihel in range(ncomb):
            hprefix = '%s_h%d' % (prefix, ihel)
            hels = write(hprefix + '_in.bin', 'i', [ihel] * nevt)
            check_sa(proc_dir, run_backend, ['--momenta', momenta, '--hel-index', hels,
                                             '--dump-me', dump(hprefix, 'me'),
                                             '--dump-hel', dump(hprefix, 'hel')] + batch,
                     hprefix + '.log')
            perhel[run_backend].append(read(dump(hprefix, 'me'), 'd'))
            if set(read(dump(hprefix, 'hel'), 'i')) != {ihel}:
                problems.append('%s: helicity %d not returned as the chosen helicity' % (run_backend, ihel))
        total = [sum(perhel[run_backend][ihel][ievt] for ihel in range(ncomb)) for ievt in range(nevt)]
        rels = [abs(a - b) / abs(b) if b != 0 else float('inf') for a, b in zip(total, summed)]
        bad = [ievt for ievt, rel in enumerate(rels) if not rel < 1e-10]
        if bad or len(summed) != nevt:
            problems.append('%s: the sum over the helicities differs from the helicity sum in %d of %d events'
                            % (run_backend, len(bad), nevt))
        # a random helicity per event, the same events
        mprefix = '%s_mixed' % prefix
        hels = write(mprefix + '_in.bin', 'i', mixed)
        check_sa(proc_dir, run_backend, ['--momenta', momenta, '--hel-index', hels,
                                         '--dump-me', dump(mprefix, 'me'),
                                         '--dump-hel', dump(mprefix, 'hel')] + batch,
                 mprefix + '.log')
        me, hel = read(dump(mprefix, 'me'), 'd'), read(dump(mprefix, 'hel'), 'i')
        expected = [perhel[run_backend][ihel][ievt] for ievt, ihel in enumerate(mixed)]
        bad = [ievt for ievt, (a, b) in enumerate(zip(me, expected)) if not abs(a - b) <= 1e-12 * abs(b)]
        if bad or list(hel) != mixed:
            problems.append('%s: a random helicity per event: %d |M|^2 differ from that helicity\'s, '
                            '%d helicities not returned' % (run_backend, len(bad),
                                                            sum(1 for a, b in zip(hel, mixed) if a != b)))
        details.append('%s: max rel diff of the helicity sum %.1e' % (run_backend, max(rels, default=0.)))
        print('  ' + details[-1])
    bad = sum(1 for ihel in range(ncomb) for a, b in zip(perhel[backend][ihel], perhel[CPU_BACKEND][ihel])
              if not abs(a - b) <= 1e-9 * max(abs(b), 1e-300))
    if bad:
        problems.append('%d (helicity, event) |M|^2 differ between %s and %s' % (bad, backend, CPU_BACKEND))
    check('helicity_index', not problems, '; '.join(problems[:3]) or
          '%d helicities x %d events; %s' % (ncomb, nevt, '; '.join(details)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--repo', required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--only', choices=CHECKS, action='append',
                        help='run this check only (repeatable)')
    args = parser.parse_args()
    if args.backend == CPU_BACKEND:
        parser.error('the checks compare a backend with %s: give another one' % CPU_BACKEND)
    start = time.time()
    for name in args.only or CHECKS:
        try:
            globals()['check_' + name](args.backend, args.repo)
        except Exception as error:
            check(name, False, 'error: %s' % error)
    summary = {key: ('ok' if ok else 'FAILED') + ' (%s)' % detail
               for key, (ok, detail) in results.items()}
    failures = [key for key, (ok, _detail) in results.items() if not ok]
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
