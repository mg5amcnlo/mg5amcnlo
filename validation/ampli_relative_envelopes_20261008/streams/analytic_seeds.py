"""Independent seeds for frozen stream mixtures, rates and corrected histograms.

Uses the adapter test's signed narrow virtual target and real production sampler.
Folded event histograms use the exact conditional expectation over folds, avoiding
an extra event-fold RNG draw. This changes no sampling or integration decision.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from madgraph.various import ampli_pool

PROTOTYPE = HERE.parent/'experiments/stream_mixture/test_ampli_adapter_prototype.py'
spec = importlib.util.spec_from_file_location('stream_adapter_prototype', PROTOTYPE)
prototype = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prototype)
STUB, DRIVER, SOURCE, ADAPTER_SOURCE = prototype.STUB, prototype.DRIVER, prototype.SOURCE, prototype.ADAPTER_SOURCE


def replace(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def sources():
    stub = replace(STUB, '  double precision :: nv_sum=0d0,v_sum=0d0,last_sign=1d0',
                   '  double precision :: nv_hist(3)=0d0,v_hist(3)=0d0\n'
                   '  double precision :: nv_sum=0d0,v_sum=0d0,last_sign=1d0')
    stub = replace(stub, '  integer :: survey_iteration', '  integer :: survey_iteration,stream_bin')
    stub = replace(stub, '     nv_sum=0d0\n     v_sum=0d0',
                   '     nv_sum=0d0\n     v_sum=0d0\n     nv_hist=0d0\n     v_hist=0d0')
    stub = replace(stub, "  if (abrv.ne.'virt') nv_sum=nv_sum+(2d0+aux_shift)*base",
                   "  stream_bin=min(3,1+int(3d0*x(1)))\n"
                   "  if (abrv.ne.'virt') then\n"
                   '     nv_sum=nv_sum+(2d0+aux_shift)*base\n'
                   '     nv_hist(stream_bin)=nv_hist(stream_bin)+(2d0+aux_shift)*base\n'
                   '  endif')
    for term in ['(virtual_coefficient+aux_shift)*residual_base',
                 '(virtual_coefficient+aux_shift)*residual_base/virtual_fraction(1)']:
        old = '        v_sum=v_sum-' + term + '\n'
        stub = replace(stub, old, old + '        v_hist(stream_bin)=v_hist(stream_bin)-' + term + '\n')
    driver = replace(DRIVER, '  double precision,allocatable :: candidate_weights(:),candidate_factors(:)',
                     '  double precision :: stream_abs_hist(3),stream_signed_hist(3),stream_normalization,stream_sign_sum\n'
                     '  double precision,allocatable :: stream_hist(:,:),stream_sign(:)\n'
                     '  double precision,allocatable :: candidate_weights(:),candidate_factors(:)')
    driver = replace(driver, "  call get_command_argument(1,task)",
                     "  call get_command_argument(1,task)\n"
                     "  call get_command_argument(2,line)\n"
                     "  if (len_trim(line).gt.0) read(line,*) rng_state")
    driver = replace(driver, '  allocate(candidate_weights(1000000),candidate_factors(1000000))',
                     '  allocate(candidate_weights(1000000),candidate_factors(1000000))\n'
                     '  allocate(stream_hist(3,1000000),stream_sign(1000000))')
    driver = replace(driver, '        candidate_weights(candidates)=abs_target',
                     '        stream_hist(:,candidates)=(nv_hist+v_hist)/(nv_sum+v_sum)\n'
                     '        stream_sign(candidates)=sign(1d0,signed_target)\n'
                     '        candidate_weights(candidates)=abs_target')
    driver = replace(driver, '  event_counts=0\n  selected=0',
                     '  event_counts=0\n  selected=0\n  stream_abs_hist=0d0\n'
                     '  stream_signed_hist=0d0\n  stream_normalization=0d0\n  stream_sign_sum=0d0')
    driver = replace(driver, '     event_counts(birth)=event_counts(birth)+1',
                     '     stream_abs_hist=stream_abs_hist+correction*factor*stream_hist(:,i)\n'
                     '     stream_signed_hist=stream_signed_hist+correction*factor*stream_sign(i)*stream_hist(:,i)\n'
                     '     stream_normalization=stream_normalization+correction*factor\n'
                     '     stream_sign_sum=stream_sign_sum+correction*factor*stream_sign(i)\n'
                     '     event_counts(birth)=event_counts(birth)+1')
    driver = replace(driver, "  print *, 'PASS ',trim(task)\ncontains",
                     "  write(*,'(a,8(1x,es25.16))') 'STREAM_OBSERVABLES',stream_normalization, &\n"
                     '       stream_sign_sum,stream_abs_hist,stream_signed_hist\n'
                     "  print *, 'PASS ',trim(task)\ncontains")
    return stub, driver


def base_integral(a, b):
    return 1.5*(.1*(b-a)+(math.exp(-10*a)-math.exp(-10*b))/10)


def exact():
    nonvirtual, virtual = [], []
    for a, b in zip([0., 1/3, 2/3], [1/3, 2/3, 1.]):
        base = base_integral(a, b)
        inner = base_integral(a, min(b, .1)) if a < .1 else 0.
        nonvirtual.append(2*base)
        virtual.append(.02*(.5*base+4.5*inner))
    absolute = sum(nonvirtual)+sum(virtual)
    signed = sum(nonvirtual)-sum(virtual)
    return dict(mean_abs=absolute, mean_signed=signed,
                abs_hist=[(n+v)/absolute for n, v in zip(nonvirtual, virtual)],
                signed_hist=[(n-v)/absolute for n, v in zip(nonvirtual, virtual)],
                signed_fraction=signed/absolute)


def stats(values, expected):
    mean = statistics.mean(values)
    sem = statistics.stdev(values)/math.sqrt(len(values))
    return dict(mean=mean, expected=expected, empirical_sem=sem,
                mean_minus_expected_sem=(mean-expected)/sem if sem else 0.)


def main():
    expected = exact()
    results = {'expected': expected, 'cases': {}, 'source_sha256': {},
               'note': 'Fresh survey and generation for every seed. Histograms are '
               'corrected reserve expectations over folded points, not randomly chosen '
               'physical fold events. Sixteen seeds per folding setting are a diagnostic, '
               'not a precision uncertainty-coverage validation.'}
    stub, driver = sources()
    for name in ['integrator_helpers.f90', 'simple_integrator.f90', 'ampli_mint_adapter.f90']:
        path = ADAPTER_SOURCE if name == 'ampli_mint_adapter.f90' else SOURCE/name
        results['source_sha256'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    results['adapter_source'] = str(ADAPTER_SOURCE.relative_to(ROOT))
    with tempfile.TemporaryDirectory(prefix='mg5-stream-seeds-') as tmp:
        work = Path(tmp)
        (work/'stub.f90').write_text(stub)
        (work/'driver.f90').write_text(driver)
        command = ['gfortran', '-O1', '-g', '-fcheck=all,no-recursion',
                   '-ffpe-trap=invalid,zero,overflow', '-fbacktrace',
                   str(SOURCE/'integrator_helpers.f90'), str(SOURCE/'simple_integrator.f90'),
                   str(work/'stub.f90'), str(ADAPTER_SOURCE),
                   str(work/'driver.f90'), '-o', str(work/'probe')]
        subprocess.run(command, cwd=work, check=True, capture_output=True, text=True)
        for task in ['stream_mixture', 'stream_mixture_adaptive']:
            runs = []
            for index in range(16):
                seed = 8111+index*104729
                directory = work/(task+str(seed))
                directory.mkdir()
                result = subprocess.run([str(work/'probe'), task, str(seed)], cwd=directory,
                                        check=True, capture_output=True, text=True, timeout=45)
                (HERE/f'{task}_{seed}.log').write_text(result.stdout)
                pool = ampli_pool.read_pool(directory)
                _, tails = ampli_pool._native_worker_status(pool)
                assert max(tails.values()) < .01
                row = next(line for line in result.stdout.splitlines() if line.startswith('STREAM_OBSERVABLES'))
                observables = list(map(float, row.split()[1:]))
                normalization = observables[0]
                points = pool['trials']
                runs.append(dict(seed=seed, trials=points, mean_abs=pool['mean_abs'], mean_signed=pool['mean_signed'],
                                 error_abs=math.sqrt(pool['m2_abs'])/points,
                                 error_signed=math.sqrt(pool['m2_signed'])/points,
                                 tails=tails, signed_fraction=observables[1]/normalization,
                                 abs_hist=[value/normalization for value in observables[2:5]],
                                 signed_hist=[value/normalization for value in observables[5:8]]))
            summary = {}
            for field in ['mean_abs', 'mean_signed', 'signed_fraction']:
                summary[field] = stats([run[field] for run in runs], expected[field])
            for field in ['abs_hist', 'signed_hist']:
                summary[field] = [stats([run[field][i] for run in runs], expected[field][i]) for i in range(3)]
            for field in ['abs', 'signed']:
                pulls = [(run['mean_'+field]-expected['mean_'+field])/run['error_'+field] for run in runs]
                summary[field+'_pulls'] = dict(mean=statistics.mean(pulls), std=statistics.stdev(pulls),
                                               max_abs=max(map(abs, pulls)))
            results['cases'][task] = dict(runs=runs, summary=summary)
    (HERE/'analytic_seeds.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({name: value['summary'] for name, value in results['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
