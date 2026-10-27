#!/usr/bin/env python3
"""Prepare or run the six-process common-source MAcNLOPS campaign.

`prepare` writes cards/configuration plans only. `run` exports fresh processes,
then generates and matches ordinary and Delta sources, stopping at LHE level.
"""
import argparse
import collections
import datetime
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / 'production_1M_sampled'
FKS_SETTINGS = dict(MCExplicitKLSum=False, MCSubtractionAtFixedFlow=True,
                    FKSExplicitSum=False)
PROCESSES = [
    dict(key='ZZ', analysis='zz', directory='ZZ_5FS', model='loop_sm-no_b_mass',
         process='p p > z z [QCD]', flavours=5, seed=2630001, born_partons=0),
    dict(key='DY', analysis='dy', directory='DY_5FS', model='loop_sm-no_b_mass',
         process='p p > e+ e- [QCD]', flavours=5, seed=2630101, born_partons=0),
    dict(key='ttbar', analysis='ttbar', directory='TTbar_5FS', model='loop_sm-no_b_mass',
         process='p p > t t~ [QCD]', flavours=5, seed=2630201, born_partons=2),
    dict(key='Wbb', analysis='wbb', directory='Wbb_4FS', model='loop_sm',
         process='p p > w+ b b~ QCD=2 QED=1 [QCD]', flavours=4,
         seed=2630301, born_partons=2),
    dict(key='single_top', analysis='single_top', directory='SingleTop_5FS',
         model='loop_sm-no_b_mass', process='p p > t j $$ w+ w- [QCD]',
         flavours=5, seed=2630401, born_partons=2, generation_cuts='none'),
    dict(key='Wj', analysis='wjet', directory='WPlusJet_5FS', model='loop_sm-no_b_mass',
         process='p p > w+ j [QCD]', flavours=5, seed=2630501, born_partons=1,
         generation_cuts='wjet',
         analysis_jets=dict(algorithm='antikt', radius=0.5, pt_min_GeV=25.,
                            abs_rapidity_max=4.5, minimum_jets=1)),
]
PHYSICS_FILES = [
    'madgraph/various/macnlops.py', 'madgraph/interface/amcatnlo_run_interface.py',
    'madgraph/iolibs/export_fks.py', 'Template/NLO/MCatNLO/srcPythia8/MacNLOPS.cc',
    'Template/NLO/SubProcesses/driver_macnlops.f',
    'Template/NLO/SubProcesses/handling_lhe_events.f',
    'Template/NLO/SubProcesses/montecarlocounter.f',
    'Template/NLO/SubProcesses/fks_singular.f', 'Template/NLO/Cards/FKS_params.dat',
]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scan_lhe(path):
    counts = collections.Counter()
    sumw = sumabs = comp = 0.
    header = False
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            line = line.strip()
            if line == b'<event>' or line.startswith(b'<event '):
                header = True
                counts['events'] += 1
            elif header:
                fields = line.split()
                if len(fields) < 6:
                    raise RuntimeError('Malformed LHE event header')
                weight = float(fields[2].replace(b'D', b'E').replace(b'd', b'e'))
                if not math.isfinite(weight):
                    raise RuntimeError('Nonfinite source weight')
                change = weight - comp
                total = sumw + change
                comp = (total - sumw) - change
                sumw = total
                sumabs += abs(weight)
                counts['negative_weights'] += weight < 0
                header = False
            elif line.startswith(b'#aMCatNLO'):
                kind = {1: 'S', 2: 'H'}.get(int(line.split()[1]))
                if kind is None:
                    raise RuntimeError('Unknown MC@NLO event type')
                counts[kind] += 1
                if weight < 0:
                    counts['negative_' + kind] += 1
            elif line.startswith(b'<macnlops_hard_event '):
                partition = re.search(br'partition="([^"]+)"', line)
                counts['partition_' + (partition[1].decode() if partition else 'missing')] += 1
            elif line.startswith(b'<macnlops_component '):
                counts['sampled_components'] += 1
    if counts['events'] != counts['H'] + counts['S']:
        raise RuntimeError('MC@NLO annotation count mismatch')
    return dict(counts=dict(counts), sumw=sumw, sumabs=sumabs,
                sha256=sha(path), path=str(path))


def cpu_slots(available=None, parallel=2, cores=6):
    """Disjoint affinity slots; never exceed two jobs or six cores per job."""
    if not 1 <= parallel <= 2 or not 1 <= cores <= 6:
        raise ValueError('Use at most two processes, each with at most six cores')
    available = sorted(set(os.sched_getaffinity(0) if available is None else available))
    if not available:
        raise ValueError('No available CPUs')
    count = min(parallel, len(available))
    width = min(cores, len(available) // count)
    return [available[i * width:(i + 1) * width] for i in range(count)]


def fingerprint():
    return {name: sha(ROOT / name) for name in PHYSICS_FILES}


def new_config(base, events=1000000, parallel=2, cores=6):
    if events < 1:
        raise ValueError('events must be positive')
    slots = cpu_slots(parallel=parallel, cores=cores)
    return dict(schema='macnlops-sampled-common-source-campaign-v2', root=str(ROOT),
                campaign_dir=str(Path(base).resolve()), events_per_source=events,
                source_modes=['ordinary', 'delta'], beam_energy_GeV=6500.,
                lhaids={'5': 21150, '4': 23300}, req_acc=.001, nevt_job=10000,
                pythia8='/export/tmp/rikkert/Pythia/pythia8318',
                lhapdf='/export/tmp/rikkert/lhapdf/bin/lhapdf-config',
                parallel_processes=len(slots), cores_per_process=cores,
                cpu_slots=slots, maximum_cores=sum(map(len, slots)),
                fks_settings=FKS_SETTINGS, partition='sampled',
                physics_sha256=fingerprint(), processes=PROCESSES,
                common_source=True,
                stop='LHE after first emission and H removal / S veto; no full shower')


def validate_config(config, base, check_physics=True):
    if (config.get('schema') != 'macnlops-sampled-common-source-campaign-v2'
            or config.get('fks_settings') != FKS_SETTINGS
            or config.get('partition') != 'sampled'):
        raise RuntimeError('Requires a fresh sampled-component campaign; old global-sum checkpoints cannot be reused')
    if Path(config['campaign_dir']) != Path(base).resolve() or Path(config['root']) != ROOT:
        raise RuntimeError('Campaign paths differ from the recorded configuration')
    if check_physics and config['physics_sha256'] != fingerprint():
        raise RuntimeError('Physics source changed; prepare a fresh campaign directory')
    slots = config['cpu_slots']
    flat = [cpu for slot in slots for cpu in slot]
    if (not 1 <= len(slots) <= 2 or any(not 1 <= len(slot) <= 6 for slot in slots)
            or len(flat) != len(set(flat)) or not set(flat) <= os.sched_getaffinity(0)):
        raise RuntimeError('CPU slots must be available, disjoint, and limited to 2 x 6')


def prepare(base, events=1000000, parallel=2, cores=6):
    base = Path(base).resolve()
    path = base / 'config.json'
    config = new_config(base, events, parallel, cores)
    if path.exists():
        existing = json.loads(path.read_text())
        validate_config(existing, base)
        if existing != config:
            raise RuntimeError('Refusing to change an existing campaign configuration')
    else:
        dump(path, config)
        (base / 'revision.txt').write_bytes(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT))
    for folder in ['commands', 'logs', 'cards', 'tmp', 'processes']:
        (base / folder).mkdir(exist_ok=True)
    return config


def configure(process, p, delta, config):
    sys.path.insert(0, str(ROOT))
    from madgraph.various import banner, shower_card
    path = process / 'Cards/run_card.dat'
    card = banner.RunCardNLO(str(path))
    values = dict(nevents=config['events_per_source'], req_acc=config['req_acc'],
        nevt_job=config['nevt_job'], iseed=p['seed'] + int(delta),
        parton_shower='PYTHIA8', event_norm='sum',
        ebeam1=config['beam_energy_GeV'], ebeam2=config['beam_energy_GeV'],
        pdlabel='lhapdf', lhaid=[config['lhaids'][str(p['flavours'])]],
        macnlops=True, macnlops_resolved_source=False, mcatnlo_delta=delta,
        store_hard_momenta=True, macnlops_flavour_count=p['flavours'],
        maxjetflavor=p['flavours'], lhe_version=3, dynamical_scale_choice=[3],
        fixed_ren_scale=False, fixed_fac_scale=False, reweight_scale=[False],
        reweight_pdf=[False], folding=[1, 1, 1], ickkw=0,
        ptj=0., etaj=-1., ptl=0., etal=-1., mll=0.,
        mll_sf=60. if p['key'] == 'DY' else 0.)
    if p.get('generation_cuts') in ('none', 'wjet'):
        card.remove_all_cut()
        values.update({key: card[key] for key in card.cuts_parameter})
        values.update(etagamma=-1., custom_fcts=[])
    if p.get('generation_cuts') == 'wjet':
        values.update(ptj=10., etaj=-1., jetalgo=-1., jetradius=.5)
    for key, value in values.items():
        card.set(key, value, user=True)
    card.write(str(path))
    parameters = process / 'SubProcesses/FKS_params.dat'
    text = parameters.read_text()
    for key, value in FKS_SETTINGS.items():
        text, n = re.subn(r'(?m)(^#' + re.escape(key) + r'\s*\n)[^\n]+',
                         lambda m: m[1] + ('.true.' if value else '.false.'), text)
        if n != 1:
            raise RuntimeError('Missing or ambiguous FKS parameter ' + key)
    parameters.write_text(text)
    shower = process / 'Cards/shower_card.dat'
    sc = shower_card.ShowerCard(str(shower))
    for key, value in dict(hadronize=False, ue_enabled=False, primordialkt=False,
                          qed_shower=False, nevents=-1, nsplit_jobs=1, pdfcode=1).items():
        sc[key] = value
    sc.write(str(shower))
    return values


def contract_id(config, p):
    return hashlib.sha256(json.dumps(dict(config=config, process=p), sort_keys=True).encode()).hexdigest()


def check_sample(sample, events):
    counts = sample['counts']
    if (counts.get('events') != events or counts.get('partition_sampled', 0) != events
            or counts.get('sampled_components', 0) != events):
        raise RuntimeError('Wrong event population or missing sampled kl/ij/flow packet')


def worker(base, config, key, cpus):
    p = next(p for p in config['processes'] if p['key'] == key)
    process = base / 'processes' / p['directory']
    token = contract_id(config, p)
    state = dict(process=key, directory=str(process), cpus=cpus, pid=os.getpid(),
                 started_utc=now(), status='running', samples={})

    def update(**values):
        state.update(values, updated_utc=now())
        dump(base / ('status_' + key + '.json'), state)

    def command(stage, text, export=False):
        cmdfile = base / 'commands' / (key + '_' + stage + '.cmd')
        log = base / 'logs' / (key + '_' + stage + '.log')
        suffix = ' --no_save' if export else ''
        cmdfile.write_text('set nb_core %d%s\nset run_mode 2%s\n%s\n' %
                          (len(cpus), suffix, suffix, text))
        update(stage=stage, log=str(log), command_file=str(cmdfile))
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MAKEFLAGS='-j1',
                   TMPDIR=str(base / 'tmp'))
        executable = ROOT / 'bin/mg5_aMC' if export else process / 'bin/aMCatNLO'
        with log.open('w') as output:
            result = subprocess.run([sys.executable, str(executable), str(cmdfile)],
                cwd=ROOT if export else process, env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('%s returned %d; see %s' % (stage, result.returncode, log))

    try:
        marker = process / 'campaign_export.json'
        if process.exists():
            if not marker.exists() or json.loads(marker.read_text()).get('contract') != token:
                raise RuntimeError('Refusing an unqualified or partial process export: ' + str(process))
        else:
            incoming = 'g u d s c b u~ d~ s~ c~ b~' if p['flavours'] == 5 else 'g u d s c u~ d~ s~ c~'
            command('export', 'set automatic_html_opening False --no_save\nimport model %s\n'
                    'define p = %s\ndefine j = %s\ngenerate %s\noutput %s -nojpeg\nquit' %
                    (p['model'], incoming, incoming, p['process'], process), export=True)
            if not (process / 'bin/aMCatNLO').is_file():
                raise RuntimeError('Process export did not complete')
            settings = process / 'Cards/amcatnlo_configuration.txt'
            text = settings.read_text()
            for name, value in [('pythia8_path', config['pythia8']), ('lhapdf', config['lhapdf'])]:
                text = re.sub(r'(?m)^\s*#?\s*' + name + r'\s*=.*$', name + ' = ' + value, text)
            settings.write_text(text)
            dump(marker, dict(contract=token, complete_utc=now()))
        for mode in config['source_modes']:
            delta = mode == 'delta'
            run = mode + '_1M' if config['events_per_source'] == 1000000 else mode + '_source'
            source = process / 'Events' / run / 'events.lhe.gz'
            accepted = source.parent / 'macnlops/events.lhe.gz'
            checkpoint = base / (key + '_' + mode + '_complete.json')
            policy = 'component-undamped-delta-v1' if delta else 'component-source-v1'
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved.get('contract') != token or saved.get('veto_prescription') != policy:
                    raise RuntimeError('Checkpoint belongs to a different matching prescription')
                for sample in ['source', 'macnlops']:
                    path = Path(saved[sample]['path'])
                    if not path.is_file() or sha(path) != saved[sample]['sha256']:
                        raise RuntimeError('Completed sample is missing or changed: ' + str(path))
                state['samples'][mode] = saved
                continue
            if accepted.parent.exists():
                raise RuntimeError('Partial matching output requires inspection: ' + str(accepted.parent))
            values = configure(process, p, delta, config)
            snapshot = base / 'cards' / key / mode
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in ['run_card.dat', 'param_card.dat', 'shower_card.dat', 'amcatnlo_configuration.txt']:
                shutil.copy2(process / 'Cards' / name, snapshot / name)
            shutil.copy2(process / 'SubProcesses/FKS_params.dat', snapshot / 'FKS_params.dat')
            if not source.exists():
                command(mode + '_source', 'launch aMC@NLO --parton -f -n ' + run)
            update(stage=mode + '_validate_source')
            original = scan_lhe(source)
            check_sample(original, config['events_per_source'])
            dump(base / (key + '_' + mode + '_source.json'), original)
            for name in ['macnlops_profile.json', 'macnlops_shower.cmnd', 'macnlops_probe.json']:
                item = process / 'Cards' / name
                if item.exists():
                    shutil.copy2(item, snapshot / name)
            command(mode + '_matching', 'shower ' + run + ' -f')
            update(stage=mode + '_validate_matching')
            collection = json.loads(accepted.with_name('collection.json').read_text())
            count = collection['counts']
            if collection.get('veto_prescription') != policy:
                raise RuntimeError('Matching used the wrong veto prescription')
            if count.get('input_H', 0) + count.get('input_S', 0) != config['events_per_source']:
                raise RuntimeError('Matching did not consume the whole common source')
            derived = scan_lhe(accepted)
            check_sample(derived, count.get('accepted_H', 0) + count.get('accepted_S', 0))
            if abs(derived['sumw'] - collection['sumw']) > 1.e-10 * max(1., derived['sumabs']):
                raise RuntimeError('Accepted weight sum differs from the collection')
            if sha(source) != original['sha256']:
                raise RuntimeError('The common source changed during matching')
            result = dict(contract=token, source=original, macnlops=derived,
                          matching_counts=count, settings=values, veto_prescription=policy,
                          fks_settings=FKS_SETTINGS, complete_utc=now(),
                          normalization='sum; retained events keep their common-source weights')
            dump(checkpoint, result)
            state['samples'][mode] = result
            update(stage=mode + '_complete')
        update(status='complete', stage='complete', finished_utc=now())
    except Exception as exc:
        update(status='failed', error=str(exc), failed_utc=now())
        traceback.print_exc()
        return 1
    return 0


def run_queue(items, slots, start, finished, poll_seconds=2):
    """Bound the entire export -> ordinary -> Delta pipeline to one CPU slot."""
    pending = collections.deque(items)
    active = []
    failed = []
    free = collections.deque(slots)
    try:
        while pending or active:
            while pending and free:
                item, cpus = pending.popleft(), free.popleft()
                active.append((item, cpus, start(item, cpus)))
            for item, cpus, child in active[:]:
                code = child.poll()
                if code is None:
                    continue
                finished(item, cpus, child)
                if code:
                    failed.append(item)
                active.remove((item, cpus, child))
                free.append(cpus)
            if active:
                time.sleep(poll_seconds)
    finally:
        for _, _, child in active:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for item, cpus, child in active:
            child.wait()
            finished(item, cpus, child)
    return failed


def run(base, config):
    with (base / 'campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (base / 'campaign.pid').write_text(str(os.getpid()) + '\n')
        logs = {}

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)

        def start(p, cpus):
            log = (base / 'logs' / ('worker_' + p['key'] + '.log')).open('a')
            command = ['nice', '-n', '19', 'taskset', '-c', ','.join(map(str, cpus)),
                       sys.executable, str(Path(__file__).resolve()), 'worker', p['key'],
                       '--campaign-dir', str(base), '--cpus', ','.join(map(str, cpus))]
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            logs[child.pid] = log
            print(now(), 'STARTED', p['key'], child.pid, 'cores', cpus, flush=True)
            return child

        def finished(p, cpus, child):
            logs.pop(child.pid).close()
            print(now(), 'FINISHED', p['key'], child.returncode, flush=True)

        failed = run_queue(config['processes'], config['cpu_slots'], start, finished)
        dump(base / 'status.json', dict(updated_utc=now(), failed=[p['key'] for p in failed],
                                      status='failed' if failed else 'complete'))
        return int(bool(failed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'plan', 'rematch', 'run', 'worker', 'status'])
    parser.add_argument('key', nargs='?')
    parser.add_argument('--campaign-dir', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--events', type=int, default=1000000)
    parser.add_argument('--parallel-processes', type=int, default=2)
    parser.add_argument('--cores-per-process', type=int, default=6)
    parser.add_argument('--cpus', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    base = args.campaign_dir.resolve()
    if args.action == 'rematch':
        from rematch_macnlops_campaign import prepare_restart
        config = prepare_restart(base)
        return run(base, config)
    if args.action in ('prepare', 'plan'):
        config = (prepare if args.action == 'prepare' else new_config)(
            base, args.events, args.parallel_processes, args.cores_per_process)
        print(json.dumps(config, indent=2))
        return 0
    config = json.loads((base / 'config.json').read_text())
    # A taskset worker inherits only its own slot; validate the full allocation
    # in the parent and the assigned slot separately in the worker.
    if args.action == 'worker':
        cpus = [int(cpu) for cpu in (args.cpus or '').split(',') if cpu]
        if cpus not in config['cpu_slots'] or not set(cpus) <= os.sched_getaffinity(0):
            raise RuntimeError('Worker affinity differs from the recorded slot')
        validation = dict(config, cpu_slots=[cpus])
        validate_config(validation, base)
        if config.get('rematch_revision'):
            from rematch_macnlops_campaign import worker as rematch_worker
            return rematch_worker(base, config, args.key, cpus)
        return worker(base, config, args.key, cpus)
    validate_config(config, base, check_physics=args.action != 'status')
    if args.action == 'run':
        return run(base, config)
    if config['physics_sha256'] != fingerprint():
        print('Saved statuses refer to the original matching code; rematching has not been run.')
    for p in config['processes']:
        path = base / ('status_' + p['key'] + '.json')
        state = json.loads(path.read_text()) if path.exists() else {}
        print('%-12s %-9s %-28s %s' % (p['key'], state.get('status', 'pending'),
                                     state.get('stage', ''), state.get('log', '')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
