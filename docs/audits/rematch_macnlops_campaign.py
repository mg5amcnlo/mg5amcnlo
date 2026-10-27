#!/usr/bin/env python3
"""Resume the sampled campaign with corrected matching and immutable sources.

Existing first-emission streams are replayed, including previously successful
chunks. Original matching directories and campaign records are archived. Only
incomplete Pythia streams are regenerated. A separate native build records the
new numerical code while retaining the original source-generation manifests.
"""
import concurrent.futures
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback

import run_macnlops_campaign as campaign

ROOT = campaign.ROOT
sys.path.insert(0, str(ROOT))
from madgraph.various import macnlops as mc

MODULUS = 1 << 256
PACKETS = re.compile(r'<(macnlops_hard_event|macnlops_component|macnlops_delta_context)\b[^>]*>.*?</\1>', re.S)


class Population:
    """Order-independent hard-packet/weight multiset, including multiplicities."""
    def __init__(self):
        self.count = self.total = self.square = 0

    def add(self, block):
        packets = [' '.join(m[0].split()) for m in PACKETS.finditer(block)]
        if len(packets) < 2:
            raise RuntimeError('Missing immutable source packets in a rematch event')
        header = re.search(r'<event(?:\s[^>]*)?>\s*([^\n]+)', block)[1].split()
        weight = float(header[2].replace('D', 'E').replace('d', 'e'))
        value = int.from_bytes(hashlib.sha256(json.dumps(
            [weight, packets], separators=(',', ':')).encode()).digest(), 'big')
        self.count += 1
        self.total = (self.total + value) % MODULUS
        self.square = (self.square + value * value) % MODULUS

    def merge(self, value):
        self.count += value['count']
        self.total = (self.total + int(value['sum'], 16)) % MODULUS
        self.square = (self.square + int(value['sum_square'], 16)) % MODULUS

    def record(self):
        return dict(count=self.count, sum='%064x' % self.total,
                    sum_square='%064x' % self.square,
                    algorithm='SHA256(weight, immutable hard/component/Delta packets); multiset sums mod 2^256')


def prepare_restart(base):
    """Archive the old contract explicitly; never relabel old accepted samples."""
    with (base / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads((base / 'config.json').read_text())
        campaign.validate_config(previous, base, check_physics=False)
        if previous.get('rematch_revision'):
            campaign.validate_config(previous, base)
            return previous
        revision = 'matching_fixes_20261025'
        archive = base / 'history' / revision
        archive.mkdir(parents=True, exist_ok=False)
        for item in base.glob('*.json'):
            shutil.copy2(item, archive / item.name)
        shutil.copytree(base / 'cards', archive / 'cards')
        config = dict(previous, physics_sha256=campaign.fingerprint(),
                      rematch_revision=revision, restarted_utc=campaign.now(),
                      previous_config=str(archive / 'config.json'),
                      restart_runner_sha256=campaign.sha(Path(__file__)))
        for p in previous['processes']:
            process = base / 'processes' / p['directory']
            marker = process / 'campaign_export.json'
            if json.loads(marker.read_text()).get('contract') != campaign.contract_id(previous, p):
                raise RuntimeError('Unqualified original process export: ' + str(process))
            shutil.copy2(marker, archive / (p['key'] + '_export.json'))
        campaign.dump(base / 'config.json', config)
        for p in config['processes']:
            for mode in config['source_modes']:
                (base / (p['key'] + '_' + mode + '_complete.json')).unlink(missing_ok=True)
            campaign.dump(base / ('status_' + p['key'] + '.json'), dict(
                process=p['key'], status='queued', stage='rematch_pending',
                rematch_revision=revision, updated_utc=campaign.now()))
        campaign.dump(base / 'status.json', dict(status='running', stage='rematch_and_resume',
                      rematch_revision=revision, updated_utc=campaign.now()))
        return config


def native_build(directory, mode, revision):
    """Link new driver/counter objects to the unchanged generated amplitudes."""
    process = directory.parents[1]
    work = directory / ('.macnlops-' + revision + '-' + mode)
    work.mkdir(exist_ok=True)
    with (work / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = work / 'build.json'
        binary = work / 'macnlops_evaluator'
        pins = {name: campaign.sha(ROOT / 'Template/NLO/SubProcesses' / name)
                for name in ('driver_macnlops.f', 'montecarlocounter.f')}
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['sources'] != pins or campaign.sha(binary) != saved['binary_sha256']:
                raise RuntimeError('Rematch evaluator build changed: ' + str(work))
            return binary
        driver = (ROOT / 'Template/NLO/SubProcesses/driver_macnlops.f').read_text()
        if driver.count('      call setrun\n') != 1:
            raise RuntimeError('Cannot freeze the native source matching mode')
        driver = driver.replace('      call setrun\n',
            '      call setrun\n      mcatnlo_delta=%s\n' %
            ('.true.' if mode == 'delta' else '.false.'), 1)
        (work / 'driver.f').write_text(driver)
        shutil.copy2(ROOT / 'Template/NLO/SubProcesses/montecarlocounter.f', work / 'counter.f')
        commands = []
        with (work / 'build.log').open('w') as log:
            for stem in ('driver', 'counter'):
                cmd = ['gfortran', '-O2', '-c', '-ffixed-line-length-132',
                       '-I' + str(directory), '-I' + str(process / 'lib'),
                       str(work / (stem + '.f')), '-o', str(work / (stem + '.o'))]
                commands.append(cmd)
                subprocess.run(cmd, cwd=work, stdout=log, stderr=subprocess.STDOUT, check=True)
            dry = subprocess.check_output(['make', '-n', '-W', 'driver_macnlops.o',
                                           'macnlops_evaluator'], cwd=directory, text=True)
            links = [line for line in dry.splitlines() if ' -o macnlops_evaluator ' in line]
            if len(links) != 1:
                raise RuntimeError('Cannot isolate native evaluator link command')
            link = shlex.split(links[0])
            link[link.index('-o') + 1] = str(binary)
            link[link.index('driver_macnlops.o')] = str(work / 'driver.o')
            link[link.index('montecarlocounter.o')] = str(work / 'counter.o')
            objects = {token: campaign.sha(directory / token) for token in link
                       if token.endswith('.o') and (directory / token).is_file()}
            commands.append(link)
            subprocess.run(link, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
        campaign.dump(receipt, dict(sources=pins, source_mode=mode, commands=commands,
                      generated_objects=objects, binary=str(binary),
                      binary_sha256=campaign.sha(binary), created_utc=campaign.now()))
        return binary


class RematchEvaluator(mc.NativeEvaluator):
    def __init__(self, process, job, profile, mode, revision, environment, seed):
        super().__init__(process, job, environment=environment, auxiliary_seed=seed,
                         partition='sampled')
        self.construction = profile['construction']
        self.mode, self.revision = mode, revision
        self.builds = {}

    def _query(self, directory, request, request_id):
        if directory not in self.workers:
            binary = native_build(directory, self.mode, self.revision)
            self.builds[str(directory)] = str(binary.parent / 'build.json')
            log = (self.log_dir / (directory.name + '.log')).open('w')
            try:
                environment = dict(self.environment)
                environment['LD_LIBRARY_PATH'] = str(self.process_dir / 'lib') + ':' + environment['LD_LIBRARY_PATH']
                child = subprocess.Popen([str(binary)], cwd=directory, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=environment)
                self.workers[directory] = child, log
                child.stdin.write('MACNLOPS_CONFIG_V1 ' + ' '.join(
                    '%.17e' % value for value in self.configuration) + '\n')
                child.stdin.flush()
            except BaseException:
                log.close()
                raise
        return super()._query(directory, request, request_id)


def regenerate_proposals(driver, original, job, profile, seed, environment, source_header):
    """The failed Wj inputs still have their original Delta raw channel files."""
    filename = Path(original['input'])
    mc.check_manifest(filename, filename.parents[3])
    with tempfile.TemporaryDirectory(prefix='rematch_input_', dir=job) as temporary:
        plain = Path(temporary) / 'input.lhe'
        with mc.LHEStream(filename) as source, plain.open('w') as target:
            target.write(source_header)
            for block in source:
                target.write(block)
            target.write('</LesHouchesEvents>\n')
        cmd = [str(driver), '--input', str(plain), '--output', str(job / 'proposals.lhe'),
               '--metadata', str(job / 'shower.jsonl'), '--seed', str(seed),
               '--pdf', profile['pdf'], '--alpha-s-mz', str(profile['alpha_s_mz']),
               '--alpha-s-order', str(profile['alpha_s_order']),
               '--settings', str(job / 'shower_input.cmnd'),
               '--flavour-count', str(profile['support']['flavour_count'])]
        if profile['construction'] == mc.DELTA_SOURCE:
            cmd.append('--mcatnlo-delta')
        with (job / 'shower.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)


def replay(task):
    process, old, job, row, profile, mode, revision, pythia, driver, source_header = task
    job.mkdir(exist_ok=True)
    report_file = job / 'report.json'
    if report_file.exists():
        report = json.loads(report_file.read_text())
        if (report.get('rematch_revision') != revision or
                campaign.sha(job / 'events.lhe.gz') != report.get('output_sha256')):
            raise RuntimeError('Invalid completed rematch chunk: ' + str(job))
        return report
    for name in ('shower_input.cmnd', 'probe.json', 'probe.json.settings.cmnd', 'probe.log'):
        if (old / name).exists() and not (job / name).exists():
            (job / name).symlink_to(old / name)
    seed = int(re.search(r'(?m)^Random:seed\s*=\s*(\d+)',
               (old / 'shower.jsonl.settings.cmnd').read_text())[1])
    card = mc.run_card_values(re.search(r'<MGRunCard>(.*?)</MGRunCard>', source_header, re.S | re.I)[1])
    pdf, info = mc.pdf_name(process, card, with_info=True)
    environment = mc.shower_environment(pythia, info)
    if (pdf != profile['pdf'] or mc.sha256(info) != profile['pdf_info_sha256'] or
            mc.pdf_member_hash(pdf, info) != profile['pdf_member_sha256'] or
            mc.shower_library_hashes(pythia) != profile['shower_libraries'] or
            json.loads((job / 'probe.json').read_text()) != profile['support']):
        raise RuntimeError('Original source PDF/Pythia profile changed')
    regenerated = row.get('status') == 'failed' and 'returned non-zero exit status' in row.get('error', '')
    # Incomplete C++ runs have no closing LHE tag; complete streams are reused.
    with (old / 'proposals.lhe').open('rb') as proposal:
        proposal.seek(max(0, (old / 'proposals.lhe').stat().st_size - 100))
        regenerated = b'</LesHouchesEvents>' not in proposal.read()
    if regenerated:
        regenerate_proposals(driver, row, job, profile, seed, environment, source_header)
    else:
        for name in ('proposals.lhe', 'shower.jsonl', 'shower.jsonl.settings.cmnd', 'shower.log'):
            if not (job / name).exists():
                (job / name).symlink_to(old / name)
    shower_profile = mc.recorded_shower_profile(profile, job / 'shower.jsonl.settings.cmnd', environment)
    population = Population()
    original_stream = mc.LHEStream

    class AuditedStream(original_stream):
        def __iter__(self):
            for block in super().__iter__():
                population.add(block)
                yield block

    # Pool workers are separate processes; this only observes the proposal read.
    mc.LHEStream = AuditedStream
    try:
        with RematchEvaluator(process, job, profile, mode, revision, environment, seed) as evaluator:
            report = mc.reject_proposals(job / 'proposals.lhe', job / 'shower.jsonl',
                job / 'events.lhe.gz', evaluator, seed=seed, signed_residuals=True,
                profile=profile['support'], require_continuation=True,
                shower_profile=shower_profile,
                matching_policy=mc.matching_prescription(profile['construction'], 'sampled'))
            report['shower_support'] = evaluator.configuration
            report['native_builds'] = evaluator.builds
    finally:
        mc.LHEStream = original_stream
    report.update(input=row['input'], original_matching=str(old),
        source_construction=profile['construction'], matching_partition='sampled',
        rematch_revision=revision, regenerated_proposals=regenerated,
        source_population=population.record(), output_sha256=campaign.sha(job / 'events.lhe.gz'),
        proposals_sha256=campaign.sha(job / 'proposals.lhe'),
        metadata_sha256=campaign.sha(job / 'shower.jsonl'),
        pdf=pdf, shower=str(pythia), pdf_info_sha256=mc.sha256(info),
        effective_settings_sha256=mc.sha256(job / 'shower.jsonl.settings.cmnd'),
        normalization='sum; retained input event weights')
    campaign.dump(report_file, report)
    return report


def qualify_source(base, p, mode, source, profile):
    """Pin original generation files, allowing only the later mode's run card."""
    process = source.parents[2]
    saved = json.loads((base / (p['key'] + '_' + mode + '_source.json')).read_text())
    if campaign.sha(source) != saved['sha256']:
        raise RuntimeError('Original common source changed: ' + str(source))
    manifest = json.loads(Path(str(source) + '.macnlops.json').read_text())
    if mc.lhe_sha256(source) != manifest['lhe_sha256']:
        raise RuntimeError('Original generation manifest no longer matches the LHE')
    ignored = []
    for name, digest in manifest['sources'].items():
        if mc.sha256(process / name) == digest:
            continue
        if name == 'Cards/macnlops_profile.json':
            snapshot = base / 'cards' / p['key'] / mode / 'macnlops_profile.json'
            if campaign.sha(snapshot) != digest:
                raise RuntimeError('Original matching profile is missing')
        elif name != 'Cards/run_card.dat' and not name.endswith('/run_card.inc'):
            raise RuntimeError('Original generation input changed: ' + name)
        ignored.append(name)
    with mc.LHEStream(source) as stream:
        header = stream.header
        card_text = re.search(r'<MGRunCard>(.*?)</MGRunCard>', header, re.S | re.I)[1]
        original = mc.run_card_values(card_text)
        current = mc.run_card_values((process / 'Cards/run_card.dat').read_text())
        differences = [key for key in original if original[key] != current.get(key)]
        if set(differences) - {'iseed', 'mcatnlo_delta'}:
            raise RuntimeError('Source and evaluator run cards differ: ' + str(differences))
        population = Population()
        for block in stream:
            population.add(block)
    campaign.check_sample(saved, base_config_events(base))
    if (mc.source_construction(original) != profile['construction'] or
            profile['construction'] != (mc.DELTA_SOURCE if mode == 'delta' else mc.ORDINARY_SOURCE)):
        raise RuntimeError('Source construction differs from its saved generation profile')
    return saved, header, population.record(), dict(
        manifest=str(source) + '.macnlops.json', manifest_sha256=campaign.sha(str(source) + '.macnlops.json'),
        original_generation_sources=manifest['sources'], later_mode_run_inputs=ignored,
        evaluator_mode_override=mode)


def base_config_events(base):
    return json.loads((base / 'config.json').read_text())['events_per_source']


def install_for_remaining_generation(base, p, config):
    process = base / 'processes' / p['directory']
    backup = Path(config['previous_config']).parent / 'generation_inputs' / p['key']
    backup.mkdir(parents=True, exist_ok=True)
    files = [('Template/NLO/SubProcesses/' + name, 'SubProcesses/' + name)
             for name in ('driver_macnlops.f', 'montecarlocounter.f')]
    files += [('Template/NLO/MCatNLO/srcPythia8/MacNLOPS.cc', 'MCatNLO/srcPythia8/MacNLOPS.cc'),
              ('madgraph/various/macnlops.py', 'bin/internal/macnlops.py')]
    for source, target in files:
        target = process / target
        saved = backup / target.relative_to(process)
        saved.parent.mkdir(parents=True, exist_ok=True)
        if not saved.exists():
            shutil.copy2(target, saved)
        if campaign.sha(target) != campaign.sha(ROOT / source):
            shutil.copyfile(ROOT / source, target)
    marker = process / 'campaign_export.json'
    campaign.dump(marker, dict(contract=campaign.contract_id(config, p),
        original_export=str(Path(config['previous_config']).parent / (p['key'] + '_export.json')),
        matching_upgrade=config['rematch_revision'], complete_utc=campaign.now()))


def worker(base, config, key, cpus):
    p = next(p for p in config['processes'] if p['key'] == key)
    process = base / 'processes' / p['directory']
    revision = config['rematch_revision']
    state = dict(process=key, directory=str(process), cpus=cpus, pid=os.getpid(),
                 status='running', samples={}, rematch_revision=revision, started_utc=campaign.now())

    def update(**values):
        state.update(values, updated_utc=campaign.now())
        campaign.dump(base / ('status_' + key + '.json'), state)

    os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', MAKEFLAGS='-j1', TMPDIR=str(base / 'tmp'))
    try:
        if campaign.sha(Path(__file__)) != config['restart_runner_sha256']:
            raise RuntimeError('Restart runner changed after launch')
        for mode in config['source_modes']:
            source = process / 'Events' / (mode + '_1M') / 'events.lhe.gz'
            checkpoint = base / (key + '_' + mode + '_complete.json')
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if saved['contract'] != campaign.contract_id(config, p):
                    raise RuntimeError('Incorrect rematch checkpoint contract')
                for name in ('source', 'macnlops'):
                    if campaign.sha(saved[name]['path']) != saved[name]['sha256']:
                        raise RuntimeError('Rematched sample changed')
                state['samples'][mode] = saved
                continue
            if not source.exists():
                install_for_remaining_generation(base, p, config)
                return campaign.worker(base, config, key, cpus)
            update(stage=mode + '_qualify_original_source')
            profile = json.loads((base / 'cards' / key / mode / 'macnlops_profile.json').read_text())
            original, header, expected, provenance = qualify_source(base, p, mode, source, profile)
            root = source.parent / 'macnlops'
            old = source.parent / ('macnlops_before_' + revision)
            if not old.exists():
                root.rename(old)
            root.mkdir(exist_ok=True)
            campaign.dump(root / 'source_provenance.json', provenance)
            rows = json.loads((old / 'summary.json').read_text())
            driver = None
            for row in rows:
                name = '__'.join(Path(row['input']).relative_to(process).parts).removesuffix('.lhe')
                with (old / name / 'proposals.lhe').open('rb') as stream:
                    stream.seek(max(0, (old / name / 'proposals.lhe').stat().st_size - 100))
                    incomplete = b'</LesHouchesEvents>' not in stream.read()
                if incomplete and driver is None:
                    build = root / 'shower_build'
                    (build / 'MCatNLO/srcPythia8').mkdir(parents=True, exist_ok=True)
                    for name2 in ('MacNLOPS.cc', 'Makefile'):
                        shutil.copy2(ROOT / 'Template/NLO/MCatNLO/srcPythia8' / name2,
                                     build / 'MCatNLO/srcPythia8' / name2)
                    driver = mc.build_shower(build, config['pythia8'], root / 'shower_build.log')
            tasks = []
            for row in rows:
                name = '__'.join(Path(row['input']).relative_to(process).parts).removesuffix('.lhe')
                tasks.append((process, old / name, root / name, row, profile, mode, revision,
                              Path(config['pythia8']), driver, header))
            update(stage=mode + '_rematching', completed_chunks=0, total_chunks=len(tasks))
            reports = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(cpus)) as pool:
                futures = {pool.submit(replay, task):i for i, task in enumerate(tasks)}
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    reports[index] = future.result()
                    update(completed_chunks=len(reports))
                    print(campaign.now(), key, mode, 'REMATCHED', len(reports), '/', len(tasks), flush=True)
            reports = [reports[i] for i in range(len(tasks))]
            actual = Population()
            for report in reports:
                actual.merge(report['source_population'])
            if actual.record() != expected:
                raise RuntimeError('Saved proposals do not reproduce the common source population')
            campaign.dump(root / 'source_population.json', dict(source=expected, proposals=actual.record()))
            outputs = [task[2] / 'events.lhe.gz' for task in tasks]
            update(stage=mode + '_collect_rematched')
            accepted = root / 'events.lhe.gz'
            if not accepted.exists():
                collection = mc.collect_matched_events(outputs, reports, accepted,
                                                       p['seed'] + int(mode == 'delta') + 1)
                campaign.dump(root / 'collection.json', collection)
            else:
                collection = json.loads((root / 'collection.json').read_text())
            summary = [dict(row, status='complete', output=str(output),
                            collected_output=str(accepted)) for row, output in zip(rows, outputs)]
            for row in summary:
                row.pop('error', None)
            campaign.dump(root / 'summary.json', summary)
            derived = campaign.scan_lhe(accepted)
            count = collection['counts']
            campaign.check_sample(derived, count.get('accepted_S', 0) + count.get('accepted_H', 0))
            if (count.get('input_H', 0) + count.get('input_S', 0) != config['events_per_source'] or
                    abs(derived['sumw'] - collection['sumw']) > 1.e-10 * max(1., derived['sumabs']) or
                    campaign.sha(source) != original['sha256']):
                raise RuntimeError('Rematched source/collection totals disagree')
            from madgraph.various.banner import RunCardNLO
            card = RunCardNLO(str(base / 'cards' / key / mode / 'run_card.dat'))
            settings = {name:card[name] for name in card.keys()}
            saved = dict(contract=campaign.contract_id(config, p), source=original,
                macnlops=derived, matching_counts=count, settings=settings,
                veto_prescription=collection['veto_prescription'], fks_settings=campaign.FKS_SETTINGS,
                rematch_revision=revision, source_provenance=str(root / 'source_provenance.json'),
                source_population=str(root / 'source_population.json'), complete_utc=campaign.now(),
                normalization='sum; retained events keep their common-source weights')
            campaign.dump(checkpoint, saved)
            state['samples'][mode] = saved
            update(stage=mode + '_complete')
        update(status='complete', stage='complete', finished_utc=campaign.now())
        return 0
    except Exception as exc:
        update(status='failed', error=str(exc), failed_utc=campaign.now())
        traceback.print_exc()
        return 1
