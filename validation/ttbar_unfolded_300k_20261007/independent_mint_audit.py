#!/usr/bin/env python3
"""Independent, streaming stdlib-only MINT tail and exact-event replay audit.

The two proposal streams are summarized independently, after pooling split
workers within each parent channel. Survey rates enter only once per stream.
No production analyzer, collector, or pool-reader code is imported.
"""
import argparse
from collections import Counter
import gzip
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re


def same(x, y):
    assert math.isclose(x, y, rel_tol=2e-10, abs_tol=1e-12), (x, y)


def number(text):
    return float(text.replace('D', 'e').replace('d', 'e'))


class Sum:
    """Neumaier compensated accumulator, with constant storage."""
    def __init__(self):
        self.s = self.c = 0.

    def add(self, value):
        total = self.s + value
        if abs(self.s) >= abs(value):
            self.c += (self.s - total) + value
        else:
            self.c += (value - total) + self.s
        self.s = total

    def total(self):
        return self.s + self.c


class Moments:
    def __init__(self):
        self.n = self.events = self.tail_events = self.zeros = 0
        self.max_ratio = 0.
        self.sums = [Sum() for _ in range(9)]

    def add(self, absolute, tail, excess, signed, accepted, ratio):
        self.n += 1
        self.events += accepted
        self.tail_events += ratio > 1.
        self.zeros += absolute == 0.
        self.max_ratio = max(self.max_ratio, ratio)
        for acc, value in zip(self.sums, (absolute, tail, excess, signed,
                                          absolute**2, tail**2, excess**2,
                                          absolute*tail, absolute*excess)):
            acc.add(value)

    def merge(self, other):
        self.n += other.n
        self.events += other.events
        self.tail_events += other.tail_events
        self.zeros += other.zeros
        self.max_ratio = max(self.max_ratio, other.max_ratio)
        for acc, source in zip(self.sums, other.sums):
            acc.add(source.total())

    def report(self, rate):
        n = self.n
        a, t, e, s, aa, tt, ee, at, ae = [x.total() for x in self.sums]
        assert n > 1 and a > 0., (n, a, rate)
        frac, excess_frac = t/a, e/a
        denominator = n*(n-1)
        return dict(rate=rate, n=n, events=self.events, tail_events=self.tail_events,
                    zero_trials=self.zeros, max_ratio=self.max_ratio,
                    frac=frac, excess_frac=excess_frac,
                    frac_err=math.sqrt(max(0., n*(tt-2*frac*at+frac**2*aa)
                                                / ((n-1)*a*a))),
                    excess_frac_err=math.sqrt(max(0., n*(ee-2*excess_frac*ae+excess_frac**2*aa)
                                                       / ((n-1)*a*a))),
                    mean=[a/n, t/n, e/n, s/n],
                    var_abs=max(0., aa-a*a/n)/denominator,
                    var_tail=max(0., tt-t*t/n)/denominator,
                    cov=(at-a*t/n)/denominator)


def events(path):
    """Yield exact event bytes, excluding surrounding LHE container text."""
    opener = gzip.open if path.suffix == '.gz' else open
    block = None
    closed = False
    with opener(path, 'rb') as handle:
        for line in handle:
            if b'</LesHouchesEvents>' in line:
                closed = True
            if block is None:
                match = re.search(rb'<event(?:\s[^>]*)?>', line)
                if match is None:
                    continue
                block = []
                line = line[match.start():]
            end = line.find(b'</event>')
            if end >= 0:
                block.append(line[:end+len(b'</event>')])
                yield b''.join(block)
                block = None
            else:
                block.append(line)
    assert closed and block is None, ('incomplete LHE', str(path))


def weight(event):
    header = next(line for line in event.splitlines()[1:]
                  if line.strip() and not line.lstrip().startswith(b'#'))
    return number(header.split()[2].decode())


def final_path(root, name):
    return next(path for path in (root/'Events'/name/'events.lhe.gz',
                                 root/'Events'/name/'events.lhe') if path.is_file())


def audit(work, run_name, expected_events):
    root, reference = work/'mint_tail', work/'mint'
    subprocesses = root/'SubProcesses'
    parents, workers, dimensions, event_multiset = {}, [], set(), Counter()
    roundoff_count, roundoff_max_abs, roundoff_max_scaled = 0, 0., 0.
    weighted_tail_sum = Sum()
    traces = sorted(subprocesses.glob('P*/GF*/mint_overweight_trials.dat'))
    assert traces, 'No diagnostic traces found'
    for path in traces:
        relative = path.parent.relative_to(subprocesses)
        parent_name = str(relative.parent/relative.name.split('_', 1)[0])
        streams = {}
        count = accepted = tails = 0
        accepted_tail_flags = []
        footer = None
        with path.open() as handle:
            assert handle.readline().strip() == 'MG5_MINT_TAIL 1'
            ndim, config, nchan = map(int, handle.readline().split())
            assert nchan == 1
            dimensions.add(ndim)
            rates = list(map(number, handle.readline().split()))
            normalizers = list(map(number, handle.readline().split()))
            assert len(rates) == len(normalizers) == 3
            parent = parents.setdefault(parent_name, dict(rates=rates, streams={},
                                                          workers=0, trials=0, events=0))
            for actual, previous in zip(rates, parent['rates']):
                same(actual, previous)
            for line in handle:
                if not line.strip():
                    continue
                assert footer is None, ('trailing records after footer', str(path))
                if line.startswith('# END'):
                    footer = list(map(int, line.split()[2:]))
                    continue
                fields = line.split()
                assert len(fields) == 9
                index, stream, acc, event_index = map(int, fields[:4])
                f, signed, bound, z, observation = map(number, fields[4:])
                count += 1
                assert index == count and stream in (1, 2, 3) and acc in (0, 1)
                assert all(math.isfinite(x) for x in (f, signed, bound, z, observation))
                assert f >= 0. and bound > 0. and z > 0.
                same(z, normalizers[stream-1])
                same(observation, f*z/bound)
                roundoff = abs(signed)-f
                scale = max(f, bound, z)
                if roundoff > 1e-10*f+1e-12:
                    # Signed and grouped absolute sums use different
                    # cancellation orders. Keep all observations unchanged;
                    # check small discrepancies against the proposal scale.
                    assert roundoff <= 1e-11*scale, (str(path), index, roundoff, scale)
                    roundoff_count += 1
                    roundoff_max_abs = max(roundoff_max_abs, roundoff)
                    roundoff_max_scaled = max(roundoff_max_scaled, roundoff/scale)
                flag = f > bound
                assert not flag or acc
                if acc:
                    accepted += 1
                    assert event_index == accepted
                    accepted_tail_flags.append(flag)
                else:
                    assert event_index in (0, accepted, accepted+1)
                tails += flag
                stats = streams.setdefault(stream, Moments())
                stats.add(observation, observation if flag else 0.,
                          (f-bound)*z/bound if flag else 0., signed*z/bound,
                          acc, f/bound)
        assert footer == [count, accepted, tails], (str(path), footer, count, accepted, tails)
        log = (path.parent/'log_MINT2.txt').read_text()
        logged_trials = re.findall(r'another call to the function:\s*(\d+)', log)
        logged_tails = re.findall(r'upper bound failure, (?:novi|virt|born):\s*(\d+)', log)
        assert logged_trials and int(logged_trials[-1]) == count
        assert sum(map(int, logged_tails)) == tails
        event_count = 0
        for event_count, (actual, baseline) in enumerate(zip_longest(
                events(path.parent/'events.lhe'),
                events(reference/'SubProcesses'/relative/'events.lhe')), 1):
            assert actual == baseline and actual is not None, ('worker event mismatch', str(relative), event_count)
            digest = hashlib.sha256(actual).digest()
            event_multiset[digest] += 1
            if accepted_tail_flags[event_count-1]:
                weighted_tail_sum.add(abs(weight(actual)))
        assert event_count == accepted
        for stream, stats in streams.items():
            parent['streams'].setdefault(stream, Moments()).merge(stats)
        parent['workers'] += 1
        parent['trials'] += count
        parent['events'] += accepted
        workers.append(dict(directory=str(relative), trials=count, events=accepted,
                            tail_events=tails, stream_trials={s: m.n for s, m in streams.items()}))

    expected = {str(p.parent.relative_to(subprocesses)):list(map(number, p.read_text().split()[:7]))
                for p in subprocesses.glob('P*/GF*/res_1.dat')
                if '_' not in p.parent.name and not p.is_symlink()}
    positive_parents = {name for name, fields in expected.items() if fields[0] > 0.}
    assert positive_parents <= set(parents), ('missing positive channels', positive_parents-set(parents))
    assert set(parents) <= set(expected), ('unknown channels', set(parents)-set(expected))
    records, channels, missing = [], {}, []
    for name, parent in sorted(parents.items()):
        nonvirtual, virtual, signed = parent['rates']
        same(expected[name][0], nonvirtual+virtual)
        same(expected[name][2], signed)
        observed = parent['streams']
        assert not (2 in observed and 3 in observed)
        active_nonvirtual = 3 if 3 in observed else 2
        summary = []
        for stream, rate in ((1, virtual), (active_nonvirtual, nonvirtual)):
            stats = observed.get(stream)
            if rate > 0. and (stats is None or stats.n == 0):
                missing.append((name, stream, rate))
            if stats is None:
                continue
            rec = dict(channel=name, stream=stream, **stats.report(rate))
            records.append(rec)
            summary.append(rec)
        rate = math.fsum(rec['rate'] for rec in summary)
        channels[name] = dict(workers=parent['workers'], trials=parent['trials'],
                              events=parent['events'], rate=rate,
                              tail=math.fsum(rec['rate']*rec['frac'] for rec in summary)/rate,
                              streams=summary)
    assert not missing, ('missing positive streams', missing)

    final_count = negative_events = 0
    sum_abs, sum_signed = Sum(), Sum()
    sequence_digest = hashlib.sha256()
    for final_count, (actual, baseline) in enumerate(zip_longest(
            events(final_path(root, run_name)), events(final_path(reference, run_name))), 1):
        assert actual == baseline and actual is not None, ('final event mismatch', final_count)
        digest = hashlib.sha256(actual).digest()
        assert event_multiset[digest] > 0, ('final event not in workers', final_count)
        event_multiset[digest] -= 1
        if event_multiset[digest] == 0:
            del event_multiset[digest]
        sequence_digest.update(digest)
        w = weight(actual)
        negative_events += w < 0.
        sum_abs.add(abs(w))
        sum_signed.add(w)
    assert not event_multiset and final_count == expected_events

    rate = math.fsum(rec['rate'] for rec in records)
    tail = math.fsum(rec['rate']*rec['frac'] for rec in records)
    error = math.sqrt(math.fsum((rec['rate']*rec['frac_err'])**2 for rec in records))/rate
    production_abs = math.fsum(rec['mean'][0] for rec in records)
    production_tail = math.fsum(rec['mean'][1] for rec in records)
    ratio = production_tail/production_abs
    production_error = math.sqrt(max(0., math.fsum(
        rec['var_tail']+ratio**2*rec['var_abs']-2*ratio*rec['cov']
        for rec in records)))/production_abs
    report = dict(all_raw_checks_passed=True, parent_channels=len(parents),
                  workers=len(workers), streams=len(records), coverage=1.,
                  missing_streams=missing, dimensions=sorted(dimensions),
                  trials=sum(p['trials'] for p in parents.values()),
                  events=final_count, negative_events=negative_events,
                  tail_events=sum(w['tail_events'] for w in workers),
                  survey_rate=rate, survey_full_tail=tail/rate,
                  survey_full_tail_error=error, production_full_tail=ratio,
                  production_full_tail_error=production_error,
                  production_absolute=production_abs,
                  production_signed=math.fsum(rec['mean'][3] for rec in records),
                  survey_excess_fraction=math.fsum(rec['rate']*rec['excess_frac'] for rec in records)/rate,
                  max_channel_tail=max(p['tail'] for p in channels.values()),
                  max_weight_ratio=max(rec['max_ratio'] for rec in records),
                  final_lhe_tail_weight_fraction=weighted_tail_sum.total()/sum_abs.total(),
                  final_lhe_sum_absolute_weights=sum_abs.total(),
                  final_lhe_sum_signed_weights=sum_signed.total(),
                  event_sequence_sha256=sequence_digest.hexdigest(),
                  exact_worker_and_final_event_replay=True,
                  exact_worker_multiset_preserved=True, channels=channels,
                  worker_records=workers)
    report['signed_absolute_roundoff'] = dict(
        rows=roundoff_count, maximum_absolute_excess=roundoff_max_abs,
        maximum_excess_over_proposal_scale=roundoff_max_scaled,
        allowed_excess_over_proposal_scale=1e-11,
        note='Signed and grouped absolute sums can differ at cancellation roundoff scale; observations are unchanged.')
    report['method'] = dict(
        proposal_correction='a = fABS * Z / H; full tail is a for fABS > H.',
        split_pooling='Pool moments within parent channel and stream; survey rate counted once.',
        uncertainty='Delta method conditional on grids, counts, and surveyed rates; accepted-event stopping and survey uncertainty are not included.',
        exact_replay='Direct byte equality of every worker and final event block in sequence; SHA256 counter checks the complete worker/final multiset.',
        memory='Streaming trial moments and LHE parsing; only event hashes and one worker accepted-event tail flags retained.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path)
    parser.add_argument('--run-name', default='benchmark_300k')
    parser.add_argument('--expected-events', type=int, default=300000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.work, args.run_name, args.expected_events)
    text = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(text)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ('channels', 'worker_records')}, indent=2))
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
