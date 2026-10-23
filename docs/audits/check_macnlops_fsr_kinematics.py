#!/usr/bin/env python3
"""Check actual one-step global FSR against invariant inverse-map identities.

Usage: python3 docs/audits/check_macnlops_fsr_kinematics.py shower.jsonl [...]
This checks momenta and evolution scales, not Sudakovs or emission rates.
Massive g->qqbar pairs are reported separately, never projected as massless.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def add(a, b):
    return [x+y for x, y in zip(a, b)]


def dot(a, b):
    return a[0]*b[0]-sum(x*y for x, y in zip(a[1:], b[1:]))


def boost(p, velocity):
    beta2 = sum(x*x for x in velocity)
    if beta2 == 0.:
        return p[:]
    gamma = 1./math.sqrt(1.-beta2)
    product = sum(x*y for x, y in zip(velocity, p[1:]))
    return [gamma*(p[0]+product)]+[
        p[i+1]+((gamma-1.)*product/beta2+gamma*p[0])*velocity[i]
        for i in range(3)]


def check_record(record, tolerance=2.e-8):
    if not record['emitted'] or record['emission_type'] != 'FSR':
        return None
    event = {particle['index']: particle for particle in record['pythia_event']}
    indices = record['branch_indices']
    old = event[indices['radiator_before']]
    rad = event[indices['radiator_after']]
    emt = event[indices['emitted']]
    if old['id'] != 21 and not (1 <= abs(old['id']) <= 6 and emt['id'] == 21):
        return {'channel': 'excluded_non_qcd'}
    if emt['id'] != 21 and (rad['mass'] or emt['mass']):
        return {'channel': 'unsupported_massive_pair'}
    incoming = record['hard_pythia_indices'][:2]
    total = add(*(event[index]['p'] for index in incoming))
    pair = add(rad['p'], emt['p'])
    spectator = [p-q for p, q in zip(total, pair)]
    s, pair2 = dot(total, total), dot(pair, pair)
    mass2, recoil2 = old['mass']**2, dot(spectator, spectator)
    if min(s, pair2) <= 0:
        raise AssertionError('Nonphysical total/pair invariant')
    root = math.sqrt(s)
    epair = dot(total, pair)/root
    eemitted = dot(total, emt['p'])/root
    z = 1.-eemitted/(epair*(1.-mass2/pair2))
    t = z*(1.-z)*(pair2-mass2)
    if not 0 < z < 1 or t <= 0:
        raise AssertionError('Nonphysical inverse shower coordinates')
    eborn = (s+mass2-recoil2)/(2*root)
    kborn = math.sqrt(max(0., eborn*eborn-mass2))
    kpair = math.sqrt(max(0., epair*epair-pair2))
    coefficient_b = kborn/kpair
    coefficient_a = (eborn-coefficient_b*epair)/root
    born = [coefficient_a*p+coefficient_b*q for p, q in zip(total, pair)]
    momentum_error = max(abs(x-y) for x, y in zip(born, old['p']))/root
    scale_error = abs(t-record['pt_evol']**2)/max(t, 1.e-30)
    if momentum_error > tolerance or scale_error > tolerance:
        raise AssertionError('FSR sequence %s: Born error %.9g, scale error %.9g' %
                             (record['sequence'], momentum_error, scale_error))
    # Native FKS and global Pythia recoil both use one longitudinal boost
    # on every spectator in the hard CM. The light-cone expression also
    # works for a massless composite recoil, without a recoil-rest frame.
    total_velocity = [p/total[0] for p in total[1:]]
    pair_cm = boost(pair, [-v for v in total_velocity])
    direction = [p/kpair for p in pair_cm[1:]]
    erecoil_new, erecoil_born = root-epair, root-eborn
    rapidity = math.log((erecoil_new+kpair)/(erecoil_born+kborn))
    recoil_velocity = [math.tanh(rapidity)*n for n in direction]
    spectator_error = 0.
    for index in indices.get('recoilers', []):
        after = event[index]
        if after['status'] != 52:
            continue
        before = event[after['mothers'][0]]
        rest = boost(after['p'], [-v for v in total_velocity])
        restored = boost(boost(rest, recoil_velocity), total_velocity)
        spectator_error = max(spectator_error,
                              max(abs(x-y) for x, y in zip(restored, before['p']))/root)
    if spectator_error > tolerance:
        raise AssertionError('FSR sequence %s: spectator boost error %.9g' %
                             (record['sequence'], spectator_error))
    physical_pt = math.sqrt(t)*(1.-mass2/pair2)
    if emt['id'] == 21 and physical_pt < record['shower_cutoffs'][1]*(1.-tolerance):
        raise AssertionError('Accepted gluon is below the physical rescaled cutoff')
    return dict(channel='quark_emission' if abs(old['id']) <= 6 else
                        ('gluon_emission' if emt['id'] == 21 else 'massless_pair'),
                born_momentum_error=momentum_error, evolution_scale_error=scale_error,
                spectator_momentum_error=spectator_error)


def audit(paths):
    counts = Counter()
    max_born, max_scale, max_spectator = 0., 0., 0.
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                record = json.loads(line)
                result = check_record(record)
                if result is None:
                    continue
                counts[result['channel']] += 1
                max_born = max(max_born, result.get('born_momentum_error', 0.))
                max_scale = max(max_scale, result.get('evolution_scale_error', 0.))
                max_spectator = max(max_spectator, result.get('spectator_momentum_error', 0.))
    if not counts:
        raise ValueError('No accepted FSR records were supplied')
    return dict(counts=dict(counts), max_born_momentum_error=max_born,
                max_evolution_scale_error=max_scale,
                max_spectator_momentum_error=max_spectator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', nargs='+', type=Path)
    print(json.dumps(audit(parser.parse_args().metadata), indent=2, sort_keys=True))
