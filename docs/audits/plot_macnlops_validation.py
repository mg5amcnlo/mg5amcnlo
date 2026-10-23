#!/usr/bin/env python3
"""Weighted first-emission/full-parton MAcNLOPS validation figures.

Run with the analysis venv containing numpy, matplotlib, awkward, vector
and fastjet. Input is analysis/datasets.json, whose primary chunks contain
unrejected first/full LHE files plus their common decision JSONL. The script
never renormalizes survivors. In common_source_mode the ordinary MC@NLO
baseline and rejection stages use one source and one shower, with paired
covariance. Production requires 200k events before rejection in every
process; --allow-incomplete visibly labels smoke results.

FastJet's array interface is documented at
https://fastjet.readthedocs.io/en/latest/Awkward.html .
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import csv
import gzip
import hashlib
import html
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np


STAGES = ('reference', 'input', 'h_only', 's_only', 'macnlops',
          'H_positive', 'H_negative', 'S_positive', 'S_negative', 'H_total',
          'H_negative_retained')
PRIMARY = STAGES[1:]
LABELS = dict(reference='MC@NLO reference', input='Source, before rejection',
              h_only='H rejection only', s_only='S veto only', macnlops='MAcNLOPS',
              H_positive='Positive H', H_negative='Negative H',
              S_positive='Positive S', S_negative='Negative S', H_total='Total H',
              H_negative_retained='Retained negative H')
COLORS = dict(reference='#222222', input='#0072B2', h_only='#D55E00',
              s_only='#009E73', macnlops='#AA3377', H_positive='#56B4E9',
              H_negative='#D55E00', S_positive='#009E73', S_negative='#AA3377', H_total='#332288',
              H_negative_retained='#CC79A7')
PROCESS_NAMES = dict(dy=r'Drell–Yan $e^-e^+$', zz=r'$ZZ$',
                     ttbar=r'$t\bar t$', wbb=r'$W^+b\bar b$')
PROCESS_OBJECTS = dict(dy=((11, r'$e^-$'), (-11, r'$e^+$')),
                       zz=((23, r'$Z_1$'), (23, r'$Z_2$')),
                       ttbar=((6, r'$t$'), (-6, r'$\bar t$')),
                       wbb=((24, r'$W^+$'), (5, r'$b$'), (-5, r'$\bar b$')))
JET_Y_MIN_PT=20.
INDEPENDENT_VARIANCE='sumw2; paired covariance within primary, independent reference'
SHARED_VARIANCE='sumw2; paired covariance including the shared ordinary MC@NLO baseline'


def comparison_mode(config):
    """Validate the explicit shared-cohort contract before reading events."""
    common=config.get('common_source_mode',False)
    if type(common) is not bool:
        raise ValidationError('common_source_mode must be a boolean')
    signed=config.get('signed_residuals',config.get('allow_signed_residuals',False))
    if type(signed) is not bool:
        raise ValidationError('signed_residuals must be a boolean')
    if 'signed_residuals' in config and 'allow_signed_residuals' in config and signed!=config['allow_signed_residuals']:
        raise ValidationError('Conflicting signed-residual policies')
    if common:
        if config.get('source_construction')!='ordinary-mcatnlo-v1':
            raise ValidationError('Shared input requires source_construction ordinary-mcatnlo-v1')
        if config.get('reference'):
            raise ValidationError('Shared input derives its baseline from primary; omit reference chunks')
        if config.get('reference_complete',config.get('primary_complete',True))!=config.get('primary_complete',True):
            raise ValidationError('Shared baseline and primary completion flags differ')
    return common,signed


def stage_labels(config):
    labels=dict(LABELS)
    if config.get('source_construction')=='resolved-convex-v1':
        labels['input']='Resolved source, before rejection'
    if config.get('common_source_mode',False):
        labels.update(reference='Ordinary MC@NLO, before rejection',input='Same ordinary input',
                      h_only='H rejection only')
    elif config.get('signed_residuals',config.get('allow_signed_residuals',False)):
        labels['h_only']='H rejection only'
    return labels


class ValidationError(RuntimeError):
    pass


@dataclass
class Event:
    weight: float
    kind: str | None
    # Stable particles are (PDG, px, py, pz, E).
    particles: list
    # The shower writers preserve the hard-event header and aMC@NLO row.
    # Their digest catches a first/full reorder even for equal-weight events.
    source_identity: str = ''


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def read_lhe(path):
    """Stream central-weight events without loading the LHE into memory."""
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        for line in stream:
            if not re.match(r'^\s*<event(?:\s[^>]*)?>\s*$', line):
                continue
            block = []
            for line in stream:
                if line.strip() == '</event>':
                    break
                block.append(line)
            else:
                raise ValidationError('Unclosed LHE event in '+str(path))
            content = [line.strip() for line in block if line.strip()]
            if not content:
                raise ValidationError('Empty LHE event in '+str(path))
            fields = content[0].split()
            n, weight = int(fields[0]), number(fields[2])
            identity = str(int(fields[1]))+' '+ ' '.join(
                format(number(x), '.17g') for x in fields[2:6])
            if not math.isfinite(weight):
                raise ValidationError('Nonfinite source weight in '+str(path))
            particles = []
            if len(content) <= n:
                raise ValidationError('Truncated particle list in '+str(path))
            for line in content[1:n+1]:
                fields = line.split()
                if len(fields) < 13:
                    raise ValidationError('Malformed LHE particle in '+str(path))
                if int(fields[1]) != 1:
                    continue
                momentum = tuple(number(x) for x in fields[6:10])
                if not all(math.isfinite(x) for x in momentum) or momentum[3] < 0:
                    raise ValidationError('Invalid stable-particle momentum in '+str(path))
                particles.append((int(fields[0]),)+momentum)
            kind = None
            for line in content[n+1:]:
                match = re.match(r'#aMCatNLO\s+([12])\s', line)
                if match:
                    kind = 'S' if match[1] == '1' else 'H'
                    identity += '\n'+' '.join(line.split())
                    break
            yield Event(weight, kind, particles, identity)


def read_decisions(path):
    with Path(path).open() as stream:
        for sequence, line in enumerate(stream, 1):
            decision = json.loads(line)
            if decision.get('sequence') != sequence:
                raise ValidationError('Nonsequential decision in '+str(path))
            if decision.get('kind') not in ('S', 'H') or type(decision.get('accepted')) is not bool:
                raise ValidationError('Invalid decision kind/acceptance in '+str(path))
            yield decision


AUDIT_SETTINGS=('first_trial_headroom','continuation_trial_headroom',
                'continuation_policy','continuation_candidate_cache',
                'global_recoil_mode','n_max_global_branch')


def read_full_audits(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def new_full_audit():
    return dict(events=0,strict_s_events=0,h_events=0,events_with_warnings=0,
                emission_counts=Counter(),native_trial_warning_counts=Counter(),
                events_with_warnings_by_kind=Counter(),native_warning_counts_by_kind=dict(S=Counter(),H=Counter()),
                settings={key:set() for key in AUDIT_SETTINGS})


def add_full_audit(summary,record,sequence,kind,path):
    if record.get('schema')!='macnlops-full-audit-v1' or record.get('sequence')!=sequence or record.get('kind')!=kind:
        raise ValidationError('Full audit event identity mismatch in '+str(path))
    stock=record.get('continuation_policy')=='native-pythia-with-recorded-warnings'
    expected_strict=kind=='S' and not stock
    if type(record.get('first_endpoint_strict')) is not bool or record['first_endpoint_strict']!=expected_strict:
        raise ValidationError('Full audit first-endpoint policy mismatch in '+str(path))
    emissions=record.get('emissions')
    if type(emissions) is not int or emissions<0:
        raise ValidationError('Invalid full-shower emission count in '+str(path))
    if any(key not in record for key in ('continuation_trial_headroom','continuation_policy')):
        raise ValidationError('Missing native continuation settings in '+str(path))
    summary['events']+=1
    summary['strict_s_events']+=kind=='S' and record['first_endpoint_strict'];summary['h_events']+=kind=='H'
    summary['emission_counts'][str(emissions)]+=1
    warnings=record.get('native_trial_warning_counts')
    if not isinstance(warnings,list):
        raise ValidationError('Missing native continuation diagnostics in '+str(path))
    warning_count=0
    for warning in warnings:
        if (not isinstance(warning.get('message'),str) or
                type(warning.get('count')) is not int or warning['count']<0):
            raise ValidationError('Invalid native continuation diagnostic in '+str(path))
        summary['native_trial_warning_counts'][warning['message']]+=warning['count']
        summary['native_warning_counts_by_kind'][kind][warning['message']]+=warning['count']
        warning_count+=warning['count']
    if record['first_endpoint_strict'] and emissions==0 and warning_count:
        raise ValidationError('Native warning contradicts strict no-emission S audit in '+str(path))
    summary['events_with_warnings']+=warning_count>0
    summary['events_with_warnings_by_kind'][kind]+=warning_count>0
    for key in AUDIT_SETTINGS:
        if key in record:
            value=record[key]
            if key.endswith('headroom') and (not isinstance(value,(int,float)) or not math.isfinite(value) or value<1):
                raise ValidationError('Invalid native trial headroom in '+str(path))
            if key in ('continuation_policy','continuation_candidate_cache') and not isinstance(value,str):
                raise ValidationError('Invalid native continuation policy in '+str(path))
            summary['settings'][key].add(value)


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def merge_full_audits(chunks):
    if not chunks or any(chunk['full_audit'] is None for chunk in chunks):return None
    summary=new_full_audit()
    for chunk in chunks:
        record=chunk['full_audit']
        for key in ('events','strict_s_events','h_events','events_with_warnings'):summary[key]+=record[key]
        for key in ('emission_counts','native_trial_warning_counts'):summary[key].update(record[key])
        summary['events_with_warnings_by_kind'].update(record['events_with_warnings_by_kind'])
        for kind in ('S','H'):summary['native_warning_counts_by_kind'][kind].update(record['native_warning_counts_by_kind'][kind])
        for key in AUDIT_SETTINGS:summary['settings'][key].update(record['settings'][key])
    summary['settings']={key:sorted(value) for key,value in summary['settings'].items()}
    summary['raw_audits']=[dict(path=c['full_audit']['path'],sha256=c['full_audit']['sha256']) for c in chunks]
    return summary


def rapidity(p):
    px, py, pz, energy = p
    if energy <= abs(pz):
        return math.copysign(math.inf, pz) if pz else 0.
    return .5*math.log((energy+pz)/(energy-pz))


def pt(p):
    return math.hypot(p[0], p[1])


def pick_objects(event, process):
    selected = []
    if process == 'zz':
        selected = sorted([p[1:] for p in event.particles if p[0] == 23],
                          key=pt, reverse=True)
        if len(selected) != 2:
            raise ValidationError('ZZ event does not have two stable Z bosons')
        return selected
    for pdg, label in PROCESS_OBJECTS[process]:
        matches = [p[1:] for p in event.particles if p[0] == pdg]
        if len(matches) != 1:
            raise ValidationError('%s event has %d stable objects with PDG %d' %
                                  (process, len(matches), pdg))
        selected.append(matches[0])
    return selected


def leading_jets(events, exclude_bottom=False, radius=.4):
    """Cluster all stable light coloured partons in an Awkward batch."""
    import awkward as ak
    import fastjet
    import vector
    vector.register_awkward()
    rows = [[p[1:] for p in event.particles
             if (1 <= abs(p[0]) <= 5 or p[0] == 21)
             and not (exclude_bottom and abs(p[0]) == 5)] for event in events]
    counts = np.asarray([len(row) for row in rows], dtype=np.int64)
    flat = np.asarray([p for row in rows for p in row], dtype=np.float64).reshape((-1, 4))
    data = ak.zip({name: ak.unflatten(flat[:, column], counts)
                   for column, name in enumerate(('px', 'py', 'pz', 'E'))},
                  with_name='Momentum4D')
    cluster = fastjet.ClusterSequence(data, fastjet.JetDefinition(fastjet.antikt_algorithm, radius))
    jets = cluster.inclusive_jets()
    leading = ak.firsts(jets[ak.argsort(jets.pt, axis=-1, ascending=False)])
    components = [np.asarray(ak.to_numpy(ak.fill_none(leading[name], np.nan)))
                  for name in ('px', 'py', 'pz', 'E')]
    transverse = np.hypot(components[0], components[1])
    transverse = np.where(np.isnan(transverse), 0., transverse)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = .5*np.log((components[3]+components[2])/(components[3]-components[2]))
    return transverse, y


def observable_values(events, process):
    objects = [pick_objects(event, process) for event in events]
    values = {name: [] for name in ('system_pt', 'system_mass', 'system_y', 'delta_phi', 'delta_y')}
    for obj in objects:
        total = np.sum(np.asarray(obj), axis=0)
        values['system_pt'].append(pt(total))
        mass2 = total[3]**2-float(np.dot(total[:3], total[:3]))
        if mass2 < -1.e-10*total[3]**2:
            raise ValidationError('Spacelike hard-object system')
        values['system_mass'].append(math.sqrt(max(0., mass2)))
        values['system_y'].append(rapidity(total))
        first, second = obj[-2:] if process == 'wbb' else obj[:2]
        difference = math.atan2(first[1], first[0])-math.atan2(second[1], second[0])
        values['delta_phi'].append(abs(math.atan2(math.sin(difference), math.cos(difference))))
        values['delta_y'].append(rapidity(first)-rapidity(second))
    for i in range(len(PROCESS_OBJECTS[process])):
        values['object_%d_pt' % (i+1)] = [pt(obj[i]) for obj in objects]
        values['object_%d_y' % (i+1)] = [rapidity(obj[i]) for obj in objects]
    if process=='wbb':
        bottom_pair=np.asarray([np.sum(np.asarray(obj[-2:]),axis=0) for obj in objects])
        mass2=bottom_pair[:,3]**2-np.sum(bottom_pair[:,:3]**2,axis=1)
        if np.any(mass2 < -1.e-10*bottom_pair[:,3]**2):
            raise ValidationError('Spacelike bottom-pair system')
        values['pair_pt']=np.hypot(bottom_pair[:,0],bottom_pair[:,1])
        values['pair_mass']=np.sqrt(np.maximum(0.,mass2))
        values['pair_y']=[rapidity(pair) for pair in bottom_pair]
    values = {name: np.asarray(value, dtype=np.float64) for name, value in values.items()}
    values['delta_phi_deficit'] = np.maximum(0.,math.pi-values['delta_phi'])
    values['jet_pt'], values['jet_y'] = leading_jets(events, exclude_bottom=process == 'wbb')
    values['jet_pt_soft']=values['jet_pt']
    values['jet_y']=np.where(values['jet_pt']>JET_Y_MIN_PT,values['jet_y'],np.nan)
    if process == 'wbb':
        values['all_jet_pt'], values['all_jet_y'] = leading_jets(events)
        values['all_jet_y']=np.where(values['all_jet_pt']>JET_Y_MIN_PT,values['all_jet_y'],np.nan)
    return values


@dataclass
class Observable:
    edges: np.ndarray
    label: str
    unit: str = ''
    logarithmic: bool = False


def observable_specs(process, custom_bins=None):
    mass_ranges = dict(dy=(60., 180.), zz=(160., 1400.),
                       ttbar=(340., 2400.), wbb=(80., 1600.))
    lo, hi = mass_ranges[process]
    specs = dict(
        system_pt=Observable(np.r_[0., np.geomspace(.2, 1200., 61)],
                             r'$p_T$ of hard-object system', 'GeV', True),
        system_mass=Observable(np.linspace(lo, hi, 61), 'Hard-object invariant mass', 'GeV'),
        system_y=Observable(np.linspace(-5., 5., 51), 'Hard-object system rapidity'),
        delta_phi=Observable(np.linspace(0., math.pi, 41), r'$|\Delta\phi|$ of object pair'),
        delta_phi_deficit=Observable(np.r_[0.,np.geomspace(1.e-5,math.pi,61)],
                                    r'$\pi-|\Delta\phi|$ of object pair', '', True),
        delta_y=Observable(np.linspace(-8., 8., 65), r'$\Delta y$ of object pair'),
        jet_pt=Observable(np.linspace(0., 600., 61), 'Leading radiation jet '+r'$p_T$', 'GeV', True),
        jet_pt_soft=Observable(np.linspace(0.,50.,101), 'Leading radiation jet '+r'$p_T$ (soft region)', 'GeV', True),
        jet_y=Observable(np.linspace(-6., 6., 61), r'Leading radiation jet rapidity ($p_T>20$ GeV)'))
    for i, (pdg, label) in enumerate(PROCESS_OBJECTS[process], 1):
        specs['object_%d_pt' % i] = Observable(np.linspace(0., 600., 61), label+' '+r'$p_T$', 'GeV')
        specs['object_%d_y' % i] = Observable(np.linspace(-5., 5., 51), label+' rapidity')
    if process == 'wbb':
        specs['pair_pt']=Observable(np.r_[0.,np.geomspace(.2,1200.,61)],r'$p_T$ of bottom pair','GeV',True)
        specs['pair_mass']=Observable(np.geomspace(9.4,1500.,61),r'$m_{b\bar b}$','GeV',True)
        specs['pair_y']=Observable(np.linspace(-5.,5.,51),'Bottom-pair rapidity')
        specs['delta_phi'].label=r'$|\Delta\phi(b,\bar b)|$'
        specs['delta_phi_deficit'].label=r'$\pi-|\Delta\phi(b,\bar b)|$'
        specs['delta_y'].label=r'$\Delta y(b,\bar b)$'
        specs['all_jet_pt'] = Observable(np.linspace(0., 600., 61), 'Leading all-parton jet '+r'$p_T$', 'GeV', True)
        specs['all_jet_y'] = Observable(np.linspace(-6., 6., 61), r'Leading all-parton jet rapidity ($p_T>20$ GeV)')
    for name, bins in (custom_bins or {}).items():
        if name not in specs:
            raise ValidationError('Unknown custom observable '+name)
        edges = np.asarray(bins, dtype=np.float64)
        if len(edges) < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValidationError('Invalid bin edges for '+name)
        specs[name].edges = edges
    return specs


def stage_masks(weights, kinds, accepted, reference=False, common_source=False):
    n = len(weights)
    masks = np.zeros((len(STAGES), n), dtype=bool)
    if reference:
        masks[0] = True
        return masks
    hard = kinds == 'H'
    masks[0] = common_source
    masks[1] = True
    masks[2] = ~hard | accepted
    masks[3] = hard | accepted
    masks[4] = accepted
    masks[5] = hard & (weights >= 0)
    masks[6] = hard & (weights < 0)
    masks[7] = ~hard & (weights >= 0)
    masks[8] = ~hard & (weights < 0)
    masks[9] = hard
    masks[10] = hard & (weights < 0) & accepted
    return masks


class Histogram:
    def __init__(self, edges):
        self.edges = np.asarray(edges, dtype=np.float64)
        shape = (len(STAGES), len(edges)-1)
        self.sumw = np.zeros(shape)
        self.sumabsw = np.zeros(shape)
        self.sumw2 = np.zeros(shape)
        self.count = np.zeros(shape, dtype=np.int64)
        self.covariance = np.zeros((len(STAGES),)+shape)
        self.flows = {name: np.zeros((len(STAGES), 3))
                      for name in ('underflow', 'overflow', 'missing', 'zero')}

    def fill(self, values, weights, masks):
        # The final right bin edge is inclusive, matching numpy.histogram.
        bins = np.searchsorted(self.edges, values, side='right')-1
        bins[values == self.edges[-1]] = len(self.edges)-2
        flow_masks = dict(underflow=values < self.edges[0], overflow=values > self.edges[-1],
                          missing=np.isnan(values), zero=values == 0.)
        inside = (bins >= 0) & (bins < len(self.edges)-1) & ~np.isnan(values)
        for i, stage in enumerate(STAGES):
            select = masks[i] & inside
            self.sumw[i] += np.bincount(bins[select], weights=weights[select], minlength=self.sumw.shape[1])
            self.sumabsw[i] += np.bincount(bins[select], weights=np.abs(weights[select]), minlength=self.sumw.shape[1])
            self.sumw2[i] += np.bincount(bins[select], weights=weights[select]**2, minlength=self.sumw.shape[1])
            self.count[i] += np.bincount(bins[select], minlength=self.sumw.shape[1])
            for name, flow in flow_masks.items():
                chosen = masks[i] & flow
                self.flows[name][i] += (int(np.sum(chosen)), float(np.sum(weights[chosen])),
                                       float(np.sum(weights[chosen]**2)))
            for j in range(i+1):
                shared = select & masks[j]
                covariance = np.bincount(bins[shared], weights=weights[shared]**2,
                                         minlength=self.sumw.shape[1])
                self.covariance[i, j] += covariance
                if i != j:
                    self.covariance[j, i] += covariance

    @property
    def display_sumw(self):
        # Equal positive/negative event weights can leave a floating-point
        # cancellation residual. Keep raw sums, but do not draw a spurious
        # fifteen-decade log tail or divide by a numerically zero bin.
        result = self.sumw.copy()
        result[np.abs(result) <= 8*np.finfo(float).eps*self.sumabsw] = 0.
        return result


def ratio_with_variance(numerator, denominator, variance_numerator,
                        variance_denominator, covariance=0.):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    covariance = np.broadcast_to(covariance, numerator.shape)
    vn = np.broadcast_to(variance_numerator, numerator.shape)
    vd = np.broadcast_to(variance_denominator, numerator.shape)
    ratio = np.full(numerator.shape, np.nan)
    variance = np.full(numerator.shape, np.nan)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    a, b = numerator[valid], denominator[valid]
    ratio[valid] = a/b
    result = vn[valid]/b**2+a*a*vd[valid]/b**4-2*a*covariance[valid]/b**3
    scale = np.maximum(vn[valid]/b**2+a*a*vd[valid]/b**4, 1.e-300)
    if np.any(result < -1.e-10*scale):
        raise ValidationError('Nonpositive-semidefinite ratio covariance')
    # Identical event masks have an exactly constant ratio, even when
    # separately evaluated floating-point terms leave a positive ulp.
    result[(a==b)&(vn[valid]==vd[valid])&(covariance[valid]==vn[valid])]=0.
    variance[valid] = np.maximum(0., result)
    return ratio, variance


def resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else base/path


def fill_chunk(chunk, level, process, config, base, histograms, totals, reference, batch_size,audit_required=False):
    if level not in chunk:
        raise ValidationError('Missing %s file for %s' % (level, process))
    path = resolve(chunk[level], base)
    common,signed=comparison_mode(config)
    decisions = None if reference else read_decisions(resolve(chunk['decisions'], base))
    sentinel = object()
    audit_path=resolve(chunk['full_audit'],base) if level=='full' and chunk.get('full_audit') else None
    if audit_required and audit_path is None:
        raise ValidationError('Production full-shower chunk has no full_audit path: '+str(path))
    audits=read_full_audits(audit_path) if audit_path else None
    full_audit=new_full_audit() if audits is not None else None
    rows = read_lhe(path)
    if decisions is not None:
        rows = itertools.zip_longest(rows, decisions, fillvalue=sentinel)
    else:
        rows = ((event, None) for event in rows)
    digest, identity_digest = hashlib.sha256(), hashlib.sha256()
    batch, kinds, accepted, probabilities = [], [], [], []
    diagnostic_groups=('total','H_positive','H_negative','S_positive','S_negative')
    veto_diagnostic={group:np.zeros(7) for group in diagnostic_groups}
    probabilities_complete=True
    n = 0

    def consume():
        if not batch:
            return
        weights = np.asarray([event.weight for event in batch])
        masks = stage_masks(weights, np.asarray(kinds), np.asarray(accepted), reference,common)
        if not reference and all(p is not None for p in probabilities):
            p=np.asarray(probabilities)
            for group in diagnostic_groups:
                selected=np.ones(len(weights),dtype=bool) if group=='total' else masks[STAGES.index(group)]
                w=weights[selected];probability=p[selected];kept=np.asarray(accepted)[selected]
                veto_diagnostic[group]+=(len(w),np.sum(w),np.sum(w[kept]),np.sum(w*probability),
                    np.sum(w*w*probability*(1-probability)),np.sum(1-probability),np.sum(~kept))
        values = observable_values(batch, process)
        for name, histogram in histograms.items():
            histogram.fill(values[name], weights, masks)
        for i in range(len(STAGES)):
            w = weights[masks[i]]
            totals[i] += (len(w), np.sum(w), np.sum(w*w), np.sum(w < 0))
        batch.clear(); kinds.clear(); accepted.clear(); probabilities.clear()

    for event, decision in rows:
        if event is sentinel or decision is sentinel:
            raise ValidationError('LHE/decision event count mismatch in '+str(path))
        kind = event.kind if reference else decision['kind']
        if decision is not None and event.kind is not None and event.kind != kind:
            raise ValidationError('Source H/S annotation differs from decision in '+str(path))
        if decision is not None:
            if kind=='H' and event.weight>=0 and not decision['accepted']:
                raise ValidationError('Positive H event was rejected in '+str(path))
            if not signed and kind == 'H' and decision['accepted'] != (event.weight >= 0):
                raise ValidationError('Pure MAcNLOPS H sign/decision mismatch in '+str(path))
        probability=None if decision is None else decision.get('acceptance')
        if probability is not None:
            if not isinstance(probability,(int,float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValidationError('Invalid acceptance probability in '+str(path))
            if (probability==0 and decision['accepted']) or (probability==1 and not decision['accepted']):
                raise ValidationError('Deterministic probability/decision mismatch in '+str(path))
            if kind=='H' and event.weight>=0 and probability!=1:
                raise ValidationError('Positive H acceptance probability differs from one in '+str(path))
        elif not reference:
            if common:
                raise ValidationError('Shared-input decision has no acceptance probability in '+str(path))
            probabilities_complete=False
        digest.update(struct.pack('<d', event.weight))
        identity = event.source_identity.encode()
        identity_digest.update(struct.pack('<Q',len(identity)))
        identity_digest.update(identity)
        n += 1
        if audits is not None:
            record=next(audits,sentinel)
            if record is sentinel:raise ValidationError('Fewer full audit than LHE events in '+str(path))
            add_full_audit(full_audit,record,n,kind,audit_path)
        batch.append(event); kinds.append(kind or 'unknown')
        accepted.append(True if reference else decision['accepted'])
        probabilities.append(probability)
        if len(batch) >= batch_size:
            consume()
    consume()
    if 'events' in chunk and n!=chunk['events']:
        raise ValidationError('Chunk LHE event count differs from declared count in '+str(path))
    if audits is not None:
        if next(audits,sentinel) is not sentinel:raise ValidationError('More full audit than LHE events in '+str(path))
        full_audit['settings']={key:sorted(value) for key,value in full_audit['settings'].items()}
        full_audit.update(path=str(audit_path.resolve()),sha256=file_sha256(audit_path))
    return dict(path=str(path.resolve()), events=n, weight_sequence_sha256=digest.hexdigest(),
                source_sequence_sha256=identity_digest.hexdigest(),
                veto_diagnostic=None if reference or not probabilities_complete else
                    {group:value.tolist() for group,value in veto_diagnostic.items()},full_audit=full_audit)


def analyze_process(process, config, base, levels, allow_incomplete=False, batch_size=2048):
    if process not in PROCESS_OBJECTS:
        raise ValidationError('Unknown process '+process)
    common,signed=comparison_mode(config)
    specs = observable_specs(process, config.get('bins'))
    output = {}
    expected = int(config.get('expected_events', 200000))
    if not allow_incomplete and expected != 200000:
        raise ValidationError('Production requires 200000 source events per process')
    if not allow_incomplete and config.get('flavours') != (4 if process=='wbb' else 5):
        raise ValidationError('Production requires genuine4F Wbb and5F DY/ZZ/ttbar samples')
    for level in levels:
        histograms = {name: Histogram(spec.edges) for name, spec in specs.items()}
        totals = np.zeros((len(STAGES), 4))
        chunk_reports = {}
        for sample in (('primary',) if common else ('primary','reference')):
            if not config.get(sample):
                raise ValidationError('Missing '+sample+' sample for '+process)
            if not allow_incomplete and config.get(sample+'_complete',True) is not True:
                raise ValidationError('Incomplete production status for '+process+' '+sample)
            chunk_reports[sample] = [fill_chunk(chunk, level, process, config, base,
                histograms, totals, sample == 'reference', batch_size,
                audit_required=level=='full' and not allow_incomplete) for chunk in config[sample]]
            signatures=[(chunk['events'],chunk['source_sequence_sha256'])
                        for chunk in chunk_reports[sample] if chunk['events']]
            paths=[chunk['path'] for chunk in chunk_reports[sample]]
            if len(set(paths)) != len(paths) or len(set(signatures)) != len(signatures):
                raise ValidationError('%s %s %s repeats a source chunk' % (process,level,sample))
            count = sum(chunk['events'] for chunk in chunk_reports[sample])
            if not allow_incomplete and count != expected:
                raise ValidationError('%s %s %s has %d events; expected %d' %
                                      (process, level, sample, count, expected))
        if common:
            chunk_reports['reference']=[dict(chunk,derived_from_primary=True,veto_diagnostic=None)
                                        for chunk in chunk_reports['primary']]
            if not np.array_equal(totals[0],totals[1]):
                raise ValidationError('Shared baseline/input totals differ for '+process)
            for histogram in histograms.values():
                if any(not np.array_equal(getattr(histogram,key)[0],getattr(histogram,key)[1])
                       for key in ('sumw','sumabsw','sumw2','count','covariance')):
                    raise ValidationError('Shared baseline/input histogram or covariance differs for '+process)
        else:
            primary_paths={chunk['path'] for chunk in chunk_reports['primary']}
            reference_paths={chunk['path'] for chunk in chunk_reports['reference']}
            if primary_paths & reference_paths:
                raise ValidationError('%s %s reuses primary events as independent reference' % (process,level))
        continuation_audit={sample:merge_full_audits(chunk_reports[sample]) for sample in ('primary','reference')}
        if all(continuation_audit.values()) and continuation_audit['primary']['settings']!=continuation_audit['reference']['settings']:
            raise ValidationError('%s primary/reference full continuation settings differ' % process)
        stage_totals = {}
        for stage, (count, sumw, sumw2, negative) in zip(STAGES, totals):
            stage_totals[stage] = dict(events=int(count), negative_events=int(negative),
                sumw=float(sumw), sumw2=float(sumw2),
                effective_events=float(sumw*sumw/sumw2) if sumw2 else 0.)
        complete = all(sum(c['events'] for c in chunk_reports[sample]) == expected
                       for sample in ('primary', 'reference'))
        veto_diagnostic=None
        if all(c['veto_diagnostic'] is not None for c in chunk_reports['primary']):
            veto_diagnostic={}
            for group in ('total','H_positive','H_negative','S_positive','S_negative'):
                value=np.sum([c['veto_diagnostic'][group] for c in chunk_reports['primary']],axis=0)
                names=('events','input_weight','accepted_weight','expected_accepted_weight',
                       'conditional_bernoulli_variance','expected_rejected_events','realized_rejected_events')
                report=dict(zip(names,map(float,value)))
                report['events']=int(report['events']);report['realized_rejected_events']=int(report['realized_rejected_events'])
                report['accepted_minus_expected']=report['accepted_weight']-report['expected_accepted_weight']
                sigma=math.sqrt(report['conditional_bernoulli_variance'])
                report['conditional_veto_z']=report['accepted_minus_expected']/sigma if sigma else None
                veto_diagnostic[group]=report
        output[level] = dict(histograms=histograms, totals=stage_totals, veto_diagnostic=veto_diagnostic,
                             common_source_mode=common,signed_residuals=signed,
                             reference_relation='identical primary events and shower' if common else 'independent cohort',
                             continuation_audit=continuation_audit,
                             chunks=chunk_reports, complete=complete)
    if len(levels) == 2:
        for sample in ('primary', 'reference'):
            first, full = output['first']['chunks'][sample], output['full']['chunks'][sample]
            if len(first) != len(full) or any(
                    a['events'] != b['events'] or a['weight_sequence_sha256'] != b['weight_sequence_sha256']
                    or a['source_sequence_sha256'] != b['source_sequence_sha256']
                    for a, b in zip(first, full)):
                raise ValidationError('%s %s first/full event weights or source identities differ' % (process, sample))
    return specs, output


def save_histograms(directory, specs, result):
    directory.mkdir(parents=True, exist_ok=True)
    arrays, flows = {}, {}
    for name, histogram in result['histograms'].items():
        arrays[name+'__edges'] = histogram.edges
        for field in ('sumw', 'sumabsw', 'display_sumw', 'sumw2', 'count', 'covariance'):
            arrays[name+'__'+field] = getattr(histogram, field)
        flows[name] = {}
        with (directory/(name+'.csv')).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('stage', 'bin_low', 'bin_high', 'count', 'sumw_pb',
                             'sumw2_pb2', 'density', 'density_variance', 'display_sumw_pb', 'sumabsw_pb'))
            for i, stage in enumerate(STAGES):
                for j, width in enumerate(np.diff(histogram.edges)):
                    writer.writerow((stage, histogram.edges[j], histogram.edges[j+1],
                        int(histogram.count[i,j]), histogram.sumw[i,j], histogram.sumw2[i,j],
                        histogram.sumw[i,j]/width, histogram.sumw2[i,j]/width**2,
                        histogram.display_sumw[i,j],histogram.sumabsw[i,j]))
                flows[name][stage] = {flow: dict(count=int(value[i,0]), sumw=float(value[i,1]),
                    sumw2=float(value[i,2])) for flow, value in histogram.flows.items()}
        for numerator, denominator in ((stage, 'reference') for stage in PRIMARY[:4]):
            i, j = STAGES.index(numerator), STAGES.index(denominator)
            ratio, variance = ratio_with_variance(histogram.display_sumw[i], histogram.display_sumw[j],
                histogram.sumw2[i], histogram.sumw2[j], histogram.covariance[i,j])
            arrays[name+'__ratio_'+numerator+'_to_'+denominator] = ratio
            arrays[name+'__ratio_variance_'+numerator+'_to_'+denominator] = variance
        for numerator in ('h_only', 's_only', 'macnlops'):
            i, j = STAGES.index(numerator), STAGES.index('input')
            ratio, variance = ratio_with_variance(histogram.display_sumw[i], histogram.display_sumw[j],
                histogram.sumw2[i], histogram.sumw2[j], histogram.covariance[i,j])
            arrays[name+'__paired_ratio_'+numerator+'_to_input'] = ratio
            arrays[name+'__paired_ratio_variance_'+numerator+'_to_input'] = variance
    arrays['stages'] = np.asarray(STAGES)
    np.savez_compressed(directory/'histograms.npz', **arrays)
    (directory/'flows.json').write_text(json.dumps(flows, indent=2, sort_keys=True)+'\n')
    # Count fractions remain meaningful when signed event weights cancel.
    # Signed flow weights and their variances preserve the absolute normalization.
    with (directory/'range_flows.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('observable','stage','range_low','range_high','source_events',
            'visible_events','visible_fraction','visible_sumw_pb','visible_sumw2_pb2')+
            tuple(prefix+'_'+field for prefix in ('underflow','overflow','missing')
                  for field in ('events','sumw_pb','sumw2_pb2')))
        for name,histogram in result['histograms'].items():
            for i,stage in enumerate(STAGES):
                total=result['totals'][stage]['events']
                visible=int(np.sum(histogram.count[i]))
                row=(name,stage,histogram.edges[0],histogram.edges[-1],total,visible,
                    visible/total if total else '',float(np.sum(histogram.sumw[i])),
                    float(np.sum(histogram.sumw2[i])))
                row+=tuple(int(histogram.flows[flow][i,0]) if field==0 else float(histogram.flows[flow][i,field])
                           for flow in ('underflow','overflow','missing') for field in range(3))
                writer.writerow(row)
    return flows


def signed_curve(ax, edges, values, variances, color, label, logarithmic=False, band=True):
    """Show signed bins honestly; reflect only on an explicitly labelled log axis."""
    error = np.sqrt(np.maximum(0., variances))
    if not logarithmic:
        ax.stairs(values, edges, color=color, label=label, linewidth=1.35)
        if band:
            ax.fill_between(edges, np.r_[values-error,(values-error)[-1]],
                np.r_[values+error,(values+error)[-1]], step='post', color=color, alpha=.10)
        return
    positive, negative = values > 0., values < 0.
    if np.any(positive):
        ax.stairs(np.where(positive,values,np.nan), edges, color=color, label=label, linewidth=1.35)
    if np.any(negative):
        ax.stairs(np.where(negative,-values,np.nan), edges, color=color,
                  label=None if np.any(positive) else label, linestyle=':', linewidth=1.45)
    if band:
        absolute = np.abs(values)
        lower, upper = absolute-error, absolute+error
        valid = (values != 0.) & (lower > 32*np.finfo(float).eps*upper)
        lower, upper = np.where(valid,lower,np.nan), np.where(valid,upper,np.nan)
        ax.fill_between(edges, np.r_[lower,lower[-1]], np.r_[upper,upper[-1]],
                        step='post', color=color, alpha=.10)


def format_title(process, config, level, result, smoke):
    count = result['totals']['input']['events']
    label = config.get('label', PROCESS_NAMES[process])
    description = 'first QCD emission' if level == 'first' else 'full parton shower'
    prefix = 'SMOKE / INCOMPLETE — ' if smoke else ''
    return '%s%s; %s\n13 TeV, %s flavours; %s source events' % (
        prefix, label, description, config.get('flavours','?'), format(count, ','))


def autoscale_ratio_axis(axis):
    # Some Matplotlib versions insert zero into stairs' data limits even
    # with baseline=None. Recompute from the open paths, retaining every
    # uncertainty collection (Axes.relim alone excludes collections).
    axis.relim()
    for collection in axis.collections:
        bounds=collection.get_datalim(axis.transData)
        if np.all(np.isfinite(bounds.get_points())):
            axis.update_datalim(bounds)
    axis.autoscale_view()


def render_observable(directory, process, config, level, name, spec, histogram, result, smoke):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.labelsize':12, 'axes.titlesize':11,
                         'legend.fontsize':8.5, 'savefig.dpi':150, 'pdf.fonttype':42})
    figures = []
    labels=stage_labels(config)
    common=config.get('common_source_mode',False)
    widths, edges = np.diff(histogram.edges), histogram.edges
    display_sumw = histogram.display_sumw
    centers = (edges[:-1]+edges[1:])/2
    variants = ('linear', 'log') if spec.logarithmic else ('linear',)
    for variant in variants:
        log = variant == 'log'
        fig, (ax, ratio_ax) = plt.subplots(2,1,figsize=(8.3,7.1),sharex=True,
            gridspec_kw={'height_ratios':[3.3,1.2], 'hspace':.06})
        for stage in STAGES[:5]:
            i = STAGES.index(stage)
            signed_curve(ax, edges, display_sumw[i]/widths, histogram.sumw2[i]/widths**2,
                         COLORS[stage], labels[stage], log)
        if log:
            ax.set_yscale('log')
            ax.set_xscale('log')
            # Zero-event/recoil bins are retained in the data. A first
            # bin beginning at zero is displayed at its positive centre.
            ratio_ax.set_xlim(edges[0] if edges[0]>0 else centers[0], edges[-1])
            ax.text(.015,.015,'Dotted: negative bins reflected; bands crossing zero omitted',
                    transform=ax.transAxes,fontsize=7.7)
        else:
            ax.axhline(0.,color='.65',linewidth=.5)
        ax.set_ylabel(r'$d\sigma/dx$ [pb'+('/'+spec.unit if spec.unit else '')+']')
        ax.set_title(format_title(process,config,level,result,smoke))
        ax.legend(loc='best',frameon=False)
        ax.grid(alpha=.18)
        ref, refvar = display_sumw[0], histogram.sumw2[0]
        relative = np.full(ref.shape,np.nan)
        valid = ref != 0.
        relative[valid] = np.sqrt(refvar[valid])/np.abs(ref[valid])
        if not common:
            ratio_ax.fill_between(centers,1-relative,1+relative,color='.5',alpha=.14,
                                  label='Reference statistics')
        ratios = []
        for stage in STAGES[1:5]:
            i = STAGES.index(stage)
            ratio, variance = ratio_with_variance(display_sumw[i],ref,histogram.sumw2[i],refvar,
                                                  histogram.covariance[i,0])
            ratio_ax.stairs(ratio,edges,color=COLORS[stage],linewidth=1.15,baseline=None)
            ratios.append((stage,ratio,np.sqrt(variance)))
            if stage == 'macnlops':
                error = np.sqrt(variance)
                ratio_ax.fill_between(centers,ratio-error,ratio+error,color=COLORS[stage],alpha=.13)
        ratio_ax.axhline(1.,color='.4',linewidth=.7)
        autoscale_ratio_axis(ratio_ax)
        ratio_ax.set_ylabel('/ MC@NLO')
        ratio_ax.set_xlabel(spec.label+(' ['+spec.unit+']' if spec.unit else ''))
        ratio_ax.grid(alpha=.18)
        fig.text(.12,.015,('Shared ordinary input; ratio bands include paired event covariance' if common else
                          'Original event weights; statistical bands from sum of squared weights'),fontsize=8)
        fig.subplots_adjust(bottom=.12,top=.91,left=.13,right=.97)
        stem=name+'_'+variant
        for suffix in ('pdf','png'):
            fig.savefig(directory/(stem+'.'+suffix))
        figures.append(stem)
        # Preserve the unrestricted panel and supply a precision comparison.
        # Explicit edge markers identify central values outside the zoom.
        zoom_low,zoom_high=.5,1.5
        offscale,intervals=0,0
        for stage,ratio,error in ratios:
            below=np.isfinite(ratio)&(ratio<zoom_low)
            above=np.isfinite(ratio)&(ratio>zoom_high)
            offscale+=int(np.sum(below|above))
            intervals+=int(np.sum(np.isfinite(ratio)&
                ((ratio-error<zoom_low)|(ratio+error>zoom_high))))
            ratio_ax.plot(centers[below],np.full(np.sum(below),zoom_low),linestyle='none',
                          marker='v',markersize=4,color=COLORS[stage],clip_on=False)
            ratio_ax.plot(centers[above],np.full(np.sum(above),zoom_high),linestyle='none',
                          marker='^',markersize=4,color=COLORS[stage],clip_on=False)
        ratio_ax.set_ylim(zoom_low,zoom_high)
        ratio_ax.set_yticks([.5,1.,1.5])
        ratio_ax.text(.01,.96,'%d central ratios marked offscale; %d intervals extend beyond zoom' %
                      (offscale,intervals),transform=ratio_ax.transAxes,fontsize=6.8,va='top',
                      bbox=dict(facecolor='white',alpha=.75,edgecolor='none',pad=1.))
        stem += '_ratio_zoom'
        for suffix in ('pdf','png'):
            fig.savefig(directory/(stem+'.'+suffix))
        plt.close(fig)
        figures.append(stem)

    # Rejection stages share proposals; a shared ordinary baseline also
    # uses this covariance in the comparison panels above.
    fig, ax = plt.subplots(figsize=(8.3,4.6))
    for stage in ('h_only','s_only','macnlops'):
        i,j = STAGES.index(stage),STAGES.index('input')
        ratio,variance = ratio_with_variance(display_sumw[i],display_sumw[j],
            histogram.sumw2[i],histogram.sumw2[j],histogram.covariance[i,j])
        error=np.sqrt(variance)
        ax.stairs(ratio,edges,color=COLORS[stage],label=labels[stage],linewidth=1.35,baseline=None)
        ax.fill_between(centers,ratio-error,ratio+error,color=COLORS[stage],alpha=.12)
    ax.axhline(1.,color='.4',linewidth=.7)
    autoscale_ratio_axis(ax)
    ax.set_xlabel(spec.label+(' ['+spec.unit+']' if spec.unit else ''))
    ax.set_ylabel('/ unrejected source')
    ax.set_title(format_title(process,config,level,result,smoke))
    ax.legend(frameon=False);ax.grid(alpha=.18)
    fig.text(.12,.015,'Paired proposal covariance included; undefined zero-denominator bins masked',fontsize=8)
    fig.subplots_adjust(bottom=.17,top=.85,left=.13,right=.97)
    stem=name+'_mechanisms'
    for suffix in ('pdf','png'):
        fig.savefig(directory/(stem+'.'+suffix))
    plt.close(fig);figures.append(stem)

    if name in ('jet_pt','jet_pt_soft','system_pt','system_mass','pair_pt','pair_mass'):
        fig,ax=plt.subplots(figsize=(8.3,5.4))
        sign_stages=('input','H_total','H_positive','H_negative','S_positive','S_negative')
        if config.get('signed_residuals',config.get('allow_signed_residuals',False)):
            sign_stages+=('H_negative_retained',)
        for stage in sign_stages:
            i=STAGES.index(stage)
            signed_curve(ax,edges,display_sumw[i]/widths,histogram.sumw2[i]/widths**2,
                         COLORS[stage],labels[stage],logarithmic=True)
        ax.set_yscale('log')
        if name=='pair_mass':
            ax.set_xscale('log');ax.set_xlim(edges[0],edges[-1])
        ax.set_xlabel(spec.label+(' ['+spec.unit+']' if spec.unit else ''))
        ax.set_ylabel(r'$|d\sigma/dx|$ [pb'+('/'+spec.unit if spec.unit else '')+']')
        ax.set_title(format_title(process,config,level,result,smoke))
        ax.legend(frameon=False);ax.grid(alpha=.18)
        fig.text(.12,.025,'Dotted: negative contributions reflected; signed values remain in CSV/NPZ',fontsize=8)
        fig.subplots_adjust(bottom=.15,top=.86,left=.13,right=.97)
        stem=name+'_source_signs'
        for suffix in ('pdf','png'):
            fig.savefig(directory/(stem+'.'+suffix))
        plt.close(fig);figures.append(stem)
    return figures


def _process_task(task):
    """Analyze and render one process; only its output subdirectory is shared."""
    process,config=task['process'],task['configuration']
    output_dir=Path(task['output_dir'])
    print('Analyzing '+process,flush=True)
    specs,results=analyze_process(process,config,Path(task['base']),task['levels'],
        task['allow_incomplete'],task['batch_size'])
    record=dict(configuration=config,stage_labels=stage_labels(config),levels={})
    index,rows=[],[]
    for level,result in results.items():
        directory=output_dir/process/level
        flows=save_histograms(directory,specs,result)
        if any(result['continuation_audit'].values()):
            (directory/'continuation_audit.json').write_text(json.dumps(
                result['continuation_audit'],indent=2,sort_keys=True)+'\n')
        record['levels'][level]=dict(totals=result['totals'],chunks=result['chunks'],complete=result['complete'],
            common_source_mode=result['common_source_mode'],signed_residuals=result['signed_residuals'],
            reference_relation=result['reference_relation'],
            veto_diagnostic=result['veto_diagnostic'],continuation_audit=result['continuation_audit'],
            flows=flows,observables={name:dict(label=spec.label,unit=spec.unit,
                range=[float(spec.edges[0]),float(spec.edges[-1])],bins=len(spec.edges)-1)
                for name,spec in specs.items()})
        for stage,total in result['totals'].items():
            rows.append((process,level,stage,total['events'],total['negative_events'],
                         total['sumw'],total['sumw2'],total['effective_events']))
        if task['render']:
            print('Rendering %s / %s' % (process,level),flush=True)
            for name,spec in specs.items():
                for stem in render_observable(directory,process,config,level,name,spec,
                        result['histograms'][name],result,task['allow_incomplete']):
                    index.append((process,level,stem))
    return dict(process=process,record=record,rows=rows,index=index)


def _subprocess_task(path):
    """Subprocess isolation keeps Matplotlib/FastJet state private per process."""
    print('Starting isolated plot worker: '+path.stem,flush=True)
    environment=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[key]='1'
    completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-task',str(path)],
        env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if completed.returncode:
        raise ValidationError('Plot worker failed for '+str(path)+':\n'+
                              completed.stdout+'\n'+completed.stderr)
    print(completed.stdout,end='',flush=True)
    return json.loads(path.with_suffix('.result.json').read_text())


def _process_results(tasks,workers,output_dir):
    if workers==1:
        yield from map(_process_task,tasks)
        return
    with tempfile.TemporaryDirectory(prefix='.process-tasks.',dir=output_dir) as temporary:
        paths=[]
        for index,task in enumerate(tasks):
            path=Path(temporary)/('%02d_%s.json' % (index,task['process']))
            path.write_text(json.dumps(task)+'\n');paths.append(path)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # map returns results in registry order, independently of finish order.
            yield from executor.map(_subprocess_task,paths)


def _run_analysis(datasets, output_dir, levels=('first','full'), allow_incomplete=False,
                  batch_size=2048, render=True,workers=1):
    datasets,output_dir=Path(datasets),Path(output_dir)
    configuration=json.loads(datasets.read_text())
    processes=configuration.get('processes',{})
    if not allow_incomplete and set(processes) != set(PROCESS_OBJECTS):
        raise ValidationError('Production validation requires dy, zz, ttbar and wbb')
    if not allow_incomplete and set(levels) != {'first','full'}:
        raise ValidationError('Production validation requires first and full parton samples')
    if len(set(levels)) != len(levels) or not set(levels) <= {'first','full'}:
        raise ValidationError('Invalid analysis levels')
    output_dir.mkdir(parents=True,exist_ok=True)
    summary=dict(schema='macnlops-validation-histograms-v1', datasets=str(datasets.resolve()),
        datasets_sha256=hashlib.sha256(datasets.read_bytes()).hexdigest(),
        smoke=bool(allow_incomplete),levels=list(levels),normalization='unaltered original sum weights, pb',
        variance=(SHARED_VARIANCE if processes and all(c.get('common_source_mode',False) for c in processes.values())
                  else INDEPENDENT_VARIANCE if not any(c.get('common_source_mode',False) for c in processes.values())
                  else 'sumw2; reference covariance follows each process common_source_mode'),
        integration_uncertainty='native hard-integration errors are separate; plotted bands use event-level sumw2 only',
        veto_diagnostic='conditional expected sum(w*p) and sum(w^2*p*(1-p)); not added to histogram sumw2',
        numerical_zero='for display and ratios only, |sumw| <= 8*machine_epsilon*sumabsw; raw sums retained',
        jet_definition='anti-kT R=0.4, stable partons |PDG|<=5 or gluon; tops excluded',
        jet_y_selection='leading jet pT >20 GeV after clustering; softer/no-jet entries in missing flow; no event cut',
        wbb_radiation_jets='remove stable b/bbar IDs before clustering; all-parton jets also included',
        wbb_pair_observables='m_bb, pT_bb and y_bb, alongside complete Wbb system; pair angles use b/bbar',
        soft_jet_binning='same leading radiation jet: 0 to 50 GeV in 0.5 GeV bins; broad 0 to 600 GeV spectrum retained',
        software={name:importlib.metadata.version(name) for name in ('numpy','matplotlib','awkward','vector','fastjet')},
        stages=LABELS,processes={})
    summary['plan']=configuration.get('plan',{})
    summary['plan']['jet_y_selection']=summary['jet_y_selection']
    if 'wbb' in processes:summary['plan']['wbb_pair_observables']=summary['wbb_pair_observables']
    summary['notes']=configuration.get('notes',[])
    summary['common_source_mode']=bool(processes) and all(c.get('common_source_mode',False) for c in processes.values())
    if summary['common_source_mode']:
        summary['stages']=stage_labels(next(iter(processes.values())))
    summary['provenance']=[]
    summary['native_source_integrals']=None
    for i,entry in enumerate(configuration.get('provenance',[]),1):
        source=resolve(entry['path'],datasets.parent)
        target=output_dir/'provenance'/('%03d_' % i+source.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        summary['provenance'].append(dict(label=entry.get('label',source.name),
            source=str(source.resolve()),path=str(target.relative_to(output_dir)),
            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        if source.name=='source_integrals.json':
            integral=json.loads(target.read_text())
            if integral.get('schema')!='macnlops-source-integration-estimates-v1':
                raise ValidationError('Unknown source integration metadata schema')
            summary['native_source_integrals']=integral
    tasks=[dict(process=process,configuration=config,base=str(datasets.parent.resolve()),
        output_dir=str(output_dir.resolve()),levels=levels,allow_incomplete=allow_incomplete,
        batch_size=batch_size,render=render) for process,config in processes.items()]
    index=[]
    with (output_dir/'stage_totals.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(('process','level','stage','events','negative_events','sumw_pb','sumw2_pb2','effective_events'))
        for result in _process_results(tasks,workers,output_dir):
            summary['processes'][result['process']]=result['record']
            writer.writerows(result['rows'])
            index.extend(result['index'])
    summary['complete']=not allow_incomplete and all(
        level['complete'] for process in summary['processes'].values() for level in process['levels'].values())
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    title='SMOKE / INCOMPLETE validation' if allow_incomplete else 'MAcNLOPS validation'
    cards=['<!doctype html><meta charset="utf-8"><title>'+title+'</title><h1>'+title+'</h1>',
           '<p>Original signed weights; no survivor renormalization. '
           'Dotted log curves show reflected negative bins. '
           '<a href="summary.json">Summary and flow counts</a>; '
           '<a href="stage_totals.csv">stage totals CSV</a>.</p>']
    cards.append('<p>Statistical bands use event-level sums of squared weights. Native hard-integration '
                 'errors quoted in the run plan are separate normalization information. '
                 'Shared-input comparisons include the event covariance with the baseline; their pre-veto '
                 'ratio is exactly one. Independent-cohort comparisons have zero reference covariance.</p>')
    if summary['common_source_mode']:
        cards.append('<p>Each process has one ordinary MC@NLO source and one first/full shower cohort. '
                     'The reference and input curves are identical before rejection. Negative H events '
                     'may remain where the declared signed-residual policy requires them; the H-only '
                     'curve follows the actual H decisions. No survivor rescaling is applied.</p>')
    cards.append('<p>Each distribution has an unrestricted ratio panel and a 0.5–1.5 zoom companion. '
                 'Triangles mark offscale central values; raw signed ratios and all variances remain in NPZ. '
                 'Full-shower panels use the recorded continuation policy; consult the run audits for '
                 'late native envelope clips. The source matching construction has the documented '
                 'physical cutoff and heavy-threshold limitations.</p>')
    if summary['plan']:
        cards.append('<h2>Physics and run settings</h2><pre>'+html.escape(
            json.dumps(summary['plan'],indent=2,sort_keys=True))+'</pre>')
    for note in summary['notes']:
        cards.append('<p>'+html.escape(str(note))+'</p>')
    if summary['provenance']:
        cards.append('<h2>Recorded settings and audits</h2><ul>')
        for report in summary['provenance']:
            cards.append('<li><a href="'+html.escape(report['path'])+'">'+
                html.escape(report['label'])+'</a> — SHA256 '+report['sha256']+'</li>')
        cards.append('</ul>')
    cards.append('<h2>Sample accounting</h2><table border="1" cellspacing="0" cellpadding="5">'
        '<tr><th>Process / level</th><th>Source / reference events</th><th>Retained events</th>'
        '<th>Reference sum w [pb]</th><th>Source sum w [pb]</th><th>Retained sum w [pb]</th>'
        '<th>Source / retained N_eff</th><th>Negative H / S retained</th></tr>')
    for process,record in summary['processes'].items():
        for level,record in record['levels'].items():
            t=record['totals'];source,reference,kept=t['input'],t['reference'],t['macnlops']
            cards.append('<tr><td>'+html.escape(process+' / '+level)+'</td><td>%s / %s</td>'
                '<td>%s</td><td>%.7g</td><td>%.7g</td><td>%.7g</td><td>%.1f / %.1f</td>'
                '<td>%s / %s</td></tr>' % (format(source['events'],','),format(reference['events'],','),
                format(kept['events'],','),reference['sumw'],source['sumw'],kept['sumw'],
                source['effective_events'],kept['effective_events'],format(t['H_negative_retained']['events'],','),
                format(kept['negative_events']-t['H_negative_retained']['events'],',')))
    cards.append('</table><p>Conditional veto expectations and variances are in summary.json; '
        'they diagnose rejection randomness and are not added again to histogram sumw2.</p>')
    if summary['native_source_integrals'] is not None:
        cards.append('<h2>Separate native hard-integration estimates</h2>'
            '<p>These estimates describe each full source run. In a preview, their event count differs '
            'from the analyzed chunk count. They do not rescale any curve or enter its event-level band.</p>'
            '<table border="1" cellspacing="0" cellpadding="5"><tr><th>Process / role / source ID</th>'
            '<th>Requested source events</th><th>Native estimate +/- integration error [pb]</th></tr>')
        with (output_dir/'source_integrals.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(('process','role','source_process_id','source_events_requested',
                             'native_cross_section_pb','native_integration_error_pb'))
            for process,record in summary['native_source_integrals'].get('processes',{}).items():
                if process not in summary['processes']:continue
                for role,source in record.items():
                    for estimate in source.get('declared_init',[]):
                        row=(process,role,estimate['process_id'],source['source_events_requested'],
                             estimate['cross_section_pb'],estimate['integration_error_pb'])
                        writer.writerow(row)
                        cards.append('<tr><td>'+html.escape('%s / %s / %s' % row[:3])+\
                            '</td><td>%s</td><td>%.7g +/- %.4g</td></tr>' %
                            (format(row[3],','),row[4],row[5]))
        cards.append('</table>')
    audit_links=[]
    for process,record in summary['processes'].items():
        audit=record['levels'].get('full',{}).get('continuation_audit',{})
        for role,counts in audit.items():
            if counts is not None:
                audit_links.append('<li>'+html.escape(process+' / '+role)+': %s events; %s strict S endpoints; '
                    '%s events with later native trial warnings. <a href="%s/full/continuation_audit.json">'
                    'Aggregated diagnostics and raw-file hashes</a>.</li>' %
                    (format(counts['events'],','),format(counts['strict_s_events'],','),
                     format(counts['events_with_warnings'],','),process))
    if audit_links:cards+=['<h2>Full continuation audits</h2><ul>']+audit_links+['</ul>']
    groups=list(dict.fromkeys((process,level) for process,level,stem in index))
    if groups:
        cards.append('<nav id="contents"><h2>Figures</h2><ul>')
        for process,level in groups:
            cards.append('<li><a href="#'+process+'-'+level+'">'+html.escape(process+' / '+level)+'</a></li>')
        cards.append('</ul></nav>')
    previous=None
    for process,level,stem in index:
        if (process,level)!=previous:
            if previous is not None:cards.append('</details>')
            cards.append('<h2 id="'+process+'-'+level+'">'+html.escape(process+' / '+level)+
                '</h2><p><a href="#contents">Back to contents</a></p>'
                '<p><a href="'+process+'/'+level+'/range_flows.csv">All range flows CSV</a>; '
                '<a href="'+process+'/'+level+'/flows.json">raw flows JSON</a>.</p>')
            cards.append('<details><summary>Histogram range accounting (counts; signed weights in pb)</summary>'
                '<p>Percentages use event counts, preserving their meaning when signed weights cancel. '
                'Missing includes the explicit jet-rapidity selection. Every source event remains in sample totals.</p>'
                '<table border="1" cellspacing="0" cellpadding="4"><tr><th>Observable / stage</th>'
                '<th>Range</th><th>Visible events</th><th>Underflow n / sum w</th>'
                '<th>Overflow n / sum w</th><th>Missing n / sum w</th></tr>')
            level_record=summary['processes'][process]['levels'][level]
            for observable,flow_record in level_record['flows'].items():
                bounds=level_record['observables'][observable]['range']
                for stage in ('reference','input','macnlops'):
                    total=level_record['totals'][stage]['events']
                    if not total:continue
                    flow=flow_record[stage]
                    visible=total-sum(flow[key]['count'] for key in ('underflow','overflow','missing'))
                    cards.append('<tr><td>'+html.escape(observable+' / '+stage)+'</td>'
                        '<td>%.5g–%.5g</td><td>%s (%.2f%%)</td>' %
                        (bounds[0],bounds[-1],format(visible,','),100.*visible/total)+
                        ''.join('<td>%s / %.6g</td>' % (format(flow[key]['count'],','),flow[key]['sumw'])
                                for key in ('underflow','overflow','missing'))+'</tr>')
            cards.append('</table></details><details open><summary>Show figures; click an image for its vector PDF</summary>')
            previous=(process,level)
        base=html.escape(process+'/'+level+'/'+stem)
        cards.append('<figure><a href="'+base+'.pdf"><img loading="lazy" width="620" src="'+base+'.png"></a>'
                     '<figcaption>'+html.escape(process+' / '+level+' / '+stem)+'</figcaption></figure>')
    if previous is not None:cards.append('</details>')
    (output_dir/'index.html').write_text('\n'.join(cards)+'\n')
    return summary


def run_analysis(datasets, output_dir, levels=('first','full'), allow_incomplete=False,
                 batch_size=2048, render=True,workers=1):
    """Publish a complete analysis directory only after every check passes."""
    output_dir=Path(output_dir).resolve()
    if type(workers) is not int or not 1<=workers<=4:
        raise ValidationError('Plot workers must be an integer from 1 to 4')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValidationError('Output directory is not empty: '+str(output_dir))
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output_dir.name+'.partial.',dir=output_dir.parent) as temporary:
        summary=_run_analysis(datasets,Path(temporary),levels,allow_incomplete,batch_size,render,workers)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(temporary,output_dir)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--levels',nargs='+',choices=('first','full'),default=['first','full'])
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--workers',type=int,choices=range(1,5),default=1,
                        help='independent process analyses/renders (default: serial)')
    parser.add_argument('--worker-task',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker_task is not None:
        result=_process_task(json.loads(args.worker_task.read_text()))
        args.worker_task.with_suffix('.result.json').write_text(json.dumps(result)+'\n')
        return
    if args.datasets is None or args.output_dir is None:
        parser.error('--datasets and --output-dir are required')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    summary=run_analysis(args.datasets,args.output_dir,tuple(args.levels),args.allow_incomplete,
                         args.batch_size,workers=args.workers)
    print(json.dumps(dict(complete=summary['complete'],smoke=summary['smoke'],
                         output=str(args.output_dir.resolve())),sort_keys=True))


if __name__ == '__main__':
    main()
