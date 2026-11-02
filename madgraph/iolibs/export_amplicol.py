################################################################################
#
# Copyright (c) 2026 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
"""mg7 output of the processes computed by AmpliCol ('[amplicol]').

pyamplicol generates the colour-ordered matrix elements of a process definition
in SubProcesses/A<n>_<name>: a leading-colour artifact (lc), a full-colour one
(full), and the sources of the UMAMI library returning their ratio (reweight).
Its makefile builds the libraries into lib/. Every UMAMI channel of every
provider becomes one entry of subprocesses.json: the colour orderings sampled
by one colour-ordered phase-space channel, each weighted by its density over
those of its partners (where the colour singlets attach, the s-channel
resonances followed), which sample copies of it in their own subprocesses.
Subprocesses related by a flavour
relabelling that is a symmetry of the allowed vertices are generated once
(pyamplicol's strict flavour-blind grouping) and evaluated through their
representative.
"""

import hashlib
import json
import logging
import os
import re
import shlex
import sys

import madgraph

logger = logging.getLogger('madgraph.export_amplicol')
pjoin = os.path.join

MODEL_FILE = 'amplicol_model.pyamplicol-model'
# the pyamplicol that generated the output, checked before compiling
PYAMPLICOL_RECORD = 'amplicol_pyamplicol.json'

MAKEFILE = """\
# Builds the AmpliCol matrix elements of this process into ../../lib.
RUSTICOL_CONFIG ?= $(CURDIR)/../rusticol-config
BUILD := $(CURDIR)/build

all:
\t$(CURDIR)/../check-pyamplicol
\t$(MAKE) -C lc/API/umami shared BUILD_DIR=$(BUILD)/lc RUSTICOL_CONFIG=$(RUSTICOL_CONFIG)
\t$(MAKE) -C reweight shared BUILD_DIR=$(BUILD)/reweight RUSTICOL_CONFIG=$(RUSTICOL_CONFIG)
%(copies)s
clean:
\trm -rf $(BUILD)
"""


# rusticol-config of the pyamplicol in pyamplicol/install, for the makefiles
RUSTICOL_CONFIG = """\
#!/bin/sh
PYTHONPATH=%(install)s${PYTHONPATH:+:$PYTHONPATH} exec %(python)s -m rusticol_config "$@"
"""

# fails unless that pyamplicol is the one recorded in PYAMPLICOL_RECORD
CHECK_PYAMPLICOL = """\
#!/bin/sh
PYTHONPATH=%(install)s${PYTHONPATH:+:$PYTHONPATH} exec %(python)s -c \\
  'import sys; sys.path.insert(0, sys.argv[1]); from madgraph.mg7.bootstrap import check_pyamplicol_build; check_pyamplicol_build(sys.argv[2])' \\
  %(mg_root)s "$(dirname "$0")/%(record)s"
"""


def check_pyamplicol(interactive=None, jobs=None):
    """Install pyamplicol in pyamplicol/install if needed and put it first on
    sys.path, for the generation in this process and for the builds.
    Returns its version and source hash."""
    from madgraph.mg7 import bootstrap
    try:
        bootstrap.ensure_pyamplicol(interactive=interactive, jobs=jobs)
        return bootstrap.check_pyamplicol_source_hash()
    except RuntimeError as error:
        raise madgraph.InvalidCmd('The [amplicol] processes need pyamplicol: %s' % error)


def write_rusticol_config(subproc_dir, pyamplicol):
    """SubProcesses/rusticol-config, the RUSTICOL_CONFIG of the makefiles, and
    check-pyamplicol with the pyamplicol it has to be."""
    from madgraph.mg7 import bootstrap
    values = {'install': shlex.quote(str(bootstrap.PYAMPLICOL_INSTALL_DIR)),
              'python': shlex.quote(sys.executable),
              'mg_root': shlex.quote(str(bootstrap.MG_ROOT)),
              'record': PYAMPLICOL_RECORD}
    for name, template in (('rusticol-config', RUSTICOL_CONFIG),
                           ('check-pyamplicol', CHECK_PYAMPLICOL)):
        path = pjoin(subproc_dir, name)
        with open(path, 'w') as stream:
            stream.write(template % values)
        os.chmod(path, 0o755)
    with open(pjoin(subproc_dir, PYAMPLICOL_RECORD), 'w') as stream:
        json.dump(pyamplicol, stream, indent=2)


def load_metadata(artifact):
    """The UMAMI metadata documents of an artifact, one per provider."""
    root = pjoin(artifact, 'API', 'umami')
    with open(pjoin(root, 'metadata.json')) as stream:
        data = json.load(stream)
    if data.get('kind') != 'pyamplicol-umami-provider-index':
        return [data]
    documents = []
    for provider in data['providers']:
        with open(pjoin(root, provider['metadata'])) as stream:
            documents.append(json.load(stream))
    return documents


def leg_pdgs(leg, model):
    """The pdg ids of a leg: the flavours MG merged into a group id, or the ids
    with the group ids expanded."""
    if leg.get('flavor'):
        return list(leg.get('flavor'))
    merged = model.get('merged_particles')
    pdgs = []
    for pid in leg.get('ids'):
        members = merged.get(abs(pid), [abs(pid)])
        pdgs.extend(member if pid > 0 else -member for member in members)
    return pdgs


def leg_sets(process, model):
    """(initial, final): the sets of pdg ids of the legs of a process definition
    or of a process, which has one id per leg."""
    initial, final = [], []
    for leg in process.get('legs'):
        if 'ids' in leg or leg.get('flavor'):
            pdgs = set(leg_pdgs(leg, model))
        else:
            # one id, which may stand for the flavours MG merged into it
            pid = leg.get('id')
            members = model.get('merged_particles').get(abs(pid), [abs(pid)])
            pdgs = {member if pid > 0 else -member for member in members}
        (final if leg.get('state') else initial).append(pdgs)
    return initial, final


def overlapping(first, second, model):
    """Whether two process definitions (or processes) have a process in common:
    the initial legs in order, the final ones in any order."""
    initial1, final1 = leg_sets(first, model)
    initial2, final2 = leg_sets(second, model)
    if len(initial1) != len(initial2) or len(final1) != len(final2) or \
            any(not (one & two) for one, two in zip(initial1, initial2)):
        return False

    def match(index, free):
        if index == len(final1):
            return True
        return any(final1[index] & final2[other] and
                   match(index + 1, free - {other}) for other in free)

    return match(0, frozenset(range(len(final2))))


def process_name(procdef, model, multiparticles):
    """A directory name for a process definition, e.g. 'pp_jjj', with the
    multiparticle labels of the interface."""
    particles = model.get('particle_dict')
    labels = []
    for leg in procdef.get('legs'):
        pdgs = leg_pdgs(leg, model)
        matches = [label for label, ids in multiparticles.items()
                   if sorted(ids) == sorted(pdgs)]
        preferred = 'j' if leg.get('state') else 'p'
        label = preferred if preferred in matches else next(iter(matches), None)
        if label is None:
            label = ''.join(particles[pid].get_name() for pid in pdgs)
        labels.append((leg.get('state'), label))
    name = ''.join(l for state, l in labels if not state) + '_' + \
        ''.join(l for state, l in labels if state)
    return re.sub(r'[^A-Za-z0-9_]', '',
                  name.replace('~', 'x').replace('+', 'p').replace('-', 'm'))


def process_expression(procdef, model):
    """The process in pyamplicol syntax, and its multiparticle definitions."""
    particles = model.get('particle_dict')
    multiparticles = {}
    labels = []
    for leg in procdef.get('legs'):
        names = [particles[pid].get_name() for pid in leg_pdgs(leg, model)]
        if len(names) == 1:
            labels.append(names[0])
            continue
        label = next((key for key, value in multiparticles.items()
                      if value == names), None)
        if label is None:
            label = 'mp%d' % len(multiparticles)
            multiparticles[label] = names
        labels.append(label)
    initial = [label for leg, label in zip(procdef.get('legs'), labels)
               if not leg.get('state')]
    final = [label for leg, label in zip(procdef.get('legs'), labels)
             if leg.get('state')]
    return '%s > %s' % (' '.join(initial), ' '.join(final)), multiparticles


def coupling_orders(procdef, model):
    """The largest orders of the couplings above the lowest hierarchy, as MG
    selects them by default: the minimal WEIGHTED order of the definition,
    with n - 2 vertices at tree level."""
    import madgraph.core.diagram_generation as diagram_generation
    weighted = diagram_generation.MultiProcess.find_optimal_process_orders(
        procdef)['WEIGHTED']
    # a hierarchy 0 is a hidden order (INTERF, from the [treextree] processes),
    # not a coupling: as in find_optimal_process_orders, it does not count
    hierarchy = {order: weight for order, weight
                 in model.get('order_hierarchy').items() if weight}
    lowest = min(hierarchy.values())
    vertices = len(procdef.get('legs')) - 2
    return {order: max((weighted - vertices * lowest) // (weight - lowest), 0)
            for order, weight in hierarchy.items() if weight > lowest}


def model_job(model, cache_path, pyamplicol=None):
    """The UFO model of the output with its restriction, compiled for
    pyamplicol once per output and again when the model, its restriction or
    pyamplicol change: the cache is keyed by a sidecar json file."""
    path = model.get('modelpath')
    reference = model.get('name')
    # the suffix after the UFO directory name, which may itself contain '-'
    suffix = model.get('restrict_name')
    if not suffix:
        restriction = 'default'
    elif suffix == 'full':
        restriction = 'none'
    else:
        restriction = pjoin(path, 'restrict_%s.dat' % suffix)
    sources = {'model': os.path.realpath(path), 'restriction': restriction,
               'restriction_card': file_hash(pjoin(path, 'restrict_default.dat')
                                             if restriction == 'default' else restriction),
               'pyamplicol': pyamplicol}
    key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    key_file = os.path.splitext(cache_path)[0] + '.json'
    if not (os.path.exists(cache_path) and cached_model_key(key_file) == key):
        logger.info('Compiling the model %s for AmpliCol' % reference)
    return {'path': path, 'restriction': restriction, 'cache': cache_path,
            'key': key, 'key_file': key_file, 'sources': sources}


def file_hash(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def cached_model_key(key_file):
    """The key of the cached model, see amplicol_generation.model_source."""
    try:
        with open(key_file) as stream:
            return json.load(stream).get('key')
    except (OSError, ValueError, AttributeError):
        return None


def representative_processes(proc_defs, model):
    """Processes standing for the [amplicol] definitions where the output needs
    MG processes (run card defaults, proc_characteristics): every initial state
    with the first particle of each final-state leg, and every particle of each
    final-state leg once, so that the default histograms see the whole group
    (a jet, not a gluon)."""
    import itertools
    import madgraph.core.base_objects as base_objects
    processes = []
    for procdef in proc_defs:
        legs = procdef.get('legs')
        initial = [leg_pdgs(leg, model) for leg in legs if not leg.get('state')]
        finals = [leg_pdgs(leg, model) for leg in legs if leg.get('state')]
        first = [pdgs[0] for pdgs in finals]
        states = [list(ids) + first for ids in itertools.product(*initial)]
        states += [[pdgs[0] for pdgs in initial] + first[:i] + [pdg] + first[i + 1:]
                   for i, options in enumerate(finals) for pdg in options[1:]]
        for pdgs in states:
            ids = pdgs[:len(initial)]
            processes.append(base_objects.Process({
                'legs': base_objects.LegList([
                    base_objects.Leg({'id': pdg, 'number': number + 1,
                                      'state': number >= len(ids)})
                    for number, pdg in enumerate(pdgs)]),
                'model': model,
                'id': procdef.get('id'),
            }))
    return processes


def color_types(model, pdgs):
    types = {}
    for pdg in pdgs:
        particle = model.get_particle(pdg)
        types[pdg] = particle.get_color()
        types[-pdg] = model.get_particle(particle.get_anti_pdg_code()).get_color()
    return types


def mirrored(pdgs, incoming, initial_legs, exported):
    """Whether the process with swapped beams is requested but not exported:
    pyamplicol keeps one process per unordered initial state."""
    if incoming != 2 or pdgs[0] == pdgs[1]:
        return False
    swapped = ((pdgs[1], pdgs[0]), tuple(sorted(pdgs[2:])))
    return (pdgs[1] in initial_legs[0] and pdgs[0] in initial_legs[1]
            and swapped not in exported)


def resonance_pdg(model, pdg, decay_pdgs):
    """The resonance of the given species carrying the charge of its outgoing
    decay products: W+ or W-, say, for the rows of one selector."""
    particle = model.get_particle(abs(pdg))
    if particle.get('self_antipart'):
        return abs(pdg)
    charge = sum(model.get_particle(p).get_charge() for p in decay_pdgs)
    return abs(pdg) if abs(particle.get_charge() - charge) < 1e-6 else -abs(pdg)


def subprocess_entries(metadata, model, rel_dir, me_path, ratio_path, qcd_power,
                       initial_legs):
    """One subprocesses.json entry per channel of a UMAMI provider; initial_legs
    are the pdg ids allowed for each initial leg."""
    pdg_ids = metadata['pdg_ids']
    provider = metadata['provider']
    incoming = provider['incoming_count']
    exported = {(tuple(pdgs[:incoming]), tuple(sorted(pdgs[incoming:])))
                for pdgs in pdg_ids}
    entries = []
    for channel_index, channel in enumerate(metadata['channels']):
        phasespace = channel['phasespace_channels']
        flavors = []
        for flavor_index, process in enumerate(channel['processes']):
            # the rows of a selector share its evaluation and incoming legs: one
            # flavour per mirror flag and resonance charges, its options written
            # in proportion to factor
            by_kind = {}
            for term in process['matrix_elements']:
                pdgs = pdg_ids[term['pdg_ids']]
                mirror = mirrored(pdgs, incoming, initial_legs, exported)
                resonances = [
                    [resonance_pdg(model, r['pdg'], [pdgs[leg] for leg in r['legs']])
                     for r in phasespace[index]['resonances']]
                    for index in process['multichannels']]
                flavor = by_kind.setdefault((mirror, str(resonances)), {
                    'index': flavor_index, 'options': [], 'weights': [],
                    'factor': 0, 'mirror': mirror,
                    'color_index': process['color_flows'] or 0,
                    'channels': process['multichannels'],
                    'resonance_pdgs': resonances})
                flavor['options'].append(pdgs)
                flavor['weights'].append(term['factor'])
                flavor['factor'] += term['factor']
            flavors.extend(by_kind.values())
        first = flavors[0]['options'][0]
        entries.append({
            'backend': 'amplicol',
            'incoming': first[:incoming],
            'outgoing': first[incoming:],
            'channels': [],
            'path': rel_dir,
            'me_path': me_path,
            'ratio_path': ratio_path,
            'artifact': pjoin(rel_dir, 'lc'),
            'ratio_artifact': pjoin(rel_dir, 'reweight'),
            'umami_channel': channel_index,
            'phasespace_channels': [
                {'order': ps['phasespace_order'], 'resonances': ps['resonances']}
                for ps in phasespace],
            'colored_resonances': channel.get('colored_resonances', []),
            'masses': provider['masses'],
            'supports_alpha_s': provider['supports_alpha_s'],
            'flavors': flavors,
            'qcd_power': qcd_power,
            'color_flows': metadata['color_flows'],
            'pdg_color_types': color_types(
                model, {abs(pdg) for pdgs in pdg_ids for pdg in pdgs}),
            'helicities': metadata['helicities'],
            'diagram_count': 0,
            'interference': False,
            'parameters': parameter_references(metadata),
        })
    return entries


def parameter_references(metadata):
    """The parameter card entry of every mutable runtime parameter."""
    references = {}
    for process in metadata['runtime_processes']:
        for parameter in process['parameters']:
            if parameter.get('mutable') and parameter.get('lhablock'):
                references[parameter['name']] = [parameter['lhablock'],
                                                 parameter['lhacode']]
    return references


def export_amplicol_processes(*args, **kwargs):
    """Write SubProcesses/A<n>_<name> for every '[amplicol]' process definition
    and return the subprocesses.json entries; see _export_amplicol_processes.

    pyamplicol/install, which holds pinned dependencies (numpy, ...) too, is on
    sys.path only while this runs, so that it does not shadow the packages of the
    rest of the MG5 session (what pyamplicol imports stays loaded)."""
    saved = list(sys.path)
    try:
        return _export_amplicol_processes(*args, **kwargs)
    finally:
        sys.path[:] = saved


def _export_amplicol_processes(dir_path, proc_defs, model, multiparticles=None,
                               interactive=None, jobs=None):
    """The body of export_amplicol_processes. ``interactive`` and ``jobs`` are
    those of the pyamplicol installation, see bootstrap.ensure_pyamplicol."""
    pyamplicol = check_pyamplicol(interactive, jobs)
    subproc_dir = pjoin(dir_path, 'SubProcesses')
    write_rusticol_config(subproc_dir, pyamplicol)
    processes = []
    for number, procdef in enumerate(proc_defs, start=1):
        name = 'A%d_%s' % (number, process_name(procdef, model, multiparticles or {}))
        path = pjoin(subproc_dir, name)
        expression, process_multiparticles = process_expression(procdef, model)
        os.makedirs(path, exist_ok=True)
        logger.info('Generating the AmpliCol matrix elements of %s in %s' %
                    (procdef.nice_string(prefix=False), path))
        processes.append({'name': name, 'path': path, 'expression': expression,
                          'multiparticles': process_multiparticles,
                          'orders': coupling_orders(procdef, model)})
    from madgraph.iolibs import amplicol_generation
    try:
        providers = amplicol_generation.generate({
            'model': model_job(model, pjoin(subproc_dir, MODEL_FILE), pyamplicol),
            'processes': processes})
    except Exception as error:
        logger.error('pyamplicol failed to generate the [amplicol] matrix elements')
        raise madgraph.InvalidCmd('pyamplicol failed to generate the [amplicol] '
                                  'matrix elements: %s' % error)
    entries = []
    for procdef, process, provider_ids in zip(proc_defs, processes, providers):
        name, path, orders = process['name'], process['path'], process['orders']
        rel_dir = pjoin('SubProcesses', name)
        # every vertex carries one coupling: the rest are powers of g_s
        qcd_power = (len(procdef.get('legs')) - 2 - sum(orders.values())
                     if len(orders) == 1 else -1)
        initial_legs = [set(leg_pdgs(leg, model))
                        for leg in procdef.get('legs') if not leg.get('state')]
        copies = []
        for provider_id, metadata in zip(provider_ids, load_metadata(pjoin(path, 'lc'))):
            library = metadata['provider']['library']
            me_path = pjoin('lib', 'libamplicol_%s_%s.so' % (name, provider_id))
            ratio_path = pjoin('lib', 'libamplicol_%s_%s_ratio.so' %
                               (name, provider_id))
            copies.append('\tcp $(BUILD)/lc/%s ../../%s' % (library, me_path))
            copies.append('\tcp $(BUILD)/reweight/libreweight_%s.so ../../%s' %
                          (provider_id, ratio_path))
            entries.extend(subprocess_entries(metadata, model, rel_dir, me_path,
                                              ratio_path, qcd_power, initial_legs))
        with open(pjoin(path, 'makefile'), 'w') as stream:
            stream.write(MAKEFILE % {'copies': '\n'.join(copies)})
        with open(pjoin(path, 'process.json'), 'w') as stream:
            json.dump({'process': procdef.nice_string(prefix=False),
                       'expression': process['expression'],
                       'multiparticles': process['multiparticles'],
                       'max_coupling_orders': orders}, stream, indent=2)
    return entries
