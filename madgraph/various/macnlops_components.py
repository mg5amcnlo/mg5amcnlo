"""Certified first-branching labels for sampled-component MAcNLOPS.

This module only identifies discrete routes.  It never samples a route,
evaluates a matrix element, assigns an FKS S-function as a probability, or
claims that a native inverse map equals the shower map.  The caller must
certify that map and use the physical component kernels when marking
multiple returned daughter orientations.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
import math
from pathlib import PurePath


class MappingError(ValueError):
    """The recorded branching does not certify a unique source ancestry."""


@dataclass(frozen=True)
class BranchComponent:
    """One native route; all permutations and endpoint positions are zero based.

    ``permutation`` gives endpoint external legs in native real order.
    ``born_permutation`` gives source Born legs in native Born order.  The
    latter can exchange identical Born particles.  Such exchanges require
    ``transport_component_basis`` before passing flow and diagram indices
    to the evaluator; ordinals cannot simply be copied.
    """

    sector: int
    context: int
    provider: int
    permutation: tuple
    born_permutation: tuple
    emitted: int
    radiator: int
    orientation: str
    requires_marking: bool = False


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise MappingError('Invalid integer for %s' % name)
    return value


def _pid(particle):
    return _integer(particle['id'], 'particle identity')


def _momentum(particle):
    values = particle.get('p', ())
    if len(values) != 4 or not all(math.isfinite(x) for x in values):
        raise MappingError('Missing or nonfinite particle four-momentum')
    return values


def _same_momentum(first, second):
    a, b = _momentum(first), _momentum(second)
    tolerance = 5.e-10 * max(1., abs(a[0]), abs(b[0]))
    return all(math.isclose(x, y, rel_tol=5.e-10, abs_tol=tolerance)
               for x, y in zip(a, b))


def _colours(particle):
    values = particle.get('colors', ())
    if len(values) != 2:
        raise MappingError('Missing particle colour pair')
    return tuple(_integer(x, 'colour tag') for x in values)


def canonical_colour_topology(particles, permutation=None):
    """Return the oriented colour topology, independent of positive tag names.

    A supplied permutation orders these particles in the comparison basis.
    Zero remains zero; colour and anticolour positions are never interchanged.
    """
    order = tuple(range(len(particles))) if permutation is None else tuple(permutation)
    if sorted(order) != list(range(len(particles))):
        raise MappingError('Colour comparison requires a complete permutation')
    labels, result = {}, []
    for index in order:
        pair = []
        for tag in _colours(particles[index]):
            if tag < 0:
                raise MappingError('Negative colour tag')
            if tag and tag not in labels:
                labels[tag] = len(labels) + 1
            pair.append(labels[tag] if tag else 0)
        result.append(tuple(pair))
    return tuple(result)


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _basis(outer):
    basis = outer.get('component_basis')
    if not isinstance(basis, dict) or basis.get('schema') != 'macnlops-component-basis-v1':
        raise MappingError('Identical Born transport requires the exported component basis')
    return basis


def _certify_born_permutation(outer, permutation):
    nborn = outer['nexternal'] - 1
    if (sorted(permutation) != list(range(nborn)) or
            tuple(permutation[:2]) != (0, 1)):
        raise MappingError('Born transport must preserve both incoming beams')
    if tuple(permutation) == tuple(range(nborn)):
        return
    states = _basis(outer)['external_states']
    if len(states) != nborn or any(_freeze(states[i]) != _freeze(states[j])
                                   for i, j in enumerate(permutation)):
        raise MappingError('Born permutation exchanges distinct tagged/helicity states')


def _graph_signature(tree, permutation):
    """Canonical interaction tree rooted at the fixed incoming leg one."""
    nodes, edges = tree['nodes'], tree['edges']
    adjacency = [[] for _ in nodes]
    if len(edges) != len(nodes) - 1:
        raise MappingError('Born diagram metadata is not a tree')
    for first, second in edges:
        if not (0 <= first < len(nodes) and 0 <= second < len(nodes)) or first == second:
            raise MappingError('Invalid Born diagram edge')
        adjacency[first].append(second)
        adjacency[second].append(first)
    inverse = {source + 1: native + 1 for native, source in enumerate(permutation)}
    leaves = {node['leg']: index for index, node in enumerate(nodes) if 'leg' in node}
    if set(leaves) != set(inverse) or sum('leg' in node for node in nodes) != len(inverse):
        raise MappingError('Born diagram has incomplete external labels')
    visited = set()

    def walk(index, parent):
        if index in visited:
            raise MappingError('Cyclic Born diagram metadata')
        visited.add(index)
        node = nodes[index]
        children = [other for other in adjacency[index] if other != parent]
        if 'leg' in node:
            if parent is not None and children:
                raise MappingError('An external Born leg is not a leaf')
            tag = ('external', inverse[node['leg']], node['pdg'])
        else:
            tag = ('vertex', _freeze(node['interaction']))
        return tag, tuple(sorted((walk(child, index) for child in children), key=repr))

    result = walk(leaves[1], None)
    if len(visited) != len(nodes):
        raise MappingError('Disconnected Born diagram metadata')
    return result


def transport_component_basis(registry, born_permutation, config, graph, flow):
    """Map an actual source (diagram configuration, amplitude, flow) exactly.

    Returns their native indices after reordering the Born legs by
    ``born_permutation``.  Matching uses interaction trees and oriented
    colour topologies from the exact provider registry.  Missing or
    ambiguous transport is an error, never a copied or guessed ordinal.
    Identity transport remains compatible with the older registries.
    """
    try:
        outer = registry.get('registry', registry)['outer']
        permutation = tuple(born_permutation)
        _certify_born_permutation(outer, permutation)
        config, graph, flow = (_integer(value, name) for value, name in
                               ((config, 'source config'), (graph, 'source graph'),
                                (flow, 'source flow')))
        if min(config, graph, flow) < 1:
            raise MappingError('Source basis indices must be positive')
        identity = tuple(range(outer['nexternal'] - 1))
        if permutation == identity:
            return config, graph, flow
        basis = _basis(outer)
        sources = [item for item in basis['graphs']
                   if item['config'] == config and item['graph'] == graph]
        if len(sources) != 1 or flow > len(basis['flows']):
            raise MappingError('Source component is absent from its Born basis')
        expected_graph = _graph_signature(sources[0]['tree'], permutation)
        graphs = [item for item in basis['graphs']
                  if _graph_signature(item['tree'], identity) == expected_graph]
        source_colours = [{'colors': pair} for pair in basis['flows'][flow - 1]]
        expected_flow = canonical_colour_topology(source_colours, permutation)
        flows = [index + 1 for index, colours in enumerate(basis['flows'])
                 if canonical_colour_topology([{'colors': pair} for pair in colours])
                 == expected_flow]
        if len(graphs) != 1 or len(flows) != 1:
            raise MappingError('Born diagram/flow permutation has no unique certified image')
        return graphs[0]['config'], graphs[0]['graph'], flows[0]
    except MappingError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise MappingError('Malformed Born component basis') from exc


def component_orbit_factors(registry, sector_number):
    """Return (per-emitter normalization, emitter orbit, source orbit).

    The global evaluator removes F = fac_i*fac_j from the reduced native
    representative.  Its get_mbar still contains I_B/I_R.  Physical
    densities per actual Born emitter multiply *all* R,H,K by I_R/I_B.
    The reduced source instead has F times that unit density and covers
    O = F*I_B/I_R equivalent Born emitters.  No veto ratio changes under
    this common rescaling.  Counts use complete external state identities,
    including tags/helicities, rather than PDGs alone.
    """
    try:
        outer = registry.get('registry', registry)['outer']
        sectors = [item for item in outer['sectors'] if item['sector'] == sector_number]
        if len(sectors) != 1:
            raise MappingError('Missing or ambiguous sector for statistical normalization')
        sector = sectors[0]
        i, j = sector['fks']['i'] - 1, sector['fks']['j'] - 1
        counts = set()
        for process in sector['processes']:
            states = [_freeze(state) for state in process[1]]
            fac_i = sum(state == states[i] for state in states[2:])
            fac_j = (sum(state == states[j] for state in states[2:])
                     - int(states[i] == states[j])) if j >= 2 else 1
            counts.add(fac_i * fac_j)
        if len(counts) != 1 or next(iter(counts)) < 1:
            raise MappingError('Native flavour rows have inconsistent identical-particle orbits')
        source_orbit = next(iter(counts))
        born_identity = _integer(outer['born_identity'], 'Born statistical denominator')
        real_identity = _integer(sector['real_identity'], 'real statistical denominator')
        if min(born_identity, real_identity) < 1:
            raise MappingError('Nonpositive native statistical denominator')
        normalization = Fraction(real_identity, born_identity)
        emitters = Fraction(source_orbit, 1) / normalization
        if emitters.denominator != 1 or emitters < 1:
            raise MappingError('Native statistical factors do not give an integer emitter orbit')
        if j < 2 and emitters != 1:
            raise MappingError('A fixed incoming beam cannot have an emitter orbit')
        return normalization, int(emitters), source_orbit
    except MappingError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise MappingError('Malformed native statistical metadata') from exc


def _source_origins(source, snapshot, explicit=None):
    if explicit is not None:
        origins = tuple(_integer(x, 'source Pythia index') for x in explicit)
        if len(origins) != len(source) or len(set(origins)) != len(origins):
            raise MappingError('Invalid explicit source Pythia indices')
    else:
        origins = []
        for leg, particle in enumerate(source):
            statuses = (21,) if leg < 2 else (22, 23)
            matches = [index for index, old in snapshot.items()
                       if abs(old['status']) in statuses
                       and _pid(old) == _pid(particle)
                       and _colours(old) == _colours(particle)
                       and _same_momentum(old, particle)]
            if len(matches) != 1:
                raise MappingError('Ambiguous or missing original Born leg %d' % (leg + 1))
            origins.append(matches[0])
        origins = tuple(origins)
        if len(set(origins)) != len(origins):
            raise MappingError('Original Born legs do not have distinct ancestry')
    for leg, index in enumerate(origins):
        old = snapshot.get(index)
        if (old is None or _pid(old) != _pid(source[leg]) or
                _colours(old) != _colours(source[leg]) or
                not _same_momentum(old, source[leg])):
            raise MappingError('Explicit source ancestry disagrees with Born leg')
    return origins


def _first_origins(index, snapshot, origin_to_leg):
    """Stop each backwards ancestry path at its first source hard leg."""
    pending, visited, found = [index], set(), set()
    while pending:
        current = pending.pop()
        if current in visited or current == 0:
            continue
        visited.add(current)
        if current in origin_to_leg:
            found.add(origin_to_leg[current])
            continue
        old = snapshot.get(current)
        if old is None:
            raise MappingError('Ancestry refers to a missing Pythia record')
        mothers = old.get('mothers', ())
        if len(mothers) != 2:
            raise MappingError('Missing Pythia mothers')
        pending.extend(_integer(x, 'mother index') for x in mothers)
    return found


def _is_quark(pid):
    return 1 <= abs(pid) <= 6


def _qcd_branch(kind, before, after, emitted):
    if kind == 'ISR':
        return ((before == after and emitted == 21 and
                 (before == 21 or _is_quark(before))) or
                (after == 21 and _is_quark(before) and emitted == -before) or
                (before == 21 and _is_quark(after) and emitted == after))
    return ((_is_quark(before) and
             ((after == before and emitted == 21) or
              (emitted == before and after == 21))) or
            (before == 21 and
             ((after == 21 and emitted == 21) or
              (_is_quark(after) and emitted == -after))))


def _process_pdgs(sector):
    """Read the complete external-process identities exported in the registry."""
    try:
        return {tuple(_integer(leg[0], 'registry PDG') for leg in process[1])
                for process in sector['processes']}
    except (KeyError, IndexError, TypeError) as exc:
        raise MappingError('Malformed native real-process registry') from exc


def branch_candidates(record, registry, source_directory=None):
    """Certify native routes for one actual S first branching.

    ``record`` is a ``macnlops-one-step-v1`` sidecar dictionary.  ``registry``
    is the exact source directory's mc_histories.json, or its ``registry``
    payload.  The source Born ordering must match that provider.  Its
    integration-owner sector is deliberately not used to select a branch.

    Identical daughters and g -> q qbar orientations are returned separately
    when the exported sectors certify them.  No orientation is selected here:
    the caller must mark with physical native K weights and verify their sum
    against the actual branching law.  In particular, S_ij is not that weight.
    Identical spectators are followed by ancestry, never by a sorted PDG list.
    """
    if not record.get('emitted'):
        return ()
    try:
        payload = registry.get('registry', registry)
        outer = payload['outer']
        if source_directory is not None and PurePath(str(source_directory)).name != outer['directory']:
            raise MappingError('Component route crosses the source Born directory')
        source, endpoint = record['source_particles'], record['particles']
        nborn, nreal = len(source), len(endpoint)
        if nborn < 3 or nreal != nborn + 1 or outer['nexternal'] != nreal:
            raise MappingError('First branching must add exactly one external parton')
        born_ids = tuple(_pid(x) for x in source)
        if born_ids not in {tuple(row) for row in outer['born_pdgs']}:
            raise MappingError('Source Born ordering does not match its native provider')
        entries = record['pythia_event']
        snapshot = {_integer(p['index'], 'Pythia index'): p for p in entries}
        if len(snapshot) != len(entries) or any(index < 1 for index in snapshot):
            raise MappingError('Duplicate or nonpositive Pythia index')
        endpoint_indices = tuple(_integer(x, 'hard Pythia index')
                                 for x in record['hard_pythia_indices'])
        if len(endpoint_indices) != nreal or len(set(endpoint_indices)) != nreal:
            raise MappingError('Endpoint requires distinct external Pythia indices')
        for index, particle in zip(endpoint_indices, endpoint):
            old = snapshot.get(index)
            if (old is None or _pid(old) != _pid(particle) or
                    _colours(old) != _colours(particle) or
                    not _same_momentum(old, particle)):
                raise MappingError('Endpoint particles disagree with the Pythia snapshot')
        origins = _source_origins(source, snapshot, record.get('source_pythia_indices'))
        origin_to_leg = {index: leg for leg, index in enumerate(origins)}
        branch = record['branch_indices']
        before = _integer(branch['radiator_before'], 'radiator before')
        after = _integer(branch['radiator_after'], 'radiator after')
        emitted = _integer(branch['emitted'], 'emitted index')
        if before not in origin_to_leg or after not in endpoint_indices or emitted not in endpoint_indices:
            raise MappingError('Branch indices do not identify source and endpoint legs')
        parent = origin_to_leg[before]
        radiator, sister = endpoint_indices.index(after), endpoint_indices.index(emitted)
        kind = record['emission_type']
        if (kind not in ('ISR', 'FSR') or radiator == sister or sister < 2 or
                (kind == 'ISR' and (parent >= 2 or radiator != parent)) or
                (kind == 'FSR' and (parent < 2 or radiator < 2))):
            raise MappingError('Branch kind, beam or external daughter indices disagree')
        if not _qcd_branch(kind, born_ids[parent], _pid(endpoint[radiator]), _pid(endpoint[sister])):
            raise MappingError('Recorded flavours are not a QCD branching')

        # Backwards ISR replaces incoming ancestors in the event history, so
        # incoming identities follow their fixed beam positions.  Final-state
        # spectators, including global-recoil copies, follow their own ancestry.
        descendants = {0: 0, 1: 1}
        for leg in range(2, nreal):
            if leg == sister:
                continue
            parents = _first_origins(endpoint_indices[leg], snapshot, origin_to_leg)
            if len(parents) != 1:
                raise MappingError('Endpoint spectator has ambiguous Born ancestry')
            source_leg = next(iter(parents))
            if source_leg < 2 or source_leg in descendants:
                raise MappingError('Two endpoint particles represent the same Born leg')
            descendants[source_leg] = leg
        if set(descendants) != set(range(nborn)) or descendants[parent] != radiator:
            raise MappingError('Endpoint does not preserve the source Born leg identities')

        orientations = [(sister, radiator, 'native')]
        if kind == 'FSR' and born_ids[parent] == 21:
            tag = 'gg-swapped' if _pid(endpoint[sister]) == 21 else 'gqq-swapped'
            orientations.append((radiator, sister, tag))
        candidates, seen = [], set()
        for sector in outer['sectors']:
            fks = sector['fks']
            if 'QCD' not in fks['splitting_type']:
                continue
            i, j = fks['i'] - 1, fks['j'] - 1
            if not (2 <= i < nreal and 0 <= j < nreal and i != j):
                raise MappingError('Invalid native FKS pair')
            born_slots = tuple(k for k in range(nreal) if k != i)
            father = fks['ij'] - 1
            if (not 0 <= father < nborn or born_slots.index(j) != father or
                    (kind == 'ISR') != (j < 2)):
                continue
            born_permutation = list(range(nborn))
            if father != parent:
                if father < 2 or parent < 2 or born_ids[father] != born_ids[parent]:
                    continue
                states = _basis(outer)['external_states']
                if _freeze(states[father]) != _freeze(states[parent]):
                    continue
                born_permutation[father], born_permutation[parent] = parent, father
                _certify_born_permutation(outer, born_permutation)
            born_permutation = tuple(born_permutation)
            physical_rows = _process_pdgs(sector)
            for marked_i, marked_j, orientation in orientations:
                permutation = [None] * nreal
                for born_leg, real_leg in enumerate(born_slots):
                    permutation[real_leg] = descendants[born_permutation[born_leg]]
                permutation[i], permutation[j] = marked_i, marked_j
                permutation = tuple(permutation)
                if (sorted(permutation) != list(range(nreal)) or
                        permutation[:2] != (0, 1)):
                    continue
                if tuple(_pid(endpoint[k]) for k in permutation) not in physical_rows:
                    continue
                key = (sector['sector'], permutation)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(BranchComponent(
                    sector=_integer(sector['sector'], 'native sector'),
                    context=_integer(outer['context'], 'native context'),
                    provider=_integer(outer['provider'], 'Born provider'),
                    permutation=permutation,
                    born_permutation=born_permutation,
                    emitted=marked_i, radiator=marked_j,
                    orientation=orientation))
        if not candidates:
            raise MappingError('Actual branching has no certified native source component')
        return tuple(replace(candidate, requires_marking=len(candidates) > 1)
                     for candidate in candidates)
    except MappingError:
        raise
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise MappingError('Malformed branching or source registry') from exc
