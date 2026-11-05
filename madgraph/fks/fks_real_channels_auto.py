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
"""Choose the default of the run-card parameter real_channels_x0 of a process,
i.e. whether the real-emission integration channels (section 6.3 of
arXiv:0908.4272) are worth enabling.

The Born channels generate a real-emission point from an underlying Born
point and the FKS radiation variables of one sector (i,j).  A propagator of
a real-emission diagram follows this parametrisation when its momentum does
not separate i from j: its virtuality is then a Born invariant (with i and j
merged), and the propagators that separate them only become singular when i
is soft or collinear to j, which the radiation variables sample.  A real
diagram is 'covered' if, for at least one of the FKS sectors of its real
process, none of its peaking propagators separates i from j.  A diagram that
is not covered has a structure that no Born channel follows: a t-channel
exchange or a resonance opened by the extra parton, as in u a > w+ h d or
u g > w+ d b b~ (gluon fusion into b b~).  Only then can the dedicated
real-emission channels improve the integration.

Only the propagators that can enhance the integrand count (peaking()).
Their masses enter through the threshold of the legs on their side, where a
resonance of the diagram counts with its mass, and a massless propagator
peaks only if its side is made of massless partons (coloured, or FKS
partons of the real process, like photons and leptons in QED corrections).
The decision is then the same whether a resonance is external or decayed
(p p > w+ j and p p > e+ ve j [QCD], p p > w+ and p p > e+ ve [QED]).
"""

import logging

logger = logging.getLogger('madgraph.fks_real_channels_auto')

# value of real_channels_x0 used when the channels are expected to help:
# a few percent gives the best performance in the benchmarks of the
# MadGraph7 paper (p p > w+ h [QED] and p p > w+ b b~ [QCD], 4FS)
AUTO_X0 = 0.05
# above this number of hard channels, the grid-adaptation overhead of the
# additional jobs is expected to outweigh the gain
MAX_HARD_CHANNELS = 1000


def diagram_propagators(diagram, legs):
    """Return, for each propagator of a base_objects.Diagram, the pair
    (set of external leg numbers on one side, pdg, s_channel).
    legs are the external legs of the process."""

    content = dict((l['number'], frozenset([l['number']])) for l in legs)
    props = []
    # the last vertex has no propagator; each other vertex produces one
    # leg, built from its other legs. Its 'state' is False for t-channel
    for vertex in diagram['vertices'][:-1]:
        side = frozenset()
        for leg in vertex['legs'][:-1]:
            side |= content[leg['number']]
        out = vertex['legs'][-1]
        content[out['number']] = side
        props.append((side, out['id'], out['state']))
    return props


class RealChannelAnalyser(object):
    """Read the masses and widths of the real-emission diagrams of FKS
    matrix elements"""

    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters

    def value(self, name):
        """abs of a mass/width, None if unknown"""
        if name.upper() == 'ZERO':
            return 0.
        if self.parameters is None or name not in self.parameters:
            return None
        return abs(self.parameters[name].real)

    def mass(self, pdg):
        part = self.model.get_particle(pdg)
        if not part:
            return None
        return self.value(part.get('mass'))

    def coloured(self, pdg):
        part = self.model.get_particle(pdg)
        return bool(part) and part.get('color') != 1

    def propagators(self, amplitude):
        """For each diagram of amplitude (a real-emission base amplitude),
        its propagators as [legs on one side, |pdg|, s-channel, mass,
        width]. The fake propagators of multi-leg vertices are dropped"""

        legs = amplitude['process']['legs']
        result = []
        for diagram in amplitude['diagrams']:
            props = []
            for side, pdg, s_channel in diagram_propagators(diagram, legs):
                part = self.model.get_particle(pdg)
                if not part:
                    continue
                props.append([sorted(side), abs(pdg), bool(s_channel),
                              self.value(part.get('mass')),
                              self.value(part.get('width'))])
            result.append(props)
        return result


def threshold(finals, masses, resonances, strict=False):
    """Lower bound of the invariant mass of the final-state legs finals:
    the masses of the resonances of the diagram that they contain (a
    resonance pins the invariant mass of its decay products to its mass),
    plus the masses of the other legs. The resonances of a tree diagram are
    nested or disjoint, so the largest ones are disjoint. With strict, a
    resonance on exactly finals is not used."""
    total = 0.
    left = set(finals)
    for products, mass in sorted(resonances.items(), key=lambda r: -len(r[0])):
        if products <= left and not (strict and products == finals):
            total += mass
            left -= products
    return total + sum(masses[n] for n in left)


def final_sides(side, initial, allnum):
    """The final-state legs on each side of a propagator: one set for an
    s-channel propagator (the side without the initial state), two for a
    t-channel propagator (the final-state legs attached to each beam)"""
    other = frozenset(allnum - side)
    if not side & initial:
        return [frozenset(side)]
    if not other & initial:
        return [other]
    return [frozenset(side - initial), other - initial]


def diagram_resonances(props, initial, masses):
    """The s-channel massive propagators of a diagram that can go on shell:
    {final-state legs: mass}. Built from the innermost one out, since the
    threshold of a resonance includes the resonances it decays into"""

    allnum = set(masses) | initial
    candidates = []
    for side, apdg, s_channel, mass, width in props:
        sides = final_sides(frozenset(side), initial, allnum)
        if len(sides) == 1 and mass and width != 0.:
            candidates.append((sides[0], mass))
    resonances = {}
    for finals, mass in sorted(candidates, key=lambda c: len(c[0])):
        if mass > threshold(finals, masses, resonances, strict=True) * \
           (1. + 1e-10):
            resonances[finals] = mass
    return resonances


def peaking(prop, initial, masses, partons, resonances, hard=0.):
    """True if the propagator can enhance the integrand away from the
    soft and collinear limits. masses are those of the final-state legs,
    partons the final-state legs that are partons of the real process
    (coloured, or the i or j of one of its sectors), hard the threshold of
    the Born process (born_threshold)"""

    side, apdg, s_channel, mass, width = prop
    side = frozenset(side)
    allnum = set(masses) | initial
    # the s-channel propagator carrying all of sqrt(s-hat): no peak
    if side == initial or allnum - side == initial:
        return False
    if mass is None or None in masses.values():
        return True
    sides = final_sides(side, initial, allnum)
    if mass == 0.:
        # a peak only if all the legs on one side are massless partons,
        # with no resonance among them: its virtuality then only stays away
        # from zero by the soft and collinear limits, or by the jet cuts.
        # Otherwise it is bounded by a mass, a resonance, or (for leptons,
        # or photons in QCD corrections) by the cuts that the Born already
        # needs on them, and as smooth as a massive one
        return any(all(n in partons for n in finals) and
                   threshold(finals, masses, resonances) == 0.
                   for finals in sides)
    if len(sides) == 1:
        return sides[0] in resonances
    # t-channel: enhanced when light compared to the hard scale, bounded
    # below by the final-state threshold of this diagram and of the Born
    # (without any, s-hat is set by the cuts, which exceed the EW masses)
    total = max(threshold(frozenset(masses), masses, resonances), hard)
    return total == 0. or mass < 0.5 * total


def peak_propagators(diagrams, initial, masses, partons, hard=0.):
    """For each diagram (as given by RealChannelAnalyser.propagators), its
    peaking propagators as [legs on one side, |pdg|]"""

    result = []
    for props in diagrams:
        resonances = diagram_resonances(props, initial, masses)
        result.append([[prop[0], prop[1]] for prop in props
                       if peaking(prop, initial, masses, partons, resonances,
                                  hard)])
    return result


def born_threshold(diagrams, initial, masses):
    """The hard scale of a Born process: the largest threshold of its final
    state over its diagrams, with the resonances (also the one carrying all
    of sqrt(s-hat), as in u d~ > e+ ve). It is the same whether a resonance
    is external (w+) or decayed (e+ ve). 0 if a mass is unknown"""

    if None in masses.values():
        return 0.
    return max([threshold(frozenset(masses), masses,
                          diagram_resonances(props, initial, masses))
                for props in diagrams] + [0.])


def canonical(side, nexternal):
    """The side of a propagator that does not contain leg 1"""
    side = frozenset(side)
    if 1 in side:
        side = frozenset(range(1, nexternal + 1)) - side
    return side


def leg_map(real_legs, born_legs, i, j, ij):
    """Born leg number of each real-emission leg but i: j becomes ij, and
    the other legs are matched by pdg and state (identical particles are
    interchangeable, since the Born diagrams are symmetric under their
    exchange)"""

    free = [(n, pdg, state) for n, pdg, state in born_legs if n != ij]
    mapping = {j: ij}
    for n, pdg, state in real_legs:
        if n in (i, j):
            continue
        for k, born in enumerate(free):
            if born[1:] == (pdg, state):
                mapping[n] = born[0]
                del free[k]
                break
        else:
            return None
    return mapping


class Sector(object):
    """An FKS sector (i, j) of a real process, with what the Born channels
    of its directory follow"""

    def __init__(self, i, j, soft, mapping, born_props, born_nexternal):
        self.i, self.j, self.soft = i, j, soft
        self.mapping = mapping
        self.born_props = born_props
        self.born_nexternal = born_nexternal

    def follows(self, side, apdg, nexternal):
        """True if the Born channels of this sector follow the propagator"""

        i, j = self.i, self.j
        if (i in side) == (j in side):
            return True
        # i attached to an external leg k: enhanced only when i is soft
        # (sampled by xi) or collinear to k (its own sector)
        if self.soft and (len(side) == 2 and i in side or
                          len(side) == nexternal - 2 and i not in side):
            return True
        # the same propagator, with i merged into j, in a Born diagram:
        # followed, up to the recoil of the emission
        if self.mapping is None:
            return False
        born_side = canonical([self.mapping[k] for k in side if k != i],
                              self.born_nexternal)
        return (born_side, apdg) in self.born_props


def covered(peaks, sectors, nexternal):
    """True if the Born channels of one of the sectors follow all the
    peaking propagators of a diagram"""

    return any(all(sector.follows(set(side), apdg, nexternal)
                   for side, apdg in peaks) for sector in sectors)


def identical_permutations(sectors, legs):
    """Close the list of (i, j, ...) under the exchange of identical
    legs: FKS keeps one sector per set of equivalent ones"""

    groups = {}
    for n, pdg, state in legs:
        groups.setdefault((pdg, state), []).append(n)
    swaps = [(a, b) for group in groups.values() if len(group) > 1
             for a in group for b in group if a < b]
    result = set(sectors)
    todo = list(result)
    while todo:
        sector = todo.pop()
        for a, b in swaps:
            swap = {a: b, b: a}
            new = (swap.get(sector[0], sector[0]),
                   swap.get(sector[1], sector[1])) + tuple(sector[2:])
            if new not in result:
                result.add(new)
                todo.append(new)
    return sorted(result)


_analysers = {}

def get_analyser(model):
    """One analyser (with the parameter values) per model. The Born and the
    real amplitudes of a directory carry different model objects, so all of
    them are kept. An entry holds its model: its id cannot be reused by a
    new model while cached, and the identity check guards it anyway"""
    entry = _analysers.get(id(model))
    if entry is None or entry.model is not model:
        from madgraph.interface.amcatnlo_interface import aMCatNLOInterface
        entry = RealChannelAnalyser(
            model, aMCatNLOInterface.get_parameter_values(model))
        _analysers[id(model)] = entry
    return entry


def clear_analysers():
    """Release the analysers (and their models) once an output is done"""
    _analysers.clear()


def leg_list(process):
    return [[l['number'], l['id'], bool(l['state'])] for l in process['legs']]


def flavours(matrix_element):
    """The pdg codes of all the processes of a matrix element: a directory
    can hold several flavours with the same matrix element (u g > u g and
    d g > d g), with the same leg positions"""
    result = []
    for process in matrix_element.get('processes'):
        pdgs = [l['id'] for l in process['legs']]
        if pdgs not in result:
            result.append(pdgs)
    return result


def with_flavour(legs, pdgs):
    """legs (as [number, pdg, state]) with the pdg codes pdgs"""
    return [(n, pdg, state) for (n, old, state), pdg in zip(legs, pdgs)]


def born_structure(born_amplitude):
    """Legs and propagators of the Born diagrams of a directory"""
    legs = born_amplitude['process']['legs']
    props = set()
    for diagram in born_amplitude['diagrams']:
        for side, pdg, s_channel in diagram_propagators(diagram, legs):
            props.add((tuple(sorted(canonical(side, len(legs)))), abs(pdg)))
    return {'legs': leg_list(born_amplitude['process']),
            'props': sorted(props)}


def real_structures(fks_matrix_element):
    """The information needed by decide() for one Born directory (an
    FKSHelasProcess), as JSON-friendly data"""

    born_amp = fks_matrix_element.born_me.get('base_amplitude')
    born = born_structure(born_amp)
    born['flavours'] = flavours(fks_matrix_element.born_me)
    legs = born_amp['process']['legs']
    analyser = get_analyser(born_amp['process']['model'])
    born['threshold'] = born_threshold(
        analyser.propagators(born_amp),
        frozenset(l['number'] for l in legs if not l['state']),
        dict((l['number'], analyser.mass(l['id'])) for l in legs if l['state']))
    reals = []
    for real in fks_matrix_element.real_processes:
        amp = real.matrix_element.get('base_amplitude')
        analyser = get_analyser(amp['process']['model'])
        reals.append({
            'legs': leg_list(amp['process']),
            'masses': [analyser.mass(l['id']) for l in amp['process']['legs']],
            'coloured': [l['number'] for l in amp['process']['legs']
                         if analyser.coloured(l['id'])],
            'flavours': flavours(real.matrix_element),
            'sectors': sorted(set((info['i'], info['j'],
                bool(info.get('need_color_links') or
                     info.get('need_charge_links')), info['ij'])
                for info in real.fks_infos)),
            'diagrams': analyser.propagators(amp)})
    return {'born': born, 'reals': reals}


def collect_reals(directories):
    """Combine the structures (real_structures) of all the Born
    directories, per real process (keyed by its pdg codes and states, in a
    canonical leg order). The sectors of a real process are collected over
    all the directories it belongs to, since the sector functions are
    normalised over all of them, and over all the flavours that a
    directory holds"""

    reals = {}
    for directory in directories:
        born = directory['born']
        born_legs = [tuple(l) for l in born['legs']]
        born_flavours = [with_flavour(born_legs, pdgs) for pdgs in
                         born.get('flavours', [[l[1] for l in born_legs]])]
        born_props = set((frozenset(side), apdg)
                         for side, apdg in born['props'])
        for real in directory['reals']:
            for pdgs in real.get('flavours', [[l[1] for l in real['legs']]]):
                legs = with_flavour(real['legs'], pdgs)
                # the same real process can have its final-state legs in a
                # different order in another directory: relabel them
                order = sorted(legs, key=lambda l: (l[2], l[2] and l[1], l[0]))
                relabel = dict((l[0], n + 1) for n, l in enumerate(order))
                legs = [(relabel[n], pdg, state) for n, pdg, state in order]
                key = tuple(l[1:] for l in legs)
                if key not in reals:
                    reals[key] = {
                        'diagrams': [[[sorted(relabel[k] for k in prop[0])]
                                      + prop[1:] for prop in diagram]
                                     for diagram in real['diagrams']],
                        'masses': dict((relabel[n], mass) for (n, pdg, state),
                                       mass in zip(real['legs'], real['masses'])
                                       if state),
                        'coloured': set(relabel[n]
                                        for n in real.get('coloured', [])),
                        'nexternal': len(legs), 'sectors': [], 'hard': 0.}
                # the hard scale of all the Born processes it comes from
                reals[key]['hard'] = max(reals[key]['hard'],
                                         born.get('threshold') or 0.)
                sectors = [(relabel[i], relabel[j], soft, ij)
                           for i, j, soft, ij in real['sectors']]
                for i, j, soft, ij in identical_permutations(sectors, legs):
                    # the Born flavour that this real flavour comes from
                    mapping = None
                    for flavour in born_flavours:
                        mapping = leg_map(legs, flavour, i, j, ij)
                        if mapping is not None:
                            break
                    reals[key]['sectors'].append(Sector(
                        i, j, soft, mapping, born_props, len(born_legs)))
    return reals


def decide(directories, n_hard_channels):
    """Return (default of real_channels_x0, summary) for the structures
    (real_structures) of all the Born directories of a process"""

    reals = collect_reals(directories)
    summary = {'n_reals': len(reals), 'n_diagrams': 0, 'n_uncovered': 0,
               'reals_uncovered': [], 'uncovered_peaks': {},
               'n_hard_channels': n_hard_channels}
    for key, entry in reals.items():
        initial = frozenset(n + 1 for n, (pdg, state) in enumerate(key)
                            if not state)
        # the coloured legs, and the other ones that are the i or j of a
        # sector (photons and leptons in QED corrections)
        partons = entry['coloured'] | set(
            n for sector in entry['sectors'] for n in (sector.i, sector.j))
        partons -= initial
        peaks = peak_propagators(entry['diagrams'], initial, entry['masses'],
                                 partons, entry['hard'])
        uncovered = [n + 1 for n, diagram in enumerate(peaks)
                     if not covered(diagram, entry['sectors'],
                                    entry['nexternal'])]
        summary['n_diagrams'] += len(peaks)
        summary['n_uncovered'] += len(uncovered)
        if uncovered:
            summary['reals_uncovered'].append(
                ([pdg for pdg, state in key], uncovered, len(peaks)))
            summary['uncovered_peaks'][tuple(pdg for pdg, state in key)] = \
                dict((n, peaks[n - 1]) for n in uncovered)
    if not summary['n_uncovered'] or n_hard_channels > MAX_HARD_CHANNELS:
        return -1., summary
    return AUTO_X0, summary
