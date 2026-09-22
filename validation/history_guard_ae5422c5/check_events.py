"""Structural and native-history checks for the archived ttbar sample.

Run with optional LHE and registry paths. Cross-section comparisons are
recorded separately in results.json.
"""

from collections import Counter
import gzip
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def check(path):
    with gzip.open(path, 'rt') as stream:
        text = stream.read()
    # MG cards in the header need not be XML escaped. Parse event blocks only.
    blocks = re.findall(r'<event>.*?</event>', text, re.S)
    assert blocks, 'No events'
    scales, weight_counts, final_states = [], set(), Counter()
    for block in blocks:
        event = ET.fromstring(block)
        lines = [line.split() for line in event.text.splitlines() if line.strip()]
        nlegs, scale = int(lines[0][0]), float(lines[0][3])
        assert math.isfinite(scale) and scale >= 0, 'Invalid shower scale'
        scales.append(scale)
        momentum = [0.0]*4
        colours = Counter()
        final = []
        for particle in lines[1:nlegs+1]:
            pdg, status = int(particle[0]), int(particle[1])
            if status not in (-1, 1):
                continue
            if status == 1:
                final.append(pdg)
            for index, value in enumerate(particle[6:10]):
                momentum[index] += status*float(value)
            for index, sign in ((4, 1), (5, -1)):
                colour = int(particle[index])
                if colour:
                    colours[colour] += sign*status
        assert max(map(abs, momentum)) < 1e-5, momentum
        assert all(value == 0 for value in colours.values()), colours
        weights = [float(node.text) for node in event.findall('rwgt/wgt')]
        assert weights and all(map(math.isfinite, weights)), 'Missing/invalid weights'
        weight_counts.add(len(weights))
        final_states[','.join(map(str, sorted(final)))] += 1
    assert 'scale_variation' in text and 'PDF_variation' in text
    return dict(events=len(blocks), shower_scale_range_GeV=[min(scales), max(scales)],
                reweights_per_event=sorted(weight_counts), final_states=dict(final_states),
                momentum_conservation=True, colour_closure=True, finite_weights=True)


def check_histories(path, registry_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from madgraph.various import lhe_parser
    registry = json.loads(Path(registry_path).read_text())
    histories = []
    for owner in registry['contexts']:
        tables = registry['histories'][owner['directory']]
        rows = [h for table in tables for h in table['histories']]
        aliases = [(owner['context'], s['sector']) for s in owner['sectors']]
        aliases += sorted({(h['context'], h['sector']) for h in rows} - set(aliases))
        index = 0
        for table in tables:
            sector = owner['sectors'][table['sector']-1]
            flavours = [tuple(leg[0] for leg in p[1]) for p in sector['processes']]
            for history in table['histories']:
                index += 1
                histories.append((owner['context'], table['sector'], index,
                    history['provider'], history['context'],
                    aliases.index((history['context'], history['sector']))+1, flavours))
    counts = Counter()
    max_central_error = 0.0
    for event in lhe_parser.EventFile(str(path)):
        pdgs = tuple(p.pid for p in event if p.status in (-1, 1))
        max_central_error = max(max_central_error,
            abs(event.parse_reweight()['1001']/event.wgt - 1))
        event.parse_nlo_weight()
        for cevent in event.nloweight.cevents:
            for weight in cevent.wgts:
                assert weight.native_provenance is not None
                provider, context, history, owner = weight.native_provenance
                assert provider > 0 and context > 0 and owner > 0
                restored = lhe_parser.OneNLOWeight(weight.__str__(mode='formatted'))
                assert restored.native_provenance == weight.native_provenance
                counts['records'] += 1
                if history == 0:
                    continue
                matches = [h for h in histories if
                    (owner, history, provider, context, weight.nfks) == h[1:6]
                    and pdgs in h[6]]
                assert len(matches) == 1, (pdgs, weight.native_provenance)
                counts['native_history_records'] += 1
                counts['foreign_context_records'] += matches[0][0] != context
    assert counts['foreign_context_records'] > 0
    # The LHE reweight output prints five significant digits.
    assert max_central_error < 1e-4
    return dict(counts, max_central_reweight_relative_error=max_central_error,
                provenance_round_trip=True, unique_outer_ownership=True)


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('ttbar_events.lhe.gz')
    registry = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent/'ttbar/registry.json'
    print(json.dumps(dict(check(path), **check_histories(path, registry)), indent=2, sort_keys=True))
