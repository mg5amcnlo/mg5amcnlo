"""Structural checks for the archived unshowered DY sample.

This does not establish cross-section or reweighting equivalence. Run with an
optional path to an LHE gzip file; the default is the archived sample beside
this script.
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


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name('dy_events.lhe.gz')
    print(json.dumps(check(path), indent=2, sort_keys=True))
