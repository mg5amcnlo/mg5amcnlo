#!/usr/bin/env python3
"""Collect published campaign panels into first-emission/full-shower PDFs.

Only completed, checked histogram datasets enter a book. Watch mode updates
the books as the streamed analysis publishes more datasets. Existing vector
panels are combined without rerendering or modifying their normalization.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import analyse_macnlops_campaign as analysis
import run_macnlops_campaign as campaign

PANELS = ('jet_pt_soft', 'system_pt_soft', 'system_mass', 'delta_phi_deficit')


def publish(base, level, configs):
    completed = []
    rows, panels = [], []
    for path in configs:
        config = json.loads(Path(path).read_text())
        if config['level'] != level:
            continue
        directory = base / level / (config['process'] + '_' + config['mode'])
        summary = directory / 'summary.json'
        if not summary.exists():
            continue
        record = json.loads(summary.read_text())
        if not record.get('complete'):
            continue
        config, data = analysis.load(path)
        if config['render_source_sha256'] != campaign.sha(Path(analysis.__file__)):
            raise RuntimeError('Plot renderer differs from the registered analysis')
        source = data['fiducial_totals']['reference']
        matched = data['fiducial_totals']['macnlops']
        rows.append(dict(process=config['process'], mode=config['mode'],
            source_events=data['events'], retained_events=data['accepted'],
            fiducial_source_events=source['events'], fiducial_retained_events=matched['events'],
            source_pb=source['sumw'], macnlops_pb=matched['sumw'],
            change_percent=100 * (matched['sumw'] / source['sumw'] - 1) if source['sumw'] else None,
            remaining_negative_H=data['totals']['H_negative_retained']['events']))
        for name in PANELS:
            panel = directory / (name + '.pdf')
            if not panel.is_file():
                raise RuntimeError('Completed dataset lacks a selected panel: ' + str(panel))
            panels.append(panel)
        completed.append(str(path))
    if not completed:
        return False
    directory = base / level
    receipt = directory / 'available_summary.json'
    inputs = {str(path):campaign.sha(path) for path in panels}
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if (previous.get('inputs') == inputs and
                campaign.sha(directory / 'validation_summary.pdf') == previous.get('sha256')):
            return previous['complete_level']
    with tempfile.TemporaryDirectory(prefix='.book_', dir=directory) as temporary:
        staged = Path(temporary) / 'summary.pdf'
        subprocess.run(['pdfunite', *map(str, panels), str(staged)], check=True)
        details = subprocess.check_output(['pdfinfo', str(staged)], text=True)
        pages = int(re.search(r'^Pages:\s*(\d+)', details, re.M)[1])
        if pages != len(panels):
            raise RuntimeError('Combined PDF has the wrong number of pages')
        staged.replace(directory / 'validation_summary.pdf')
    campaign.dump(directory / 'rates.json', rows)
    with (directory / 'rates.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected = sum(json.loads(Path(path).read_text())['level'] == level for path in configs)
    campaign.dump(receipt, dict(created_utc=campaign.now(), completed_datasets=completed,
        complete_level=len(completed) == expected, total_datasets=expected,
        pages=pages, ratio_denominator='macnlops', inputs=inputs,
        output=str(directory / 'validation_summary.pdf'),
        sha256=campaign.sha(directory / 'validation_summary.pdf'),
        normalization='Original signed weights; W+jet rates include the analysis jet selection'))
    print(campaign.now(), level, len(completed), '/', expected, 'datasets;', pages, 'pages', flush=True)
    return len(completed) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, default=campaign.DEFAULT_BASE)
    parser.add_argument('--analysis-dir', type=Path)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    base = (args.analysis_dir or args.campaign_dir / 'validation').resolve()
    analysis.BASE = base
    configs = json.loads((base / 'configs.json').read_text())
    while True:
        complete = [publish(base, level, configs) for level in ('first', 'full')]
        if all(complete) or not args.watch:
            return
        status = base / 'status.json'
        if status.exists():
            state = json.loads(status.read_text())
            if not state.get('active') and any(row['status'] == 'failed' for row in state.get('results', [])):
                raise RuntimeError('Streamed analysis failed; available books contain only completed datasets')
        time.sleep(15)


if __name__ == '__main__':
    main()
