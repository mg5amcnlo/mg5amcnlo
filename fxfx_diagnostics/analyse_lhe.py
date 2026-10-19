from pathlib import Path
import gzip
import math
import numpy as np
import json
import argparse

out = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Analyse the existing w0j and w1j LHE files before showering.')
parser.add_argument('--root', type=Path, default=out.parent)
root = parser.parse_args().root.resolve()
edges = np.arange(-3., 3.00001, 0.1)
results = {}
for sample in ['w1j', 'w0j']:
    for run in ['showered-ckkw', 'showered']:
        path = root/sample/'Events'/run/'events.lhe.gz'
        counts = np.zeros((3, len(edges)-1))
        squares = counts.copy()
        totals = np.zeros(3)
        raw_counts = np.zeros(3, dtype=int)
        nevents = 0
        with gzip.open(path, 'rt') as f:
            for line in f:
                if not line.startswith('<event'): continue
                hdr = next(f).split()
                npart = int(hdr[0]); wgt = float(hdr[2]); scalup = float(hdr[3])
                qcd = []
                for i in range(npart):
                    p = next(f).split()
                    if p[1] == '1' and (abs(int(p[0])) in [1, 2, 3, 4, 5, 21]):
                        qcd.append(tuple(float(x) for x in p[6:10]))
                nevents += 1
                category = len(qcd)
                assert category <= 2
                totals[category] += wgt; raw_counts[category] += 1
                if category == 0: continue
                pt = [math.hypot(q[0],q[1]) for q in qcd]
                djr = pt[0]
                if category == 2:
                    phi = [math.atan2(q[1],q[0]) for q in qcd]
                    rapidity = [0.5*math.log((q[3]+q[2])/(q[3]-q[2])) for q in qcd]
                    dr2 = (rapidity[0]-rapidity[1])**2 + math.remainder(phi[0]-phi[1],2*math.pi)**2
                    djr = math.hypot(qcd[0][0]+qcd[1][0], qcd[0][1]+qcd[1][1]) if dr2 < 1. else max(pt)
                if djr > 0:
                    j = int(math.floor((math.log10(djr)+3.)/0.1))
                    if 0 <= j < counts.shape[1]:
                        counts[category,j] += wgt
                        squares[category,j] += wgt*wgt
        counts /= nevents; squares /= nevents**2; totals /= nevents
        label = sample+'/'+run
        results[label] = dict(edges=edges.tolist(), counts=counts.tolist(), squares=squares.tolist(),
                              totals=totals.tolist(), raw_counts=raw_counts.tolist(), nevents=nevents)
        print(label, 'events',nevents, 'cross section', totals.sum(), 'by parton count',totals,flush=True)
        for j, x in enumerate(edges[:-1]):
            if 1. <= x <= 1.8:
                print(f'{10**x:.2f}--{10**edges[j+1]:.2f}', counts[:,j], 'total',counts[:,j].sum(),flush=True)
(out/'lhe_histograms.json').write_text(json.dumps(results))
