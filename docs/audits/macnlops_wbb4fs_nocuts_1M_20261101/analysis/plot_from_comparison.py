#!/usr/bin/env python3
"""Render plots only from the frozen numerical comparison; no event reprocessing."""
from pathlib import Path
import argparse,json,hashlib
from matplotlib.backends.backend_pdf import PdfPages
from analyse import plot_page,DEFINITION
p=argparse.ArgumentParser(description=__doc__);p.add_argument('comparison',type=Path);args=p.parse_args()
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
before=sha(args.comparison);d=json.loads(args.comparison.read_text());index={x['name']:x for x in DEFINITION['histograms']}
with PdfPages(args.comparison.parent/'wbb_nocuts_validation.pdf') as pdf:
 for names,title in [(['primary scale 0-100','radiation jet1 pT 0-50','bb invariant mass'],'Radiation and bottom-pair mass'),(['W pT 0-500','bottom pT 0-500','antibottom pT 0-500'],'Stable hard-object transverse momenta'),(['Wbb invariant mass','bb pT 0-500','all jet1 pT 0-500'],'System and all-parton jet controls')]:
  plot_page(pdf,d['histograms']['first'],d['histograms']['full'],[index[n] for n in names],title)
assert before==sha(args.comparison)
print('Rendered three pages; numerical comparison SHA256 unchanged:',before)
