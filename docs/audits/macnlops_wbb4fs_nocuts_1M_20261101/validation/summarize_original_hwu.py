"""Independent no-cut scalar summary from the complete ORIGINAL native histograms."""
from pathlib import Path
import hashlib,json
import numpy as np
B=Path('/export/slow1/rikkert/macnlops_wbb4fs_1M_nocuts_20261101')
root=B/'processes/Wbb_4FS/MCatNLO/RUN_PYTHIA8_3'
records=[];chunks=[]
for i in range(1,51):
    p=root/('run_%d'%i)/'MADatNLO.HwU'
    lines=iter(p.read_text().splitlines());found=None
    for line in lines:
        if line.startswith('<histogram>') and 'primary scale 0-100' in line:
            rows=[]
            for line in lines:
                if line.startswith('<\\histogram>'):break
                if line.strip():rows.append([float(s.replace('D','E')) for s in line.split()])
            a=np.asarray(rows);sel=(a[:,0]>=5)&(a[:,1]<=20)
            assert sel.sum()==15
            # The labels in these frozen files are xmin,xmax,source,dy,matched,veto,removal.
            z=a[sel].sum(axis=0);W,M,V,R=z[2],z[4],z[5],z[6]
            found=[W,M,W-V,R-W]
            break
    assert found is not None
    chunks.append(found)
    records.append(dict(job=i,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
x=np.asarray(chunks);W,M,S,H=x.sum(axis=0);ratio=H/S
se=float(np.sqrt(50/49*np.sum(((x[:,3]-ratio*x[:,2])/S)**2)))
net=x[:,3]-x[:,2]
r=dict(scope='Original complete HwU histograms; saved per-event streams were incomplete.',cuts='none',range_GeV=[5,20],source_pb=float(W),matched_pb=float(M),S_pb=float(S),H_pb=float(H),H_over_S=float(ratio),ratio_error=se,net_H_minus_S_pb=float(H-S),net_error_pb=float(np.sqrt(50/49*np.sum((net-net.mean())**2))),matched_source_minus1=float(M/W-1),rounding='Scalar sums inherit per-bin 8-digit HwU print rounding; final binary replay is checked independently against these original histograms.',bindings=records)
(B/'validation/original_hwu_summary.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='bindings'},indent=2))
