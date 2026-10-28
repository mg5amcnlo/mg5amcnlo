#!/usr/bin/env python3
"""Check raw-dummy versus colour-line factors in the native Delta consumer.

Usage: dipole_consumer_check.py OUTPUT_DIRECTORY
Only temporary copies of the raw PDFmode-1 archive are changed. The production
connection selector, leg probability, PDF callback ABI and table reader are
compiled verbatim by DeltaEvaluator. These are colour/kinematic fixtures, not
matrix-element or shower samples.
"""
import argparse
import ctypes as C
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from delta_evaluator import DeltaEvaluator, ROOT, _array, _ptr


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PATCHES = {
    'raw': '',
    'gluon_sqrt_all': '''      if(abs(id).eq.21)pysudakov=sqrt(max(tmp,0d0))
''',
    'per_colour_line': '''      if(itype.eq.2)then
        pysudakov=sqrt(max(tmp,0d0))
        if(abs(id).eq.21)pysudakov=sqrt(pysudakov)
      elseif(abs(id).eq.21)then
        pysudakov=sqrt(max(tmp,0d0))
      endif
''',
    'selected_default': None,
}


def fixture(topology, pdg):
    anti = pdg if pdg==21 else -pdg
    end = [501,502] if pdg==21 else [501,0]
    crossed = [502,501] if pdg==21 else [0,501]
    if topology=='II':
        ids = [pdg,anti,25]
        colors = [end,crossed,[0,0]]
        momenta = [[250,0,0,250],[250,0,0,-250],[500,0,0,0]]
        masses = [0,0,500]
    elif topology=='FI':
        ids = [pdg,22,pdg,22]
        colors = [end,[0,0],end,[0,0]]
        momenta = [[250,0,0,250],[250,0,0,-250],
                   [250,0,0,-250],[250,0,0,250]]
        masses = [0]*4
    else:
        assert topology=='FF'
        ids = [11,-11,pdg,anti]
        colors = [[0,0],[0,0],end,crossed]
        momenta = [[250,0,0,250],[250,0,0,-250],
                   [250,250,0,0],[250,-250,0,0]]
        masses = [0]*4
    n = len(ids)
    return dict(ids=ids,colors=colors,p=momenta,masses=masses,
                starts=[[200.]*n for _ in range(n)],scalup=200.,x=[.1,.1])


def table_ratio(evaluator, pdg, kind):
    f=evaluator.lib.pysudakov_
    f.restype=C.c_double
    f.argtypes=[C.POINTER(C.c_double),C.POINTER(C.c_double),
                C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_double)]
    masses=(C.c_double*21)(.33,.33,.5,1.5,4.8,*([0.]*16))
    def value(scale):
        q,m,i,t=C.c_double(scale),C.c_double(500),C.c_int(pdg),C.c_int(kind)
        return f(C.byref(q),C.byref(m),C.byref(i),C.byref(t),masses)
    return value(20)/value(200)


def direct_if_leg(evaluator, pdg):
    # The physical selector maps incoming IF connections onto typeII. This
    # explicit native-leg call checks the typeIF table contract independently.
    nline=2 if pdg==21 else 1
    partner=_array([3,3] if nline==2 else [3,0],int)
    kinds=_array([3,3] if nline==2 else [3,0],int)
    starts=_array([200.,200.] if nline==2 else [200.,0.])
    stops=_array([20.,20.] if nline==2 else [20.,0.])
    masses=_array([500.,500.] if nline==2 else [500.,0.])
    flavours=_array([pdg],int)
    value,raw=C.c_double(),C.c_double()
    evaluator.lib.delta_leg(1,pdg,1,_ptr(flavours),1,.1,nline,
        _ptr(partner),_ptr(kinds),_ptr(starts),_ptr(stops),_ptr(masses),
        C.byref(value),C.byref(raw))
    assert np.isfinite([value.value,raw.value]).all()
    return value.value,nline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'vendor/SudGen/sudakov_r515_PDFmode1.f.gz'
    source=gzip.decompress(archive.read_bytes()).decode()
    needle='      pysudakov=tmp\n'
    assert source.count(needle)==1
    rows=[];ratios={};hashes={};native_hashes=None
    with tempfile.TemporaryDirectory(prefix='delta_dipole_consumer_') as temp:
        work=Path(temp)
        for variant,patch in PATCHES.items():
            if patch is None:
                path=ROOT/'vendor/SudGen/sudakov.f'
            else:
                path=work/(variant+'.f')
                path.write_text(source.replace(needle,needle+patch))
            hashes[variant]=sha(path)
            with DeltaEvaluator(work=work/variant,table=path) as evaluator:
                native_hashes=evaluator.provenance['source_sha256']
                if variant=='raw':
                    ratios={str(kind):{str(pdg):table_ratio(evaluator,pdg,kind)
                            for pdg in [2,21]} for kind in [1,2,3,4]}
                for topology,kind in [('II',1),('FI',4),('FF',2)]:
                    for pdg in [2,21]:
                        b=fixture(topology,pdg)
                        result=evaluator.evaluate(b,20.,lambda *_:1.)
                        assert not any(result['nonfinite_raw'])
                        P=ratios[str(kind)][str(pdg)]
                        if topology=='II':
                            power=4 if pdg==21 and variant=='raw' else 2
                            measured=result['delta']
                            expected_counts=[2,2,0] if pdg==21 else [1,1,0]
                        elif topology=='FI':
                            power=2 if pdg==21 and variant=='raw' else 1
                            measured=result['fsr']
                            expected_counts=[2,0,2,0] if pdg==21 else [1,0,1,0]
                            PII=ratios['1'][str(pdg)]
                            assert np.isclose(result['delta'],(PII*P)**power,rtol=1e-12,atol=0)
                            assert result['connections'][0][0]['type']==1
                        else:
                            measured=result['delta']
                            expected_counts=[0,0,2,2] if pdg==21 else [0,0,1,1]
                            power=(1 if variant in ['per_colour_line','selected_default'] else
                                   4 if pdg==21 and variant=='raw' else 2)
                        expected=P**power
                        assert np.isclose(measured,expected,rtol=1e-12,atol=0)
                        counts=list(map(len,result['connections']))
                        assert counts==expected_counts
                        weighted=evaluator.evaluate(b,20.,lambda pid,x,scale:scale**.25)
                        pdf_power=.5 if topology=='II' else .25 if topology=='FI' else 0.
                        pdf_factor=(20./200.)**pdf_power
                        assert np.isclose(weighted['delta'],result['delta']*pdf_factor,
                                          rtol=1e-12,atol=0)
                        rows.append(dict(variant=variant,topology=topology,pdg=pdg,
                            raw_dummy_probability=P,quantity=('fsr' if topology=='FI' else 'event_delta'),
                            measured=measured,expected=expected,raw_dummy_power=power,
                            connections=result['connections'],counts=counts,legs=result['legs'],
                            total_delta=result['delta'],
                            nontrivial_PDF_total_delta=weighted['delta'],
                            expected_external_PDF_factor=pdf_factor))
                # Restore unit PDFs for the explicit typeIF call below.
                evaluator.evaluate(fixture('FF',2),20.,lambda *_:1.)
                for pdg in [2,21]:
                    value,count=direct_if_leg(evaluator,pdg)
                    P=ratios['3'][str(pdg)]
                    power=2 if pdg==21 and variant=='raw' else 1
                    assert np.isclose(value,P**power,rtol=1e-12,atol=0)
                    rows.append(dict(variant=variant,topology='IF',pdg=pdg,
                        raw_dummy_probability=P,quantity='explicit_native_leg',
                        measured=value,expected=P**power,raw_dummy_power=power,count=count,
                        note='Production selector uses typeII for incoming IF; this calls the native leg with typeIF explicitly.'))
    selected=ROOT/'vendor/SudGen/sudakov.f'
    for row in (r for r in rows if r['variant']=='selected_default'):
        reference=next(r for r in rows if r['variant']=='per_colour_line'
            and r['topology']==row['topology'] and r['pdg']==row['pdg'])
        assert row['measured']==reference['measured']
    data=dict(rows=rows,raw_table_ratios=ratios,patches=PATCHES,
        fixtures={name+'_'+str(pdg):fixture(name,pdg)
                  for name in ['II','FI','FF'] for pdg in [2,21]},
        transformed_source_sha256=hashes,native_source_sha256=native_hashes,
        selected_default=dict(path=str(selected),resolved_path=str(selected.resolve()),
            symlink_target=str(selected.readlink()) if selected.is_symlink() else None,
            sha256=sha(selected)),
        input_sha256={str(archive):sha(archive),str(Path(__file__)):sha(__file__),str(selected):sha(selected)},
        definition='q20GeV,Q200GeV,mdip500GeV; unit beam PDFs; rawmode1archive. '
            'Unmodified native get_delta_connections and delta_leg_probability compiled directly. '
            'Only isolated table copies are patched. Every final factor is consumed per directed colour line.',
        checks=['FFgg selects two colour lines per gluon even when both connect to the same partner',
                'Raw FFgg dummy probability P becomes event P^4; a gluon square root alone gives P^2; fourth root restores P',
                'Raw FFqqbar dummy probability P becomes event P^2; square root restores P',
                'FI gluon raw probability P becomes FSR P^2; square root restores P',
                'II gluon raw single-radiator P becomes two-gluon event P^4; square root restores P^2',
                'Incoming IF is mapped onto typeII by production selector; explicit typeIF native-leg contract checked separately',
                'Actual selected default agrees bitwise with independently patched per-colour-line archive in all8fixtures',
                'Nontrivial scale-dependent PDFs enter once per incoming leg for quarks and gluons in every table variant',
                'All quark and gluon fixture results agree with their expected powers within1e-12'],
        limitations=['Colour and kinematic fixtures only; no matrix elements or fresh showers',
                     'Common scales and symmetric massless dummy-end interpretation',
                     'Actual unequal scales and geometries retain the native weighted prescription'])
    (args.output/'consumer_check.json').write_text(json.dumps(data,indent=2)+'\n')
    for r in rows:
        print(r['variant'],r['topology'],r['pdg'],r['quantity'],r['measured'],f"P^{r['raw_dummy_power']}",flush=True)


if __name__=='__main__':main()
