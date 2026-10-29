"""Numerical integration checks for the production Delta standalone bridge."""
import ctypes as C
import math
import numpy as np
from delta_evaluator import DeltaEvaluator, _array, _ptr


def main():
    mt=173.; energy=250.; p=math.sqrt(energy**2-mt**2)
    born=dict(p=[[250.,0.,0.,250.],[250.,0.,0.,-250.],
                 [250.,p,0.,0.],[250.,-p,0.,0.]],
              ids=[21,21,6,-6], colors=[[501,502],[503,501],[503,0],[0,502]],
              masses=[0.,0.,mt,mt],scalup=250.,x=[500/13000,500/13000])
    with DeltaEvaluator() as ev:
        # Independent common-cut product directly from the actual table.
        fn=ev.lib.pysudakov_; fn.restype=C.c_double
        masses=_array([.33,.33,.5,1.5,4.8,0,0,0,0,0,.000510998928,0,.1056583715,0,1.77682,0,0,0,0,0,0])
        def table(q,m,id,typ):
            args=[C.c_double(q),C.c_double(m),C.c_int(id),C.c_int(typ)]
            return fn(*[C.byref(a) for a in args],_ptr(masses))
        for q in [2.,5.,10.,20.,50.,100.,200.,249.,250.,400.]:
            r=ev.evaluate(born,q,lambda id,x,scale:scale**.2)
            manual=[]
            for i,lines in enumerate(r['connections']):
                p=math.prod(table(d['stop'],d['mass'],born['ids'][i],d['type'])/
                            table(d['start'],d['mass'],born['ids'][i],d['type']) for d in lines)
                if i<2 and lines: p*=(q/lines[0]['start'])**.2
                manual.append(min(max(p,0),1))
            if any(r['nonfinite_raw']):
                assert q==249. and r['legs'][:2]==[0.,0.],r
            else:
                assert np.max(np.abs(np.array(manual)-r['legs']))<1e-13,(q,manual,r)
        # Flavor sums must use sums of PDFs, not an arithmetic ratio average.
        bb=dict(born,ids=[1,-1,6,-6],colors=[[501,0],[0,502],[501,0],[0,502]],
                flavours=[[1,3,5],[-1,-3,-5],[6],[-6]])
        pdf=lambda id,x,q:abs(id)*q**(.01*abs(id))
        r=ev.evaluate(bb,20.,pdf)
        line=r['connections'][0][0]
        manual=table(20.,line['mass'],1,line['type'])/table(line['start'],line['mass'],1,line['type'])
        manual*=sum(pdf(id,bb['x'][0],20.) for id in [1,3,5])/sum(pdf(id,bb['x'][0],line['start']) for id in [1,3,5])
        assert abs(r['legs'][0]-min(manual,1))<1e-13
        # Independent native-scale analytic fixtures copied from established
        # regression expectations, exercising C/Fortran layout and labeling.
        rb=dict(p=[[500,0,0,500],[500,0,0,-500],[400,240,0,320],
                   [350,-210,0,-280],[50,-30,0,-40]],ids=[1,-1,2,-2,21],emitted=5)
        bc=dict(colors=[[501,0],[0,502],[501,0],[0,502]])
        stop,mass,dead=ev.real_scales(bc,rb)
        assert abs(stop[0,2]-math.sqrt(9000))<1e-11
        assert abs(mass[0,2]-math.sqrt(900000))<1e-11
        assert abs(mass[2,0]-300)<1e-11
        assert not dead[0,2]
        order=[0,1,4,2,3]
        remap=dict(p=[rb['p'][i] for i in order],ids=[rb['ids'][i] for i in order],emitted=3)
        s2,m2,d2=ev.real_scales(bc,remap)
        assert np.array_equal(s2,stop) and np.array_equal(m2,mass) and np.array_equal(d2,dead)
        # Imported LHE scales have priority even when post-reader momenta
        # would give slightly different native FSR limits.
        native_start=ev.starting_scales(born)
        supplied=native_start.copy()
        supplied[2,1]*=.8
        assert np.array_equal(ev.starting_scales(dict(born,starts=supplied)),supplied)
        for invalid in [supplied[:3],np.full((4,4),float('nan')),np.zeros((4,4))]:
            try:ev.starting_scales(dict(born,starts=invalid))
            except ValueError:pass
            else:raise AssertionError('Invalid imported scales accepted')
        # Production connection selector against independent construction,
        # including dead zones, bounds, unequal starts, and duplicate gluons.
        rng=np.random.default_rng(1907)
        for colors,ids in [(born['colors'],born['ids']),
            ([[501,502],[502,501],[0,0],[0,0]],[21,21,25,25])]:
            fixture=dict(colors=colors,ids=ids)
            for _ in range(50):
                st=rng.choice([.8,10.,300.,8000.],(4,4))
                sp=rng.choice([.6,10.,400.,9000.],(4,4))
                ma=rng.uniform(100,500,(4,4));de=rng.random((4,4))<.3
                selected=ev.select_connections(fixture,st,sp,ma,de)
                for i,partners in enumerate(ev.connections(colors)):
                    live=sorted(j for j in partners if not de[i,j] and st[i,j]>=sp[i,j])
                    expected=[dict(partner=j+1,type=1 if i<2 else 2 if j>=2 else 4,
                        start=max(min(st[i,j],7000.),1.01),stop=min(sp[i,j],7000.),
                        mass=ma[i,j]) for j in live]
                    assert expected==selected[i],(expected,selected[i])
        print('PASS production table product, PDF flavor sums, cutoff endpoints, native scale ABI, emitted-leg remapping, imported scales and production connection selector')

if __name__=='__main__': main()
