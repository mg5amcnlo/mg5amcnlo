"""Independent kinematic/ABI checks for the no-cut campaign helper and HwU hook."""
from pathlib import Path
import ctypes as C
import importlib.util
import json
import math
import subprocess
import tempfile
import numpy as np
from jet_tools import observables,FIELDS

BASE=Path(__file__).resolve().parent

def check():
    mb=4.7;mw=80.419
    eb=math.sqrt(30.**2+40.**2+mb**2)
    born=np.array([[24,0,0,0,mw],[5,30,0,40,eb],[-5,-30,0,-40,eb]],dtype=float)
    b=observables(born)
    assert b[0]==0. and b[15]==0. and b[16]==0.
    assert b[17]==30. and b[18]==30.
    assert abs(b[4]-2*eb)<1e-12 and abs(b[1]-(mw+2*eb))<1e-12
    assert abs(b[12]-math.pi)<1e-15
    trials=[]
    for pt in (1.e-8,1.e-4,.1,1.,10.):
        point=born.copy();point[0,1]=-pt;point[0,4]=math.hypot(mw,pt)
        g=[21,pt,0,4*pt/3,5*pt/3]
        point=np.vstack([point,g]);v=observables(point)
        assert math.isclose(v[0],pt,rel_tol=1e-12,abs_tol=1e-14)
        assert math.isclose(v[15],pt,rel_tol=1e-12,abs_tol=1e-14)
        assert v[16]==0. and math.isclose(v[17],30.+pt,rel_tol=1e-12)
        # Azimuthal rotation preserves every observable and jet assignment.
        angle=.734;co,si=math.cos(angle),math.sin(angle)
        rotated=point.copy();rotated[:,1]=co*point[:,1]-si*point[:,2];rotated[:,2]=si*point[:,1]+co*point[:,2]
        assert np.allclose(observables(rotated),v,rtol=2e-13,atol=2e-13,equal_nan=True)
        # Longitudinal boosts preserve pT, masses and angular differences,
        # while all hard-object/system rapidities shift by the boost.
        boost=.8;ch,sh=math.cosh(boost),math.sinh(boost)
        shifted=point.copy();shifted[:,3]=ch*point[:,3]+sh*point[:,4];shifted[:,4]=sh*point[:,3]+ch*point[:,4]
        u=observables(shifted)
        for i in [2,5,7,9,11]:assert abs(u[i]-v[i]-boost)<1e-12
        indices=[i for i in range(len(v)) if i not in [2,5,7,9,11]]
        assert np.allclose(u[indices],v[indices],rtol=2e-12,atol=2e-12,equal_nan=True)
        trials.append(dict(radiation_pt=pt,system_pt=v[0],radiation_jet_pt=v[15],all_jet_pt=v[17]))
    # Check strict W,b,bbar object counting; this prevents hidden g->bb or
    # a decayed hard object from silently changing the observable definition.
    for invalid in [born[:2],np.vstack([born,born[1]])]:
        try:observables(invalid)
        except RuntimeError:pass
        else:raise AssertionError('Invalid object multiplicity accepted')
    with tempfile.TemporaryDirectory(prefix='wbb-analysis-abi-') as temporary:
        work=Path(temporary)
        (work/'context.f90').write_text('module HwU_wgts_info_len\ninteger,parameter::wgts_info_len=80\nend module\n')
        (work/'HEPMC.INC').write_text('      integer nmxhep,nhep,nevhep\n      parameter(nmxhep=4000)\n      integer isthep(nmxhep),idhep(nmxhep)\n      double precision phep(5,nmxhep),evwgt\n      common /test_hep/nhep,nevhep,isthep,idhep,phep,evwgt\n')
        (work/'reweight0.inc').write_text('      integer max_weight_shower\n      parameter(max_weight_shower=100)\n')
        source=(BASE/'py8an_HwU_wbb.f').read_text();(work/'hook.f').write_text(source)
        result=subprocess.run(['gfortran','-O2','-c','-ffixed-line-length-132','context.f90','hook.f'],cwd=work,capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    report=dict(passed=True,fields=FIELDS,tests=['massive Born kinematics','radiation jet excludes b/bbar','radiation retained to1e-8GeV without threshold','collinear gluon clustered with b in all-jet control','azimuthal rotation','longitudinal boost','strict object counting','Fortran HwU hook compilation'],trials=trials)
    (BASE/'analysis_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print('No-cut Wbb analysis checks passed:',len(report['tests']))

if __name__=='__main__':check()
