"""Residual polynomial kernels and matched native MINT replay."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

from madgraph.various.virtual_pool import polynomial as poly

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(np is not None, 'requires numpy')
class PolynomialPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='polynomial_pool_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_numpy_only_solver_preserves_signed_rank_deficient_fits(self):
        rng = np.random.default_rng(624)
        coordinate = rng.uniform(-1, 1, 300)
        x = np.column_stack((coordinate, coordinate, np.ones(len(coordinate))))
        born = rng.normal(size=len(x))
        born[::17] = 0.
        virtual = born*(.4+.7*coordinate)
        weights = rng.uniform(.1, 2., len(x))
        expected = poly.fit(x, born, virtual, weights, 1, 1.e-4)
        with mock.patch.object(poly, '_positive_solver', return_value=None):
            actual = poly.fit(x, born, virtual, weights, 1, 1.e-4)
        np.testing.assert_allclose(
            poly.predict(actual, x, born, np.zeros(len(x))),
            poly.predict(expected, x, born, np.zeros(len(x))),
            rtol=1.e-11, atol=1.e-12)

    def test_mint_bulk_predictions_keep_bin_edges_and_empty_iterations(self):
        model = poly.Marginals(2, 4)
        rows = [dict(x=x, born=b, virtual=v) for x, b, v in (
            ([0., 1.], 1., 2.), ([1., 0.], -2., 3.),
            ([.5, .5], 0., 0.), ([.25, .75], 3., -1.))]
        for bins in (4, 8):
            model.resize(bins)
            model.update(rows)
            average, count = model.average.copy(), model.count.copy()
            model.update([])
            np.testing.assert_array_equal(model.average, average)
            np.testing.assert_array_equal(model.count, count)
            np.testing.assert_array_equal(model.predict_many(rows), [model.predict(r) for r in rows])

    def test_born_noise_fit_generalizes_with_large_signed_integration_weights(self):
        rng=np.random.default_rng(928)
        x=rng.uniform(-1,1,(1500,2))
        born=np.exp(rng.normal(0,2,len(x)))
        born[::101]=0.;born[1::71]*=-1
        virtual=born*(.4+.6*x[:,0]-.2*x[:,1]+rng.normal(0,.3,len(x)))
        weight=np.ones(len(x))/len(x)
        other=rng.uniform(-1,1,(1000,2))
        truth=.4+.6*other[:,0]-.2*other[:,1]
        entry=poly.fit(x,born,virtual,weight,1,1e-4)
        prediction=poly.predict(entry,other,np.ones(len(other)),np.zeros(len(other)))
        self.assertLess(np.mean(np.abs(prediction-truth)), .05)
        zero=poly.predict(entry,other,np.zeros(len(other)),np.zeros(len(other)))
        np.testing.assert_array_equal(zero,0.)

    @unittest.skipUnless(shutil.which('gfortran'),'requires gfortran')
    def test_marginal_replay_matches_literal_native_recursion_and_bin_doubling(self):
        source=(ROOT/'Template/NLO/SubProcesses/mint_module.f90').read_text()
        routines='\n'.join(re.search(r'^  subroutine '+name+r'\b.*?^  end subroutine '+name,
            source,re.M|re.S).group(0) for name in
            ('get_ave_virt','fill_ave_virt','regrid_ave_virt','double_ave_virt'))
        fixture='''module fixture
 implicit none
 integer,parameter :: ndim=5,ndimmax=5,nchans=1
 integer :: nint_used_virt=4,ichan=1
 real(8) :: average_virtual(0:0,1)=0d0
 real(8) :: ave_virt(8,5,0:0,1)=0d0,ave_virt_acc(8,5,0:0,1)=0d0,ave_born_acc(8,5,0:0,1)=0d0
 integer :: nvirt(8,5,0:0,1)=0,nvirt_acc(8,5,0:0,1)=0
contains
'''+routines+'''
end module
program check
 use fixture
 implicit none
 integer :: i,j,it,nb,n
 real(8) :: x(5),b,v
 open(10,file='observations')
 do it=1,3
  read(10,*)nb,n
  if(nb/=nint_used_virt)then
   call double_ave_virt(0)
   nint_used_virt=nb
  endif
  do i=1,n
   read(10,*)x(1:2),b,v
   call fill_ave_virt(x,0,v,b)
  enddo
  call regrid_ave_virt(0)
 enddo
 do i=1,2
  write(*,'(*(ES25.16E3,1X))')ave_virt(:,i,0,1)
  write(*,*)nvirt(:,i,0,1)
 enddo
end program
'''
        (self.root/'literal.f90').write_text(fixture)
        subprocess.run(['gfortran','-O2','-fcheck=all','-ffpe-trap=invalid,zero,overflow',
                        str(self.root/'literal.f90'),'-o',str(self.root/'literal')],cwd=self.root,
                        check=True,capture_output=True)
        rng=np.random.default_rng(3710);lines=[];model=poly.Marginals(2,4)
        for bins in (4,8,8):
            rows=[dict(x=rng.uniform(size=2),born=(-1 if i%5==0 else 1)*rng.uniform(.1,2),
                       virtual=rng.normal()) for i in range(32)]
            rows[0]['born']=0.;rows[0]['virtual']=0.
            lines += ['%d %d'%(bins,len(rows))]+[' '.join(map(str,[*r['x'],r['born'],r['virtual']])) for r in rows]
            model.resize(bins);model.update(rows)
        (self.root/'observations').write_text('\n'.join(lines)+'\n')
        run=subprocess.run([str(self.root/'literal')],cwd=self.root,check=True,capture_output=True,text=True)
        actual=np.array([list(map(float,line.split())) for line in run.stdout.splitlines()])
        np.testing.assert_allclose(actual[::2],model.average,rtol=3.e-14)
        np.testing.assert_array_equal(actual[1::2],model.count)


if __name__=='__main__':
    unittest.main()
