"""Matched MINT validation, bounded polynomial inference, and native parity."""
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

from madgraph.various import virtual_pool as pool
from madgraph.various import virtual_pool_polynomial as poly

ROOT = Path(__file__).resolve().parents[3]


@unittest.skipUnless(np is not None, 'requires numpy')
class PolynomialPoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='polynomial_pool_')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.metadata = dict(version=2, catalog=5, nf=8, norders=1, nlegs=4, nx=2,
                             pdgs=(21, 21, 6, -6), masses=(0., 0., 173., 173.), incoming=(1, 1, 0, 0))

    def records(self, n=600):
        rng = np.random.default_rng(4901)
        x = rng.uniform(-1, 1, (n, 8));x[:, 1] = 0.
        born = rng.uniform(.2, 3., n);born[::13] *= -1;born[::17] = 0
        virtual = born*(.2+x[:, 0]*x[:, 2]+.6*x[:, 3]**2)
        mintx = rng.uniform(size=(n, 2))
        return [dict(sample=i+1, channel=i%3+1, iteration=i//200+1, order=1,
                     probability=.5, born=born[i], virtual=virtual[i], baseline=1.e8,
                     features=x[i].tolist(), x=mintx[i].tolist(), bins=8,
                     directory='job', weight=1./n) for i in range(n)]

    def write_job(self, rows):
        directory = self.root/'job';directory.mkdir()
        header = 'VIRTUAL_POOL_SAMPLES 2\n5 8 1 4 2\n21 21 6 -6\n0 0 173 173\n1 1 0 0\n'
        text = []
        for r in rows:
            values = [r[k] for k in ('sample','channel','iteration','order','bins',
                                     'probability','born','virtual','baseline')]
            text.append(' '.join(str(v) for v in values+r['features']+r['x']))
        (directory/pool.SAMPLES).write_text(header+'\n'.join(text)+'\n')
        iterations = sorted({r['iteration'] for r in rows})
        (directory/pool.ITERATIONS).write_text(''.join('%d 1 400\n'%i for i in iterations))
        return directory

    def test_training_roundtrip_and_frozen_independent_prediction(self):
        rows = self.records();job = self.write_job(rows)
        report = pool.train_pool([job], self.root/'model')
        self.assertEqual(report['version'], 2)
        self.assertTrue(report['enabled'], report)
        self.assertNotIn('baseline_validation_limitation', report)
        model = pool.load_model(self.root/'model')
        self.assertEqual(model['orders'][0]['degree'], 2)
        other = self.records(850)[600:]
        x,b,y = (np.asarray([r[k] for r in other]) for k in ('features','born','virtual'))
        predicted = pool.predict(model,1,x,b,np.zeros(len(b)))
        self.assertLess(np.linalg.norm(predicted-y)/np.linalg.norm(y), .15)
        np.testing.assert_array_equal(predicted[b==0], 0.)
        self.assertEqual((self.root/'model').read_text(), pool._model_text(model['metadata'],model['orders']))

    def test_causal_baseline_cannot_influence_selection_and_folds_do_not_leak(self):
        rows = self.records()
        folds = np.arange(len(rows))%3
        original = poly.fit
        calls = []
        def checked(x,b,y,w,*args):
            selected = {tuple(row) for row in x}
            if len(x) != len(rows):
                fold = len(calls)%3
                expected = {tuple(rows[i]['features']) for i in range(len(rows)) if folds[i]!=fold}
                self.assertEqual(selected,expected)
            calls.append(len(x))
            return original(x,b,y,w,*args)
        with mock.patch.object(poly,'fit',side_effect=checked):
            first, report = poly.fit_order(rows,folds,self.metadata,1)
        self.assertGreater(first['blend'],0)
        for r in rows: r['baseline']=-1.e30
        second,_ = poly.fit_order(rows,folds,self.metadata,1)
        self.assertEqual(first,second)
        self.assertEqual(calls[-1],len(rows))
        for candidate in report['candidates']:
            if candidate['eligible']:
                self.assertLessEqual(candidate['cv_rms'],report['baseline_cv_rms'])

    def test_small_samples_and_zero_orders_use_mint(self):
        rows=self.records(40)
        entry,_=poly.fit_order(rows,[i%3 for i in range(len(rows))],self.metadata,1)
        self.assertEqual(entry['blend'],0)
        rows=self.records()
        for r in rows:r['virtual']=0.
        entry,_=poly.fit_order(rows,[i%3 for i in range(len(rows))],self.metadata,1)
        self.assertEqual(entry['reason'],'zero_born_or_virtual_order')

    def test_exact_mint_wins_even_with_bad_causal_predictions(self):
        rows=self.records()
        for r in rows:r['virtual']=2.*r['born']
        entry,report=poly.fit_order(rows,pool._folds(rows,3),self.metadata,1)
        self.assertEqual(entry['blend'],0)
        self.assertLess(report['baseline_cv_rms'],1.e-13)

    def test_weight_tail_cannot_alone_establish_significant_gain(self):
        rows=self.records()
        for i,r in enumerate(rows):r['weight']=1. if i==49 else 1.e-15
        entry,_=poly.fit_order(rows,np.arange(len(rows))%3,self.metadata,1)
        self.assertEqual(entry['blend'],0)

    def test_high_multiplicity_is_bounded_before_fit(self):
        rows=self.records()
        for i,r in enumerate(rows):r['features'] += [np.sin(i+j) for j in range(45)]
        metadata=dict(self.metadata,nf=53)
        _,report=poly.fit_order(rows,np.arange(len(rows))%3,metadata,1)
        self.assertIn(2,report['skipped_degrees'])
        self.assertTrue(all(c['parameters']<=poly.MAX_TERMS for c in report['candidates']))

    def test_malformed_coordinates_and_polynomial_rejected(self):
        rows=self.records(2);rows[0]['x'][0]=1.5
        job=self.write_job(rows)
        with self.assertRaisesRegex(ValueError,'coordinates'):
            pool.collect_samples([job])
        entry=poly.empty_order(1,8,'test')
        entry.update(blend=1.,degree=1,terms=[dict(pair=(-1,2),coefficient=1.)])
        (self.root/'model').write_text(pool._model_text(self.metadata,[entry]))
        with self.assertRaisesRegex(ValueError,'indices'):
            pool.load_model(self.root/'model')

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

    @unittest.skipUnless(shutil.which('gfortran'),'requires gfortran')
    def test_generic_native_invariants_for_massless_and_unequal_mass_states(self):
        executable=self.root/'coordinates'
        subprocess.run(['gfortran','-O2','-fcheck=all','-ffpe-trap=invalid,zero,overflow',
                        str(ROOT/'Template/NLO/SubProcesses/virtual_pooled.f90'),
                        str(ROOT/'tests/input_files/check_virtual_pooled.f90'),'-o',str(executable)],
                        cwd=self.root,check=True,capture_output=True)
        cases=[([21,2,21,2],[0.,0.,0.,0.]),
               ([21,21,1000001,1000021],[0.,0.,120.,200.])]
        for pdgs,mass in cases:
            with self.subTest(pdgs=pdgs):
                e3=(500.**2+mass[2]**2-mass[3]**2)/1000.
                q=np.sqrt(e3**2-mass[2]**2)
                p=np.array([[250.,0,0,250.],[250.,0,0,-250.],
                            [e3,q,0,0],[500.-e3,-q,0,0]])
                mur2=31000.
                lines=['4 '+str(mur2)]+[' '.join(map(str,[pdgs[i],mass[i],int(i<2),*p[i]]))
                                       for i in range(4)]
                (self.root/'coordinates_input.dat').write_text('\n'.join(lines)+'\n')
                run=subprocess.run([str(executable),'invariants'],cwd=self.root,
                                   check=True,capture_output=True,text=True)
                fields=run.stdout.split()
                expected=[np.log(250000./mur2)]+[
                    np.log(abs(2*(p[i,0]*p[j,0]-p[i,1:]@p[j,1:]))/250000.)
                    for i in range(4) for j in range(i+1,4)]+[
                    np.log(250000./m**2) for m in sorted(set(mass)) if m>0]
                self.assertEqual(list(map(int,fields[:2])),[5,len(expected)])
                np.testing.assert_allclose(list(map(float,fields[2:])),expected,rtol=3e-14,atol=1e-15)

    @unittest.skipUnless(shutil.which('gfortran'),'requires gfortran')
    def test_native_capture_and_polynomial_prediction_with_tail_probability(self):
        executable=self.root/'check'
        subprocess.run(['gfortran','-O2','-fcheck=all','-ffpe-trap=invalid,zero,overflow',
                        str(ROOT/'Template/NLO/SubProcesses/virtual_pooled.f90'),
                        str(ROOT/'tests/input_files/check_virtual_pooled.f90'),'-o',str(executable)],
                        cwd=self.root,check=True,capture_output=True)
        (self.root/'virtual_pool_mode.dat').write_text('1\n')
        subprocess.run([str(executable),'collect'],cwd=self.root,check=True,capture_output=True)
        meta,rows,_=pool.collect_samples([self.root])
        self.assertEqual((meta['version'],meta['catalog'],meta['nf'],meta['nx']),(2,5,8,2))
        self.assertEqual(len(rows),2) # First iteration was rejected.
        np.testing.assert_allclose([r['virtual'] for r in rows],[5.,-2.])
        self.assertEqual(rows[0]['x'],[.3,.6])
        entry=poly.empty_order(1,8,'test')
        entry.update(degree=2,blend=.75,lower=[-10.]*8,upper=[10.]*8,
                     terms=[dict(pair=(-1,-1),coefficient=.2),
                            dict(pair=(0,-1),coefficient=.5),
                            dict(pair=(0,2),coefficient=-.7)])
        metadata=dict(self.metadata,masses=(0.,0.,172.5,172.5))
        model=dict(metadata=metadata,orders=[entry])
        (self.root/'virtual_pool_model.dat').write_text(pool._model_text(metadata,[entry]))
        (self.root/'virtual_pool_mode.dat').write_text('2\n')
        beta=np.sqrt(250.**2-172.5**2)
        p=np.array([[250.,0,0,250.],[250.,0,0,-250.],
                    [250.,beta,0,0],[250.,-beta,0,0]])
        inputs=['4 1 2','21 0 1','21 0 1','6 172.5 0','-6 172.5 0']
        features=[]
        for mur2 in (10000.,1.e-12):
            inputs += [str(mur2)+' 2']+[' '.join(map(str,leg)) for leg in p]
            features.append([np.log(250000./mur2)]+[
                np.log(2*(p[i,0]*p[j,0]-p[i,1:]@p[j,1:])/250000.)
                for i in range(4) for j in range(i+1,4)]+[np.log(250000./172.5**2)])
        (self.root/'batch_input.dat').write_text('\n'.join(inputs)+'\n')
        run=subprocess.run([str(executable),'batch_probability'],cwd=self.root,check=True,capture_output=True,text=True)
        actual=np.array([list(map(float,line.split())) for line in run.stdout.splitlines()])
        expected=pool.predict(model,1,np.asarray(features),np.ones(2),np.full(2,2.))
        np.testing.assert_allclose(actual[:,1],expected,rtol=3.e-14)
        np.testing.assert_array_equal(actual[:,2],[.125,1.])
        self.assertEqual(actual[1,1],2.) # Out-of-domain MINT fallback.


if __name__=='__main__':
    unittest.main()
