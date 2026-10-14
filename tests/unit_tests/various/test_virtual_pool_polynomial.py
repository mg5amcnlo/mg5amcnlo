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
        permitted = [{tuple(rows[i]['features']) for i in range(len(rows)) if folds[i]!=fold}
                     for fold in range(3)]
        earlier = poly.forward_split(rows)
        permitted.append({tuple(rows[i]['features']) for i in range(len(rows)) if earlier[i]})
        def checked(x,b,y,w,*args):
            selected = {tuple(row) for row in x}
            if len(x) != len(rows):
                self.assertIn(selected,permitted)
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
                self.assertLessEqual(candidate['forward_l1'],report['forward_validation']['baseline_l1'])

    def test_forward_validation_keeps_whole_iterations_and_workers_separate(self):
        rows=self.records()
        earlier=poly.forward_split(rows)
        assignments={}
        for row,selected in zip(rows,earlier):
            key=(row['directory'],row['channel'],row['iteration'])
            self.assertEqual(assignments.setdefault(key,selected),selected)
        self.assertTrue(all(earlier[i] for i,r in enumerate(rows) if r['iteration']<3))
        self.assertFalse(any(earlier[i] for i,r in enumerate(rows) if r['iteration']==3))

    def test_born_noise_fit_generalizes_with_large_signed_integration_weights(self):
        rng=np.random.default_rng(928)
        x=rng.uniform(-1,1,(1500,2))
        born=np.exp(rng.normal(0,2,len(x)))
        born[::101]=0.;born[1::71]*=-1
        virtual=born*(.4+.6*x[:,0]-.2*x[:,1]+rng.normal(0,.3,len(x)))
        weight=np.ones(len(x))/len(x)
        other=rng.uniform(-1,1,(1000,2))
        truth=.4+.6*other[:,0]-.2*other[:,1]
        errors={}
        for loss in ('square','born_square','born_huber'):
            entry=poly.fit(x,born,virtual,weight,1,1e-4,loss)
            prediction=poly.predict(entry,other,np.ones(len(other)),np.zeros(len(other)))
            errors[loss]=np.mean(np.abs(prediction-truth))
            zero=poly.predict(entry,other,np.zeros(len(other)),np.zeros(len(other)))
            np.testing.assert_array_equal(zero,0.)
        self.assertLess(errors['born_square'],.35*errors['square'])
        self.assertLess(errors['born_huber'],.05)

    def test_noisy_baseline_does_not_hide_resolved_quadratic_improvement(self):
        rows=self.records()
        # A noisy comparator makes the improvement over it uncertain. It does
        # not make the two fitted approximations mutually indistinguishable.
        comparator=np.full(len(rows),1000.)
        comparator[49]=50000.
        with mock.patch.object(poly,'mint_predictions',return_value=comparator):
            entry,report=poly.fit_order(rows,pool._folds(rows,3),self.metadata,1)
        self.assertEqual(entry['degree'],2)
        linear=[c for c in report['candidates'] if c['degree']==1 and c['eligible']]
        self.assertTrue(linear)
        self.assertLess(report['selected']['cv_rms'],.15*min(c['cv_rms'] for c in linear))

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
        self.assertIn(3,report['skipped_degrees'])
        self.assertTrue(all(c['parameters']<=poly.MAX_TERMS for c in report['candidates']))

    def test_cubic_is_selected_only_after_forward_and_matched_fold_validation(self):
        rows=self.records(2400)
        for row in rows:
            x=row['features']
            row['virtual']=row['born']*(.4+.7*x[0]*x[2]*x[3]+.3*x[4]**3)
        folds=pool._folds(rows,3)
        earlier=poly.forward_split(rows)
        allowed=[{tuple(rows[i]['features']) for i in range(len(rows)) if folds[i]!=fold}
                 for fold in range(3)]
        allowed.append({tuple(rows[i]['features']) for i in range(len(rows)) if earlier[i]})
        original=poly.fit
        cubic_calls=[]
        def checked(x,b,y,w,degree,*args):
            if degree==3:
                selected={tuple(row) for row in x}
                if len(x)!=len(rows): self.assertIn(selected,allowed)
                cubic_calls.append(len(x))
            return original(x,b,y,w,degree,*args)
        with mock.patch.object(poly,'fit',side_effect=checked):
            entry,report=poly.fit_order(rows,folds,self.metadata,1)
        self.assertEqual(entry['degree'],3)
        self.assertEqual(report['cubic_screen']['reason'],'passed')
        self.assertEqual(len(cubic_calls),5) # Forward, three CV fits, final fit.
        self.assertEqual(cubic_calls[-1],len(rows))
        self.assertTrue(report['selected']['forward_eligible'])
        self.assertLess(report['selected']['cv_rms'],.1*report['baseline_cv_rms'])

    def test_sparse_cubic_caps_coefficients_and_generalizes(self):
        rng=np.random.default_rng(9021)
        x=rng.uniform(-1,1,(2000,16));b=rng.uniform(.5,2.,len(x))
        b[::91]=0.;b[1::37]*=-1
        y=b*(.3+.6*x[:,0]*x[:,1]*x[:,2]+.3*x[:,3]**3)
        w=np.ones(len(x))/len(x)
        entry=poly.fit(x[:1500],b[:1500],y[:1500],w[:1500],3,.01,'born_square')
        self.assertEqual(len(entry['terms']),poly.MAX_TERMS)
        self.assertTrue(any(all(i>=0 for i in term['pair']) for term in entry['terms']))
        predicted=poly.predict(entry,x[1500:],b[1500:],np.zeros(500))
        self.assertLess(np.linalg.norm(predicted-y[1500:])/np.linalg.norm(y[1500:]),.2)
        np.testing.assert_array_equal(predicted[b[1500:]==0],0.)

    def test_cubic_budget_is_checked_before_matrix_allocation(self):
        rng=np.random.default_rng(472)
        x=rng.normal(size=(100,poly.MAX_CUBIC_FEATURES+1));v=np.ones(100)
        with mock.patch.object(poly,'design',side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError,'fit budget'):
                poly.fit(x,v,v,v,3,.01,'born_square')
            with mock.patch.object(poly,'MAX_CUBIC_DESIGN_ELEMENTS',100):
                with self.assertRaisesRegex(ValueError,'fit budget'):
                    poly.fit(x[:,:4],v,v,v,3,.01,'born_square')

    def test_relative_cubic_excludes_absolute_energy_at_every_degree(self):
        rng=np.random.default_rng(731)
        x=rng.uniform(-1,1,(800,10));x[:,1]=0.
        b=rng.uniform(.5,2.,len(x));y=b*(.5+.6*x[:,0]*x[:,2]*x[:,8])
        entry=poly.fit(x,b,y,np.ones(len(x)),3,.01,'born_square','relative')
        self.assertEqual(len(entry['lower']),10)
        self.assertTrue(all(9 not in term['pair'] for term in entry['terms']))
        self.assertTrue(any(8 in term['pair'] for term in entry['terms']))
        entry.update(order=1,blend=1.)
        metadata=dict(self.metadata,catalog=6,nf=10)
        path=self.root/'relative_model';path.write_text(pool._model_text(metadata,[entry]))
        model=pool.load_model(path)
        changed=x.copy();changed[:,-1]=100.
        first=pool.predict(model,1,x,b,np.zeros(len(x)))
        second=pool.predict(model,1,changed,b,np.zeros(len(x)))
        np.testing.assert_array_equal(first,second)

    def test_catalog6_compares_both_cubic_bases_under_the_same_guards(self):
        rows=self.records(2400);rng=np.random.default_rng(32)
        for row in rows:
            x=row['features']
            row['virtual']=row['born']*(.4+.7*x[0]*x[2]*x[3]+.3*x[4]**3)
            row['features'] += [0.,float(rng.uniform(-1,1))]
        metadata=dict(self.metadata,catalog=6,nf=10)
        entry,report=poly.fit_order(rows,pool._folds(rows,3),metadata,1)
        self.assertEqual(entry['degree'],3)
        self.assertEqual({c['basis'] for c in report['cubic_screens']},{'full','relative'})
        eligible=[c for c in report['candidates'] if c['degree']==3 and c['eligible']]
        self.assertEqual({c['basis'] for c in eligible},{'full','relative'})
        for candidate in eligible:
            self.assertLessEqual(candidate['cv_rms'],report['baseline_cv_rms'])
            self.assertTrue(candidate['forward_eligible'])

    def test_model_versions_keep_legacy_quadratics_and_validate_triples(self):
        entry=poly.empty_order(1,8,'test')
        entry.update(degree=2,blend=1.,terms=[dict(pair=(0,2),coefficient=.7)])
        text=pool._model_text(self.metadata,[entry])
        self.assertTrue(text.startswith('VIRTUAL_POOL_MODEL 2\n'))
        entry.update(degree=3,terms=[dict(pair=(0,2,4),coefficient=.7)])
        text=pool._model_text(self.metadata,[entry])
        path=self.root/'model';path.write_text(text)
        model=pool.load_model(path)
        self.assertEqual(model['metadata']['version'],3)
        self.assertEqual(pool._model_text(model['metadata'],model['orders']),text)
        for malformed in (text.replace('MODEL 3','MODEL 2'),
                          text.replace('1 3 5 ', '1 0 5 '),
                          text.replace('1 3 5 ', '1 3 9 '),
                          text.replace('1 3 5 ', '1 3 '),
                          text.replace('1 3\n', '1 2\n')):
            path.write_text(malformed)
            with self.assertRaises(ValueError):pool.load_model(path)

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
                        str(ROOT/'Template/NLO/SubProcesses/virtual_envelope_carrier.f90'),
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
                        str(ROOT/'Template/NLO/SubProcesses/virtual_envelope_carrier.f90'),
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
        self.assertNotEqual(actual[1,1],2.) # Bounded continuation, with exact correction.

    @unittest.skipUnless(shutil.which('gfortran'),'requires gfortran')
    def test_catalog6_model_reads_actual_loop_and_absolute_scale_in_both_runtimes(self):
        executable=self.root/'check'
        subprocess.run(['gfortran','-O2','-fcheck=all','-ffpe-trap=invalid,zero,overflow',
                        str(ROOT/'Template/NLO/SubProcesses/virtual_envelope_carrier.f90'),
                        str(ROOT/'Template/NLO/SubProcesses/virtual_pooled.f90'),
                        str(ROOT/'tests/input_files/check_virtual_pooled.f90'),'-o',str(executable)],
                        cwd=self.root,check=True,capture_output=True)
        meta=dict(self.metadata,catalog=6,nf=10)
        entry=poly.empty_order(1,10,'test')
        entry.update(degree=1,blend=1.,lower=[-20.]*10,upper=[20.]*10,
                     terms=[dict(pair=(8,-1),coefficient=2.),
                            dict(pair=(9,-1),coefficient=3.)])
        text=pool._model_text(meta,[entry])
        (self.root/'virtual_pool_model.dat').write_text(text)
        model=pool.load_model(self.root/'virtual_pool_model.dat')
        (self.root/'virtual_pool_mode.dat').write_text('2\n')
        beta=np.sqrt(250.**2-173.**2)
        p=np.array([[250.,0,0,250.],[250.,0,0,-250.],
                    [250.,beta,0,0],[250.,-beta,0,0]])
        lines=['4 1 2','21 0 1','21 0 1','6 173 0','-6 173 0']
        features=[]
        for qes in (10000.,40000.):
            lines += ['10000 %.17e 2'%qes]+[' '.join(map(str,leg)) for leg in p]
            features.append([np.log(25.)]+[
                np.log(2*(p[i,0]*p[j,0]-p[i,1:]@p[j,1:])/250000.)
                for i in range(4) for j in range(i+1,4)]+[
                np.log(250000./173.**2),np.log(250000./qes),np.log(250000.)])
        (self.root/'batch_input.dat').write_text('\n'.join(lines)+'\n')
        run=subprocess.run([str(executable),'batch_predict6'],cwd=self.root,check=True,
                           capture_output=True,text=True)
        actual=np.array([list(map(float,line.split())) for line in run.stdout.splitlines()])[:,1]
        expected=pool.predict(model,1,np.asarray(features),np.ones(2),np.full(2,2.))
        np.testing.assert_allclose(actual,expected,rtol=3.e-14)
        self.assertNotEqual(actual[0],actual[1])

    @unittest.skipUnless(shutil.which('gfortran'),'requires gfortran')
    def test_native_cubic_matches_python_with_mixed_orders_and_exact_tails(self):
        executable=self.root/'cubic'
        subprocess.run(['gfortran','-O2','-fno-automatic','-fcheck=all',
                        '-ffpe-trap=invalid,zero,overflow',
                        str(ROOT/'Template/NLO/SubProcesses/virtual_envelope_carrier.f90'),
                        str(ROOT/'Template/NLO/SubProcesses/virtual_pooled.f90'),
                        str(ROOT/'tests/input_files/check_virtual_pooled.f90'),'-o',str(executable)],
                        cwd=self.root,check=True,capture_output=True)
        meta=dict(self.metadata,catalog=6,nf=10,norders=2)
        entry=poly.empty_order(1,10,'test')
        entry.update(degree=3,blend=.75,born_scale=2.,target_scale=3.,
                     lower=[-20.]*10,upper=[20.]*10,
                     terms=[dict(pair=(-1,-1,-1),coefficient=.4),
                            dict(pair=(0,0,0),coefficient=.7),
                            dict(pair=(2,8,9),coefficient=-1.2)])
        quadratic=poly.empty_order(2,10,'test')
        quadratic.update(degree=2,blend=1.,lower=[-20.]*10,upper=[20.]*10,
                         terms=[dict(pair=(0,8),coefficient=.9)])
        text=pool._model_text(meta,[entry,quadratic])
        (self.root/'virtual_pool_model.dat').write_text(text)
        model=pool.load_model(self.root/'virtual_pool_model.dat')
        (self.root/'virtual_pool_mode.dat').write_text('2\n')
        q=np.sqrt(250.**2-173.**2)
        p=np.array([[250.,0,0,250.],[250.,0,0,-250.],
                    [250.,q,0,0],[250.,-q,0,0]])
        lines=['4 2 3','21 0 1','21 0 1','6 173 0','-6 173 0'];features=[]
        for mur2,qes2 in ((10000.,20000.),(40000.,90000.),(1e-30,1e-30)):
            lines+=['%.17e %.17e 2 -3'%(mur2,qes2)]+[' '.join(map(str,leg)) for leg in p]
            features.append([np.log(250000./mur2)]+[
                np.log(2*(p[i,0]*p[j,0]-p[i,1:]@p[j,1:])/250000.)
                for i in range(4) for j in range(i+1,4)]+[
                np.log(250000./173.**2),np.log(250000./qes2),np.log(250000.)])
        (self.root/'batch_input.dat').write_text('\n'.join(lines)+'\n')
        run=subprocess.run([str(executable),'batch_probability6'],cwd=self.root,check=True,
                           capture_output=True,text=True)
        actual=np.array([list(map(float,line.split())) for line in run.stdout.splitlines()])
        for order,mint in ((1,2.),(2,-3.)):
            expected=pool.predict(model,order,np.asarray(features),np.ones(3),np.full(3,mint))
            np.testing.assert_allclose(actual[:,order],expected,rtol=3e-14,atol=1e-15)
        np.testing.assert_array_equal(actual[:,-1],[.125,.125,1.])
        # Malformed triples fail before native inference, matching Python.
        for bad in (text.replace('3 9 10 ', '3 0 10 '),
                    text.replace('3 9 10 ', '3 9 11 '),
                    text.replace('3 9 10 ', '3 9 '),
                    text.replace('MODEL 3','MODEL 2')):
            (self.root/'virtual_pool_model.dat').write_text(bad)
            with self.assertRaises(ValueError):pool.load_model(self.root/'virtual_pool_model.dat')
            run=subprocess.run([str(executable),'batch_probability6'],cwd=self.root,
                               capture_output=True,text=True)
            self.assertNotEqual(run.returncode,0)
            self.assertIn('Pooled virtual:',run.stdout)


if __name__=='__main__':
    unittest.main()
