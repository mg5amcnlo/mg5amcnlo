"""Physics-object, covariance and strict-count checks for validation plots."""
import importlib.util
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

# The validation plots (docs/audits) need NumPy; MG5_aMC itself does not.
# Without it this module must still import, because the test collector
# imports every module, and the plot tests are skipped.
try:
    import numpy as np
except ImportError:
    np=None
HAVE_NUMPY=np is not None


ROOT=Path(__file__).resolve().parents[3]
if HAVE_NUMPY:
    SPEC=importlib.util.spec_from_file_location('macnlops_validation_plots',
        ROOT/'docs/audits/plot_macnlops_validation.py')
    PLOTS=importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name]=PLOTS
    SPEC.loader.exec_module(PLOTS)
else:
    PLOTS=None
try:
    import awkward,fastjet,vector
    HAVE_FASTJET=True
except ImportError:
    HAVE_FASTJET=False


def event(weight,kind,jet_pt=0.,sequence=1):
    particles=[(11,40.,0.,0.,40.),(-11,-40.,0.,0.,40.)]
    if jet_pt:
        particles.append((21,jet_pt,0.,0.,jet_pt))
    lines=['<event>','%d 1 %.17g %.17g .007 .118' % (len(particles),weight,100.+sequence)]
    for pdg,px,py,pz,energy in particles:
        lines.append('%d 1 1 2 0 0 %.17g %.17g %.17g %.17g 0 0 9' %
                     (pdg,px,py,pz,energy))
    lines.append('#aMCatNLO %d 3 1 1 2 100 0 -9 4 0 0 0 0 0 0' % (1 if kind=='S' else 2))
    lines.append('</event>')
    return '\n'.join(lines)+'\n'


def fixture(folder):
    weights=[2.,-1.,3.,-2.]
    kinds=['H','H','S','S']
    accepted=[True,False,False,True]
    for level in ('first','full'):
        blocks=[event(w,k,j,seq) for seq,(w,k,j) in enumerate(
            zip(weights,kinds,[10.,20.,0.,5.]),1)]
        (folder/(level+'.lhe')).write_text('<LesHouchesEvents>\n'+''.join(blocks)+'</LesHouchesEvents>\n')
    (folder/'reference.lhe').write_text('<LesHouchesEvents>\n'+''.join(event(.5,'S',j)
        for j in (0.,5.,10.,20.))+'</LesHouchesEvents>\n')
    (folder/'decisions.jsonl').write_text(''.join(json.dumps(dict(sequence=i,kind=k,accepted=a,acceptance=p))+'\n'
        for i,(k,a,p) in enumerate(zip(kinds,accepted,[1.,0.,.25,.75]),1)))
    for role,labels in (('primary',kinds),('reference',['S']*4)):
        rows=[]
        for i,kind in enumerate(labels,1):
            rows.append(dict(schema='macnlops-full-audit-v1',sequence=i,kind=kind,
                emissions=i,first_endpoint_strict=kind=='S',first_trial_headroom=100.,
                continuation_trial_headroom=1.,continuation_policy='native-pythia-veto-with-audited-clipping',
                global_recoil_mode=1,n_max_global_branch=1,
                continuation_candidate_cache='native; losing pre-first candidates retained',
                native_trial_warning_counts=[dict(message='native test warning',count=2)] if i==3 else []))
        (folder/(role+'_full_audit.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return dict(label='Drell–Yan fixture',flavours=5,expected_events=200000,
        primary=[dict(first='first.lhe',full='full.lhe',decisions='decisions.jsonl',full_audit='primary_full_audit.jsonl')],
        reference=[dict(first='reference.lhe',full='reference.lhe',full_audit='reference_full_audit.jsonl')])


def common_fixture(folder):
    config=fixture(folder)
    config.pop('reference')
    config.update(common_source_mode=True,source_construction='ordinary-mcatnlo-v1',signed_residuals=True)
    weights=[2.,-1.,-.5,3.,-2.,-.25]
    kinds=['H','H','H','S','S','S']
    accepted=[True,True,False,False,True,False]
    probabilities=[1.,.4,.4,.25,.75,.75]
    for level in ('first','full'):
        blocks=[event(w,k,j,seq) for seq,(w,k,j) in enumerate(
            zip(weights,kinds,[10.,20.,25.,0.,5.,7.]),1)]
        (folder/(level+'.lhe')).write_text('<LesHouchesEvents>\n'+''.join(blocks)+'</LesHouchesEvents>\n')
    (folder/'decisions.jsonl').write_text(''.join(json.dumps(dict(sequence=i,kind=k,accepted=a,acceptance=p))+'\n'
        for i,(k,a,p) in enumerate(zip(kinds,accepted,probabilities),1)))
    template=json.loads((folder/'primary_full_audit.jsonl').read_text().splitlines()[0])
    rows=[dict(template,sequence=i,kind=k,emissions=i,first_endpoint_strict=k=='S')
          for i,k in enumerate(kinds,1)]
    (folder/'primary_full_audit.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config['bins']={'jet_pt':[0.,100.]}
    return config


@unittest.skipUnless(HAVE_NUMPY,'NumPy is required by the validation plots')
class TestMacNLOPSPlotAlgebra(unittest.TestCase):
    def test_stock_no_emission_warning_is_recorded_without_filtering(self):
        row=dict(schema='macnlops-full-audit-v1',sequence=1,kind='S',emissions=0,
                 first_endpoint_strict=False,first_trial_headroom=1.,
                 continuation_trial_headroom=1.,
                 continuation_policy='native-pythia-with-recorded-warnings',
                 native_trial_warning_counts=[dict(message='weight above unity',count=1)])
        summary=PLOTS.new_full_audit()
        PLOTS.add_full_audit(summary,row,1,'S',Path('stock.jsonl'))
        self.assertEqual(summary['events'],1)
        self.assertEqual(summary['strict_s_events'],0)
        self.assertEqual(summary['events_with_warnings'],1)
        self.assertEqual(summary['emission_counts']['0'],1)

    def test_ratio_lines_have_no_artificial_zero_endpoints(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.patches import StepPatch

        weights=np.array([10.,-1.,10.,-1.,10.,-1.])
        kinds=np.array(['S','H']*3)
        accepted=np.array([True,False]*3)
        masks=PLOTS.stage_masks(weights,kinds,accepted,common_source=True)
        histogram=PLOTS.Histogram([0.,1.,2.,3.])
        histogram.fill(np.repeat([.5,1.5,2.5],2),weights,masks)
        checks=[]

        def inspect_save(figure,*args,**kwargs):
            for axis in figure.axes:
                if axis.get_ylabel() not in ('/ MC@NLO','/ unrejected source'):
                    continue
                curves=[patch for patch in axis.patches if isinstance(patch,StepPatch)]
                self.assertTrue(curves)
                for curve in curves:
                    self.assertIsNone(curve.get_data().baseline)
                    # Every actual ratio is between 1 and 10/9. No
                    # vertical return to zero belongs at either edge.
                    self.assertGreater(np.nanmin(curve.get_path().vertices[:,1]),.9)
                self.assertGreater(axis.get_ylim()[0],.5 if axis.get_ylabel()=='/ unrejected source' else .0)
                for collection in axis.collections:
                    bounds=collection.get_datalim(axis.transData)
                    if np.all(np.isfinite(bounds.get_points())):
                        self.assertLessEqual(axis.get_ylim()[0],bounds.ymin)
                        self.assertGreaterEqual(axis.get_ylim()[1],bounds.ymax)
                checks.append(axis.get_ylabel())

        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(Figure,'savefig',inspect_save):
            PLOTS.render_observable(Path(temporary),'zz',dict(common_source_mode=True,flavours=5),
                'first','ratio_endpoint_fixture',PLOTS.Observable(histogram.edges,'Fixture'),histogram,
                dict(totals={'input':{'events':6}}),True)
        self.assertIn('/ MC@NLO',checks)
        self.assertIn('/ unrejected source',checks)

    def test_ratio_autoscale_keeps_real_zero_and_uncertainty_extents(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig,axis=plt.subplots()
        try:
            ratio=np.array([1.,0.,1.])
            axis.stairs(ratio,[0.,1.,2.,3.],baseline=None)
            axis.fill_between([.5,1.5,2.5],[.9,-.2,.9],[1.3,.2,1.3])
            PLOTS.autoscale_ratio_axis(axis)
            self.assertLessEqual(axis.get_ylim()[0],-.2)
            self.assertGreaterEqual(axis.get_ylim()[1],1.3)
            np.testing.assert_array_equal(ratio,[1.,0.,1.])
        finally:
            plt.close(fig)

    def test_common_source_covariance_and_partial_negative_h_masks(self):
        weights=np.array([2.,-1.,-.5,3.,-2.,-.25])
        accepted=np.array([True,True,False,False,True,False])
        masks=PLOTS.stage_masks(weights,np.array(['H']*3+['S']*3),accepted,common_source=True)
        self.assertEqual(masks[PLOTS.STAGES.index('h_only')].tolist(),[True,True,False,True,True,True])
        self.assertEqual(masks[PLOTS.STAGES.index('H_negative_retained')].tolist(),[False,True,False,False,False,False])
        histogram=PLOTS.Histogram([0.,1.]);histogram.fill(np.full(6,.5),weights,masks)
        np.testing.assert_array_equal(histogram.sumw[0],histogram.sumw[1])
        np.testing.assert_array_equal(histogram.covariance[0],histogram.covariance[1])
        ratio,variance=PLOTS.ratio_with_variance(histogram.sumw[1],histogram.sumw[0],
            histogram.sumw2[1],histogram.sumw2[0],histogram.covariance[1,0])
        np.testing.assert_array_equal(ratio,[1.]);np.testing.assert_array_equal(variance,[0.])
        i=PLOTS.STAGES.index('macnlops');a,b=histogram.sumw[i,0],histogram.sumw[0,0]
        ratio,variance=PLOTS.ratio_with_variance(histogram.sumw[i],histogram.sumw[0],
            histogram.sumw2[i],histogram.sumw2[0],histogram.covariance[i,0])
        self.assertAlmostEqual(variance[0],np.sum(((weights*accepted*b-a*weights)/b**2)**2))

    def test_paired_ratio_covariance_matches_eventwise_derivatives(self):
        weights=np.array([2.,-1.,3.])
        kinds=np.array(['S']*3)
        accepted=np.array([True,True,False])
        masks=PLOTS.stage_masks(weights,kinds,accepted)
        histogram=PLOTS.Histogram([0.,1.])
        histogram.fill(np.array([.5,.5,.5]),weights,masks)
        i,j=PLOTS.STAGES.index('macnlops'),PLOTS.STAGES.index('input')
        a,b=histogram.sumw[i,0],histogram.sumw[j,0]
        ratio,variance=PLOTS.ratio_with_variance(histogram.sumw[i],histogram.sumw[j],
            histogram.sumw2[i],histogram.sumw2[j],histogram.covariance[i,j])
        derivatives=(weights*accepted*b-a*weights)/(b*b)
        self.assertAlmostEqual(ratio[0],.25)
        self.assertAlmostEqual(variance[0],np.sum(derivatives**2))
        independent=PLOTS.ratio_with_variance(histogram.sumw[i],histogram.sumw[j],
            histogram.sumw2[i],histogram.sumw2[j])[1][0]
        self.assertGreater(independent,variance[0])
        self.assertEqual(histogram.covariance[0,i,0],0.)

    def test_signed_zero_bins_and_flow_accounting(self):
        ratio,variance=PLOTS.ratio_with_variance(np.array([1.,-2.,0.]),
            np.array([0.,-4.,2.]),np.array([1.,4.,0.]),np.array([0.,16.,4.]))
        self.assertTrue(np.isnan(ratio[0]) and np.isnan(variance[0]))
        self.assertEqual(ratio[1],.5)
        self.assertEqual(ratio[2],0.)
        weights=np.array([2.,-1.,3.,4.,-5.,6.])
        masks=PLOTS.stage_masks(weights,np.array(['S']*6),np.ones(6,dtype=bool))
        histogram=PLOTS.Histogram([0.,1.,2.])
        histogram.fill(np.array([-math.inf,.5,2.,math.inf,np.nan,0.]),weights,masks)
        i=PLOTS.STAGES.index('input')
        count=int(np.sum(histogram.count[i]))+sum(int(histogram.flows[f][i,0])
              for f in ('underflow','overflow','missing'))
        total=np.sum(histogram.sumw[i])+sum(histogram.flows[f][i,1]
              for f in ('underflow','overflow','missing'))
        self.assertEqual(count,6)
        self.assertEqual(total,np.sum(weights))
        self.assertEqual(histogram.flows['zero'][i,0],1.)
        self.assertEqual(histogram.count[i,-1],1) # Rightmost edge is included.

    def test_rejection_mechanisms_are_distinct(self):
        masks=PLOTS.stage_masks(np.array([2.,-1.,3.,-2.]),np.array(['H','H','S','S']),
                                np.array([True,False,False,True]))
        self.assertEqual(masks[PLOTS.STAGES.index('h_only')].tolist(),[True,False,True,True])
        self.assertEqual(masks[PLOTS.STAGES.index('s_only')].tolist(),[True,True,False,True])
        self.assertEqual(masks[PLOTS.STAGES.index('macnlops')].tolist(),[True,False,False,True])
        self.assertEqual(masks[PLOTS.STAGES.index('H_total')].tolist(),[True,True,False,False])

    def test_only_floating_cancellation_is_masked_in_signed_log_display(self):
        histogram=PLOTS.Histogram([0.,1.,2.])
        weights=np.array([.1,.2,-.3,-1.e-9])
        masks=PLOTS.stage_masks(weights,np.array(['S']*4),np.ones(4,dtype=bool))
        histogram.fill(np.array([.5,.5,.5,1.5]),weights,masks)
        i=PLOTS.STAGES.index('input')
        self.assertNotEqual(histogram.sumw[i,0],0.)
        self.assertEqual(histogram.display_sumw[i,0],0.)
        self.assertEqual(histogram.display_sumw[i,1],-1.e-9)
        self.assertAlmostEqual(histogram.sumabsw[i,0],.6)

    def test_log_statistical_band_does_not_create_roundoff_decades(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # For one weighted event the exact lower statistical edge is zero.
        # Division and sqrt can leave it a positive floating-point ulp.
        w,width=.031834867158391,.157863764591
        value=np.array([w/width]);variance=np.array([w*w/width**2])
        fig,ax=plt.subplots()
        PLOTS.signed_curve(ax,np.array([1.,2.]),value,variance,'k','one event',True)
        ax.set_yscale('log')
        self.assertGreater(ax.get_ylim()[0],value[0]/100.)
        plt.close(fig)


@unittest.skipUnless(HAVE_FASTJET,'analysis venv with awkward, vector and fastjet is required')
class TestMacNLOPSPlotEvents(unittest.TestCase):
    def test_shared_source_parallel_export_matches_serial(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);processes={}
            for process in ('dy','zz'):
                directory=folder/process;directory.mkdir();config=common_fixture(directory)
                if process=='zz':
                    for level in ('first','full'):
                        path=directory/(level+'.lhe');rows=[]
                        for line in path.read_text().splitlines():
                            fields=line.split()
                            if len(fields)==13 and abs(int(fields[0]))==11:
                                fields[0]='23';line=' '.join(fields)
                            rows.append(line)
                        path.write_text('\n'.join(rows)+'\n')
                for chunk in config['primary']:
                    for key,value in list(chunk.items()):chunk[key]=str(directory/value)
                processes[process]=config
            path=folder/'datasets.json';path.write_text(json.dumps(dict(processes=processes)))
            serial=PLOTS.run_analysis(path,folder/'serial',allow_incomplete=True,render=False,workers=1)
            parallel=PLOTS.run_analysis(path,folder/'parallel',allow_incomplete=True,render=False,workers=2)
            self.assertEqual(serial,parallel)
            for process in processes:
                for level in ('first','full'):
                    with np.load(folder/'serial'/process/level/'histograms.npz') as a,np.load(folder/'parallel'/process/level/'histograms.npz') as b:
                        self.assertEqual(a.files,b.files)
                        for key in a.files:np.testing.assert_array_equal(a[key],b[key])

    def test_shared_ordinary_export_reads_one_cohort_and_keeps_signed_residuals(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config=common_fixture(folder)
            datasets=folder/'datasets.json';datasets.write_text(json.dumps(dict(processes=dict(dy=config))))
            with mock.patch.object(PLOTS,'read_lhe',wraps=PLOTS.read_lhe) as reader:
                summary=PLOTS.run_analysis(datasets,folder/'plots',allow_incomplete=True,render=False,batch_size=2)
            self.assertEqual(reader.call_count,2) # one cohort at each endpoint, no reference reread
            self.assertTrue(summary['common_source_mode']);self.assertEqual(summary['variance'],PLOTS.SHARED_VARIANCE)
            for level,data in summary['processes']['dy']['levels'].items():
                self.assertTrue(data['common_source_mode']);self.assertTrue(data['signed_residuals'])
                self.assertEqual(data['totals']['reference'],data['totals']['input'])
                self.assertEqual(data['totals']['input']['events'],6)
                self.assertEqual(data['totals']['H_negative_retained']['events'],1)
                self.assertEqual(data['totals']['H_negative_retained']['sumw'],-1.)
                self.assertEqual(data['totals']['macnlops']['sumw'],-1.)
                self.assertEqual(data['totals']['macnlops']['sumw2'],9.)
                self.assertEqual(data['totals']['macnlops']['negative_events'],2)
                self.assertEqual(data['totals']['h_only']['sumw'],1.75)
                self.assertEqual(data['chunks']['primary'][0]['path'],data['chunks']['reference'][0]['path'])
                self.assertTrue(data['chunks']['reference'][0]['derived_from_primary'])
                with np.load(folder/'plots/dy'/level/'histograms.npz') as archive:
                    np.testing.assert_array_equal(archive['jet_pt__ratio_input_to_reference'],[1.])
                    np.testing.assert_array_equal(archive['jet_pt__ratio_variance_input_to_reference'],[0.])
                    i=PLOTS.STAGES.index('H_negative_retained')
                    np.testing.assert_array_equal(archive['jet_pt__sumw'][i],[-1.])
            text=(folder/'plots/index.html').read_text()
            self.assertIn('one ordinary MC@NLO source',text)
            self.assertIn('Negative H / S retained',text)

    def test_shared_mode_rejects_ambiguous_cohorts_policy_and_positive_h_veto(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config=common_fixture(folder)
            for change,match in ((dict(reference=[dict(first='reference.lhe')]),'omit reference'),
                                 (dict(source_construction='resolved-convex-v1'),'ordinary-mcatnlo'),
                                 (dict(common_source_mode='true'),'boolean'),
                                 (dict(reference_complete=False),'completion flags')):
                with self.subTest(change=change),self.assertRaisesRegex(PLOTS.ValidationError,match):
                    PLOTS.analyze_process('dy',dict(config,**change),folder,('first',),True)
            with self.assertRaisesRegex(PLOTS.ValidationError,'Pure MAcNLOPS'):
                PLOTS.analyze_process('dy',dict(config,signed_residuals=False),folder,('first',),True)
            path=folder/'decisions.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0].update(accepted=False,acceptance=.5)
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(PLOTS.ValidationError,'Positive H event was rejected'):
                PLOTS.analyze_process('dy',config,folder,('first',),True)

    def test_parallel_process_exports_equal_serial_and_fail_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);configuration={}
            settings=folder/'settings';settings.mkdir()
            for process in ('dy','zz'):
                directory=folder/process;directory.mkdir()
                config=fixture(directory)
                if process=='zz':
                    for name in ('first.lhe','full.lhe','reference.lhe'):
                        lines=[]
                        for line in (directory/name).read_text().splitlines():
                            fields=line.split()
                            if len(fields)==13 and abs(int(fields[0]))==11:
                                fields[0]='23'
                                fields[9]=format(math.sqrt(sum(float(v)**2 for v in fields[6:9])+91.1876**2),'.17g')
                                fields[10]='91.1876';line=' '.join(fields)
                            lines.append(line)
                        (directory/name).write_text('\n'.join(lines)+'\n')
                for role in ('primary','reference'):
                    for chunk in config[role]:
                        for key,value in list(chunk.items()):
                            chunk[key]=os.path.relpath(directory/value,settings)
                configuration[process]=config
            datasets=settings/'datasets.json'
            datasets.write_text(json.dumps(dict(processes=configuration)))
            serial=PLOTS.run_analysis(datasets,folder/'serial',allow_incomplete=True,render=False,workers=1)
            parallel=PLOTS.run_analysis(datasets,folder/'parallel',allow_incomplete=True,render=False,workers=2)
            self.assertEqual(serial,parallel)
            self.assertEqual(list(parallel['processes']),['dy','zz'])
            self.assertEqual((folder/'serial/stage_totals.csv').read_bytes(),
                             (folder/'parallel/stage_totals.csv').read_bytes())
            for process in configuration:
                for level in ('first','full'):
                    a=np.load(folder/'serial'/process/level/'histograms.npz')
                    b=np.load(folder/'parallel'/process/level/'histograms.npz')
                    self.assertEqual(a.files,b.files)
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                    self.assertEqual((folder/'serial'/process/level/'range_flows.csv').read_bytes(),
                                     (folder/'parallel'/process/level/'range_flows.csv').read_bytes())
            # A failed source/full identity gate in one child must discard
            # all staged outputs, including another child's successful result.
            path=folder/'zz/full.lhe'
            path.write_text(path.read_text().replace('3 1 2 101','3 1 7 101',1))
            with self.assertRaisesRegex(PLOTS.ValidationError,'Plot worker failed'):
                PLOTS.run_analysis(datasets,folder/'failed_parallel',allow_incomplete=True,render=False,workers=2)
            self.assertFalse((folder/'failed_parallel').exists())

    def test_soft_jet_alias_resolves_subgev_bins_and_preserves_broad_tail(self):
        events=[PLOTS.Event(1.,'S',[(11,40.,0.,0.,40.),(-11,-40.,0.,0.,40.),
                                  (21,p,0.,0.,p)]) for p in (.25,.75,30.,51.)]
        values=PLOTS.observable_values(events,'dy')
        np.testing.assert_array_equal(values['jet_pt_soft'],values['jet_pt'])
        specs=PLOTS.observable_specs('dy')
        np.testing.assert_array_equal(specs['jet_pt_soft'].edges,np.linspace(0.,50.,101))
        self.assertEqual(specs['jet_pt'].edges[-1],600.)
        histogram=PLOTS.Histogram(specs['jet_pt_soft'].edges)
        masks=PLOTS.stage_masks(np.ones(4),np.array(['S']*4),np.ones(4,dtype=bool))
        histogram.fill(values['jet_pt_soft'],np.ones(4),masks)
        i=PLOTS.STAGES.index('input')
        self.assertEqual(histogram.count[i,0],1)
        self.assertEqual(histogram.count[i,1],1)
        self.assertEqual(histogram.flows['overflow'][i,0],1.)

    def test_massive_bottom_pair_differs_from_three_body_system(self):
        mb,mw=4.7,80.4
        e1,e2=math.sqrt(mb*mb+25.),math.sqrt(mb*mb+17.)
        sample=PLOTS.Event(1.,'S',[(24,-3.,-4.,7.,math.sqrt(mw*mw+74.)),
            (5,3.,0.,4.,e1),(-5,0.,4.,-1.,e2)])
        values=PLOTS.observable_values([sample],'wbb')
        self.assertEqual(values['pair_pt'][0],5.)
        self.assertEqual(values['system_pt'][0],0.)
        expected=math.sqrt(2*mb*mb+2*(e1*e2+4.))
        self.assertAlmostEqual(values['pair_mass'][0],expected)
        self.assertAlmostEqual(values['pair_y'][0],.5*math.log((e1+e2+3)/(e1+e2-3)))
        self.assertGreater(values['system_mass'][0],values['pair_mass'][0]+mw-1.)
        specs=PLOTS.observable_specs('wbb')
        self.assertIn('pair_mass',specs)
        self.assertNotIn('pair_mass',PLOTS.observable_specs('ttbar'))

    def test_jet_rapidity_requires_hard_cluster_but_pt_spectrum_is_uncut(self):
        events=[]
        for transverse in (5.,20.,25.):
            events.append(PLOTS.Event(1.,'S',[(11,40.,0.,0.,40.),
                (-11,-40.-transverse,0.,-5.,math.hypot(40.+transverse,5.)),
                (21,transverse,0.,5.,math.hypot(transverse,5.))]))
        values=PLOTS.observable_values(events,'dy')
        self.assertEqual(values['jet_pt'].tolist(),[5.,20.,25.])
        self.assertTrue(np.all(np.isnan(values['jet_y'][:2])))
        self.assertAlmostEqual(values['jet_y'][2],math.asinh(.2))
        weights=np.array([2.,3.,4.]);masks=PLOTS.stage_masks(weights,np.array(['S']*3),np.ones(3,dtype=bool))
        histogram=PLOTS.Histogram([-6.,6.]);histogram.fill(values['jet_y'],weights,masks)
        i=PLOTS.STAGES.index('input')
        self.assertEqual(histogram.flows['missing'][i,0],2)
        self.assertEqual(histogram.flows['missing'][i,1],5.)
        self.assertEqual(histogram.sumw[i,0],4.)
        # The all-parton Wbb requirement is applied after clustering: two
        # individually soft nearby b partons make a jet above20GeV.
        mb=4.7
        sample=PLOTS.Event(1.,'S',[(24,-18.,-1.,0.,math.sqrt(80.4**2+325.)),
            (5,12.,0.,0.,math.sqrt(mb*mb+144.)),
            (-5,11.,1.,0.,math.sqrt(mb*mb+122.)),(21,-5.,0.,0.,5.)])
        wbb=PLOTS.observable_values([sample],'wbb')
        self.assertTrue(np.isnan(wbb['jet_y'][0]))
        self.assertAlmostEqual(wbb['all_jet_y'][0],0.)
        self.assertGreater(wbb['all_jet_pt'][0],20.)

    def test_back_to_back_endpoint_and_finite_azimuthal_deficit(self):
        events=[PLOTS.Event(1.,'S',[(11,40.,0.,0.,40.),(-11,-40.,0.,0.,40.)]),
                PLOTS.Event(1.,'S',[(11,40.,0.,0.,40.),(-11,0.,40.,0.,40.)])]
        values=PLOTS.observable_values(events,'dy')
        self.assertEqual(values['delta_phi_deficit'][0],0.)
        self.assertAlmostEqual(values['delta_phi_deficit'][1],math.pi/2.)

    def test_bottom_jet_merging_radiation_definition_and_top_exclusion(self):
        sample=PLOTS.Event(1.,'S',[(5,20.,0.,0.,20.6),(-5,18.,1.,0.,18.7),
                                 (21,-10.,0.,0.,10.),(6,1000.,0.,0.,1015.)])
        all_pt,all_y=PLOTS.leading_jets([sample])
        rad_pt,rad_y=PLOTS.leading_jets([sample],exclude_bottom=True)
        self.assertAlmostEqual(all_pt[0],math.hypot(38.,1.))
        self.assertAlmostEqual(rad_pt[0],10.)
        self.assertAlmostEqual(all_y[0],0.)
        empty_pt,empty_y=PLOTS.leading_jets([PLOTS.Event(1.,'S',[])])
        self.assertEqual(empty_pt[0],0.)
        self.assertTrue(np.isnan(empty_y[0]))

    def test_smoke_exports_original_weights_and_neff_without_rescaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            config=fixture(folder)
            datasets=folder/'datasets.json'
            card=folder/'physics_card.txt';card.write_text('massive bottom = 4.7\n')
            integrals=folder/'source_integrals.json'
            integrals.write_text(json.dumps(dict(schema='macnlops-source-integration-estimates-v1',
                processes=dict(dy=dict(primary=dict(source_events_requested=200000,
                    declared_init=[dict(process_id=0,cross_section_pb=900.,integration_error_pb=12.)]))))))
            datasets.write_text(json.dumps(dict(processes=dict(dy=config),
                plan={'collider_energy_GeV':13000},notes=['Recorded continuation test.'],
                provenance=[dict(path=card.name,label='Physics card'),
                            dict(path=integrals.name,label='Native integration estimates')])) )
            output=folder/'plots'
            with mock.patch.object(PLOTS,'render_observable',return_value=['fixture']):
                summary=PLOTS.run_analysis(datasets,output,allow_incomplete=True,render=True,batch_size=2)
            result=summary['processes']['dy']['levels']['first']
            self.assertFalse(summary['complete'])
            self.assertTrue(summary['smoke'])
            self.assertEqual(result['totals']['input']['events'],4)
            self.assertEqual(result['totals']['input']['sumw'],2.)
            self.assertEqual(result['totals']['input']['sumw2'],18.)
            self.assertEqual(result['totals']['macnlops']['events'],2)
            self.assertEqual(result['totals']['macnlops']['sumw'],0.)
            self.assertEqual(result['totals']['macnlops']['effective_events'],0.)
            veto=result['veto_diagnostic']['total']
            self.assertEqual(veto['expected_accepted_weight'],1.25)
            self.assertEqual(veto['accepted_weight'],0.)
            self.assertEqual(veto['conditional_bernoulli_variance'],2.4375)
            # The conditional veto variance is a separate diagnostic; the
            # observed weighted-histogram variance remains sum(w_kept^2).
            self.assertEqual(result['totals']['macnlops']['sumw2'],8.)
            audit=summary['processes']['dy']['levels']['full']['continuation_audit']
            self.assertEqual(audit['primary']['strict_s_events'],2)
            self.assertEqual(audit['reference']['strict_s_events'],4)
            self.assertEqual(audit['primary']['events_with_warnings'],1)
            self.assertEqual(audit['primary']['native_trial_warning_counts']['native test warning'],2)
            self.assertEqual(audit['primary']['settings']['continuation_trial_headroom'],[1.])
            self.assertEqual(audit['primary']['settings']['global_recoil_mode'],[1])
            self.assertEqual(audit['primary']['raw_audits'][0]['sha256'],
                hashlib.sha256((folder/'primary_full_audit.jsonl').read_bytes()).hexdigest())
            self.assertIn('SMOKE / INCOMPLETE',(output/'index.html').read_text())
            report=summary['provenance'][0]
            self.assertEqual((output/report['path']).read_bytes(),card.read_bytes())
            self.assertEqual(report['sha256'],hashlib.sha256(card.read_bytes()).hexdigest())
            self.assertIn('Recorded continuation test.',(output/'index.html').read_text())
            self.assertEqual(summary['native_source_integrals']['processes']['dy']['primary']
                ['declared_init'][0]['integration_error_pb'],12.)
            self.assertIn('900.0,12.0',(output/'source_integrals.csv').read_text())
            self.assertEqual(result['totals']['input']['sumw'],2.) # Not native estimate900.
            data=np.load(output/'dy/first/histograms.npz')
            self.assertIn('jet_pt__covariance',data)
            i=PLOTS.STAGES.index('input')
            self.assertAlmostEqual(np.sum(data['jet_pt__sumw'][i]),2.)
            self.assertEqual(data['jet_pt__count'][i].sum(),4)
            with (output/'dy/first/range_flows.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            flow=next(row for row in rows if row['observable']=='jet_y' and row['stage']=='input')
            self.assertEqual(int(flow['source_events']),4)
            self.assertEqual(int(flow['missing_events']),4)
            self.assertEqual(float(flow['missing_sumw_pb']),2.)
            self.assertEqual(float(flow['visible_fraction']),0.)
            self.assertIn('Histogram range accounting',(output/'index.html').read_text())
            self.assertIn('dy/first/range_flows.csv',(output/'index.html').read_text())

    def test_production_count_and_first_full_weights_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config=fixture(folder)
            with self.assertRaisesRegex(PLOTS.ValidationError,'expected 200000'):
                PLOTS.analyze_process('dy',config,folder,('first',),False,batch_size=2)
            text=(folder/'full.lhe').read_text().replace('4 1 2 ', '4 1 7 ',1)
            # Change the first event's weight independently of its objects.
            text=text.replace('3 1 2 101','3 1 7 101',1)
            (folder/'full.lhe').write_text(text)
            with self.assertRaisesRegex(PLOTS.ValidationError,'first/full event weights or source identities differ'):
                PLOTS.analyze_process('dy',config,folder,('first','full'),True,batch_size=2)
            datasets=folder/'datasets.json'
            datasets.write_text(json.dumps(dict(processes=dict(dy=config))))
            with self.assertRaises(PLOTS.ValidationError):
                PLOTS.run_analysis(datasets,folder/'failed',allow_incomplete=True,render=False)
            self.assertFalse((folder/'failed').exists())

    def test_equal_weight_first_full_reordering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config=fixture(folder)
            config['primary'][0].pop('full_audit')
            # Equal event weights cannot establish event identity. The source
            # SCALUP/header and aMC row are preserved in both shower outputs.
            blocks=[event(1.,'S',jet_pt=float(i),sequence=i) for i in (1,2)]
            for level,rows in (('first',blocks),('full',blocks[::-1])):
                (folder/(level+'.lhe')).write_text('<LesHouchesEvents>\n'+''.join(rows)+'</LesHouchesEvents>\n')
            (folder/'decisions.jsonl').write_text(''.join(json.dumps(dict(sequence=i,kind='S',accepted=True))+'\n'
                for i in (1,2)))
            with self.assertRaisesRegex(PLOTS.ValidationError,'source identities differ'):
                PLOTS.analyze_process('dy',config,folder,('first','full'),True,batch_size=2)

    def test_repeated_chunk_cannot_inflate_source_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config=fixture(folder)
            config['primary']*=2
            with self.assertRaisesRegex(PLOTS.ValidationError,'repeats a source chunk'):
                PLOTS.analyze_process('dy',config,folder,('first',),True,batch_size=2)

    def test_full_audit_identity_and_reference_continuation_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config=fixture(folder)
            path=folder/'reference_full_audit.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['sequence']=9
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(PLOTS.ValidationError,'Full audit event identity mismatch'):
                PLOTS.analyze_process('dy',config,folder,('full',),True,batch_size=2)
            rows[0]['sequence']=1
            for row in rows:row['continuation_trial_headroom']=10.
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(PLOTS.ValidationError,'continuation settings differ'):
                PLOTS.analyze_process('dy',config,folder,('full',),True,batch_size=2)

    def test_each_full_audit_record_must_declare_continuation_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config=fixture(folder)
            path=folder/'primary_full_audit.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            # An earlier valid declaration must not hide a later missing one.
            del rows[1]['continuation_policy']
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(PLOTS.ValidationError,'Missing native continuation settings'):
                PLOTS.analyze_process('dy',config,folder,('full',),True,batch_size=2)


if __name__ == '__main__':
    unittest.main()
