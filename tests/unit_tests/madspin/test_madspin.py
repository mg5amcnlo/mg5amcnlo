################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################

"""Unit test library for the spin correlated decay routines
in the madspin directory"""

from __future__ import absolute_import
import sys
import os
import string
import shutil
pjoin = os.path.join

from subprocess import Popen, PIPE, STDOUT

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.insert(0, os.path.join(root_path,'..','..'))

import tests.unit_tests as unittest
import madgraph.interface.master_interface as Cmd
import madgraph.various.banner as banner

import copy
import array
import collections
import math

import madgraph.core.base_objects as MG
import madgraph.various.misc as misc
import MadSpin.decay as madspin
import madgraph.various.lhe_parser as lhe_parser
import MadSpin.interface_madspin as interface_madspin
import models.import_ufo as import_ufo


from madgraph import MG5DIR
#
class TestBanner(unittest.TestCase):
    """Test class for the reading of the banner"""

    def test_extract_info(self):
        """Test that the banner is read properly"""

        path=pjoin(MG5DIR, 'tests', 'input_files', 'tt_banner.txt')
        inputfile = open(path, 'r')
        mybanner = banner.Banner(inputfile)
#        mybanner.ReadBannerFromFile()
        process=mybanner.get("generate")
        model=mybanner.get("model")
        self.assertEqual(process,"p p > t t~ @1")
        self.assertEqual(model,"sm")
        
    
    def test_get_final_state_particle(self):
        """test that we find the final state particles correctly"""

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        fct = lambda x: cmd.get_final_part(x)
        
        # 
        self.assertEqual(set([11, -11]), fct('p p > e+ e-'))
        self.assertEqual(set([11, 24]), fct('p p > w+ e-'))
        self.assertEqual(set([11, 24]), fct('p p > W+ e-'))
        self.assertEqual(set([1, 2, 3, 4, -1, 11, 21, -4, -3, -2]), fct('p p > W+ e-, w+ > j j'))
        self.assertEqual(fct('p p > t t~, (t > b w+, w+ > j j) ,t~ > b~ w-'), set([1, 2, 3, 4, -1, 21, -4, -3, -2,5,-5,-24]))
        self.assertEqual(fct('e+ e- > all all, all > e+ e-'), set([-11,11]))
        self.assertEqual(fct('e+ e- > j w+, j > e+ e-'), set([-11,11,24]))

    def test_get_proc_with_decay_LO(self):

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        
        # Note the ; at the end of the line is important!
        #1 simple case
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, t> w+b  --no_warning=duplicate;'],[out])

        #2 with @0
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ @0', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~ , t> w+b @0 --no_warning=duplicate;'],[out])

        #3 with @0 and --no_warning=duplicate
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ @0 --no_warning=duplicate', 't> w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~ , t> w+b @0 --no_warning=duplicate;'],[out])

        #4 test with already present decay chain
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~, t > w+ b @0 --no_warning=duplicate', 't~ > w+b', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, t~ > w+b, ( t > w+ b , t~ > w+b) @0  --no_warning=duplicate;'],[out])
        
        #4 test with already present decay chain
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~, t > w+ b, t~ > w- b~ @0 --no_warning=duplicate', 'w >  all all', cmd._curr_model)
        self.assertEqual(['generate p p > t t~, w >  all all, ( t > w+ b, w >  all all), ( t~ > w- b~ , w >  all all) @0 --no_warning=duplicate;'],[out])

        #6 case with noborn=QCD
        # This is technically not yet supported by MS, but it is nice that this functions supports it.
        out = madspin.decay_all_events.get_proc_with_decay('generate g g > h QED=1 [noborn=QCD]', 'h > b b~', cmd._curr_model)
        self.assertEqual(['add process g g > h QED=1 [sqrvirt=QCD], h > b b~  --no_warning=duplicate;'], 
                         [out]) 

        # simple case but failing initial implementation. Handle it now but raising a critical message [mute here]
        with misc.MuteLogger(['decay'], [60]):
            out = madspin.decay_all_events.get_proc_with_decay('p p > t t~', 't~ > w- b~  QCD=99, t > w+ b  QCD=99', cmd._curr_model)
            self.assertEqual(['add process p p > t t~, t~ > w- b~  QCD=99, t > w+ b  QCD=99  --no_warning=duplicate;'],[out])
        
        self.assertRaises(Exception, madspin.decay_all_events.get_proc_with_decay, 'generate p p > t t~, (t> w+ b, w+ > e+ ve)')

    def test_get_proc_with_decay_NLO(self):

        cmd = Cmd.MasterCmd()
        cmd.do_import('sm')
        
        #1 simple case
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ [QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #2 simple case with QED=1
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #3 simple case with options
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [QCD] --test', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate --test',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate --test'],
                         out.split(';')[:-1])

        #4 case with LOonly
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD,QED]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])

        #5 case with LOonly=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [LOonly=QCD QED]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])
        
        
        #6 case with all=QCD
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [all=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [ all= QCD]', 't> w+b', cmd._curr_model)
         
        self.assertEqual(['add process p p > t t~ QED=1, t> w+b  --no_warning=duplicate',
                          'define pert_QCD = -4 -3 -2 -1 1 2 3 4 21',
                          'add process p p > t t~ pert_QCD QED=1, t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

        #6 case with virt=QCD, technically not valid but I like that the function can do it
        out = madspin.decay_all_events.get_proc_with_decay('generate p p > t t~ QED=1 [virt=QCD]', 't> w+b', cmd._curr_model)
        self.assertEqual(['add process p p > t t~ QED=1 [virt=QCD], t> w+b  --no_warning=duplicate'],
                         out.split(';')[:-1])       

          

class TestDensity(unittest.TestCase):
    """Test class for the reading of the lhe input file"""

    def test_get_density_mapping(self):
        """test the utility function that return the order of the density matrix-elements"""

        fct = madspin.DensityMatrix.get_map_density_matrix

        # check one single fermion case first
        out = fct([-1,1], n_changing=1)

        ordered_keys = list(out.keys())
        ordered_keys.sort()
        self.assertEqual(ordered_keys, [(-1, -1), (-1, 1), (1, -1), (1, 1)])

        self.assertEqual(out[(-1, -1)], (True, 0))
        self.assertEqual(out[(-1,  1)], (True, 1))    
        self.assertEqual(out[( 1, -1)], (False,1))
        self.assertEqual(out[(1, 1)], (True, 2))

        # check one massive boson next
        out = fct([-1,0,1], n_changing=1)

        ordered_keys = list(out.keys())
        ordered_keys.sort()
        self.assertEqual(ordered_keys, [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])

        self.assertEqual(out[(-1, -1)], (True, 0))
        self.assertEqual(out[(-1,  0)], (True, 1))    
        self.assertEqual(out[(-1, 1)],  (True ,2))
        self.assertEqual(out[(0, 0)], (True, 3))
        self.assertEqual(out[(0, 1)], (True,4))
        self.assertEqual(out[(1, 1)], (True, 5))

        self.assertEqual(out[(1,  0)], (False, 4))    
        self.assertEqual(out[(1, -1)], (False,2))


        # check a tt~ case
        out = fct([1,1,-1,1,1,-1,-1,-1], n_changing=2)
        ordered_keys = list(out.keys())
        ordered_keys.sort()
        self.assertEqual(len(ordered_keys), 16) 
        self.assertEqual(out[(1,1,1,1)], (True,0))
        self.assertEqual(out[(1,-1,1,1)], (True,1))
        self.assertEqual(out[(1,1,1,-1)], (True,2))
        self.assertEqual(out[(1,-1,1,-1)], (True,3))
        self.assertEqual(out[(-1,-1,1,1)], (True,4))
        self.assertEqual(out[(-1,1,1,-1)], (True,5))
        self.assertEqual(out[(-1,-1,1,-1)], (True,6))
        self.assertEqual(out[(1,1,-1,-1)], (True,7))
        self.assertEqual(out[(1,-1,-1,-1)], (True,8))
        self.assertEqual(out[(-1,-1,-1,-1)], (True,9))    

        nb_false = len([True for key in out if not out[key][0]])
        self.assertEqual(nb_false, 6)

class TestEvent(unittest.TestCase):
    """Test class for the reading of the lhe input file"""
    
    
    def test_madspin_event(self):
        """check the reading/writting of the events inside MadSpin"""
        
        inputfile = open(pjoin(MG5DIR, 'tests', 'input_files', 'madspin_event.lhe'))
        
        events = madspin.Event(inputfile)
        
        # First event
        event = events.get_next_event()
        self.assertEqual(event, 1)
        event = events
        self.assertEqual(event.string_event_compact(), """21 0.0 0 586.8395 586.84 0.7505772
21 0.0 0 -182.0876 182.0891 0.7488873
6 197.60403 48.42486 76.8186 277.8892 173
-6 -212.77359 -34.66934 359.4546 453.4437 173
21 15.169561 -13.75551 -31.52123 37.59628 0.7499895
""")
#        21 0.0 0.0 586.83954 586.84002    0.750577236977    
#21 0.0 0.0 -182.0876 182.08914    0.748887294316    
#6 197.60403 48.424858 76.818601 277.88922    173.00000459    
#-6 -212.77359 -34.669345 359.45458 453.44366    172.999981581    
#21 15.169561 -13.755513 -31.521232 37.59628    0.749989476383 
        self.assertEqual(event.get_tag(), (((21, 21), (-6, 6, 21)), [[21, 21], [6, -6, 21]]))   
        event.assign_scale_line("8 3 0.1 125 0.1 0.3")
        event.change_wgt(factor=0.4)
        
        self.assertEqual(event.string_event().split('\n'), """<event>
  8      3 +4.0000000e-02 1.25000000e+02 1.00000000e-01 3.00000000e-01
       21 -1    0    0  503  502 +0.00000000000e+00 +0.00000000000e+00 +5.86839540000e+02  5.86840020000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
       21 -1    0    0  501  503 +0.00000000000e+00 +0.00000000000e+00 -1.82087600000e+02  1.82089140000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
        6  1    1    2  504    0 +1.97604030000e+02 +4.84248580000e+01 +7.68186010000e+01  2.77889220000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       -6  1    1    2    0  502 -2.12773590000e+02 -3.46693450000e+01 +3.59454580000e+02  4.53443660000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       21  1    1    2  501  504 +1.51695610000e+01 -1.37555130000e+01 -3.15212320000e+01  3.75962800000e+01  7.50000000000e-01 0.0000e+00 0.0000e+00
#aMCatNLO 2  5  3  3  1 0.45933500E+02 0.45933500E+02 9  0  0 0.99999999E+00 0.69338413E+00 0.14872513E+01 0.00000000E+00 0.00000000E+00
  <rwgt>
   <wgt id='1001'>  +1.2946800e+02 </wgt>
   <wgt id='1002'>  +1.1581600e+02 </wgt>
   <wgt id='1003'>  +1.4560400e+02 </wgt>
   <wgt id='1004'>  +1.0034800e+02 </wgt>
   <wgt id='1005'>  +8.9768000e+01 </wgt>
   <wgt id='1006'>  +1.1285600e+02 </wgt>
   <wgt id='1007'>  +1.7120800e+02 </wgt>
   <wgt id='1008'>  +1.5316000e+02 </wgt>
   <wgt id='1009'>  +1.9254800e+02 </wgt>
</rwgt>
</event> 
""".split('\n'))
        
        # Second event
        event = events.get_next_event()    
        self.assertEqual(event, 1)
        event =events
        self.assertEqual(event.get_tag(), (((21, 21), (-6, 6, 21)), [[21, 21], [6, 21, -6]]))
                
        self.assertEqual(event.string_event().split('\n'), """<event>
  5     66 +3.2366351e+02 4.39615290e+02 7.54677160e-03 1.02860750e-01
       21 -1    0    0  503  502 +0.00000000000e+00 +0.00000000000e+00 +1.20582240000e+03  1.20582260000e+03  7.50000000000e-01 0.0000e+00 0.0000e+00
       21 -1    0    0  501  503 +0.00000000000e+00 +0.00000000000e+00 -5.46836110000e+01  5.46887540000e+01  7.50000000000e-01 0.0000e+00 0.0000e+00
        6  1    1    2  501    0 -4.03786550000e+01 -1.41924320000e+02 +3.66089980000e+02  4.30956860000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
       21  1    1    2  504  502 -2.46716450000e+01 +3.98371210000e+01 +2.49924260000e+02  2.54280130000e+02  7.50000000000e-01 0.0000e+00 0.0000e+00
       -6  1    1    2    0  504 +6.50503000000e+01 +1.02087200000e+02 +5.35124510000e+02  5.75274350000e+02  1.73000000000e+02 0.0000e+00 0.0000e+00
#aMCatNLO 2  5  4  4  4 0.40498390E+02 0.40498390E+02 9  0  0 0.99999997E+00 0.68201705E+00 0.15135239E+01 0.00000000E+00 0.00000000E+00
  <mgrwgt>
  some information
  <scale> even more infor
  </mgrwgt>
  <clustering>
  blabla
  </clustering>
  <rwgt>
   <wgt id='1001'> 0.32367e+03 </wgt>
   <wgt id='1002'> 0.28621e+03 </wgt>
   <wgt id='1003'> 0.36822e+03 </wgt>
   <wgt id='1004'> 0.24963e+03 </wgt>
   <wgt id='1005'> 0.22075e+03 </wgt>
   <wgt id='1006'> 0.28400e+03 </wgt>
   <wgt id='1007'> 0.43059e+03 </wgt>
   <wgt id='1008'> 0.38076e+03 </wgt>
   <wgt id='1009'> 0.48987e+03 </wgt>
  </rwgt>
</event> 
""".split('\n'))
        
        # Third event ! Not existing
        event = events.get_next_event()
        self.assertEqual(event, "no_event")
        



#class Testtopo(unittest.TestCase):
#    """Test the extraction of the topologies for the undecayed process"""
#
#    def test_topottx(self):
#
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'
#        path_for_me=pjoin(MG5DIR, 'tests','unit_tests','madspin')
#        shutil.copyfile(pjoin(MG5DIR, 'tests','input_files','param_card_sm.dat'),\
#		pjoin(path_for_me,'param_card.dat'))
#        curr_dir=os.getcwd()
#        os.chdir('/tmp')
#        temp_dir=os.getcwd()
#        mgcmd=Cmd.MasterCmd()
#        process_prod=" g g > t t~ "
#        process_full=process_prod+", ( t > b w+ , w+ > mu+ vm ), "
#        process_full+="( t~ > b~ w- , w- > mu- vm~ ) "
#        decay_tools=madspin.decay_misc()
#        topo=decay_tools.generate_fortran_me([process_prod],"sm",0, mgcmd, path_for_me)
#        decay_tools.generate_fortran_me([process_full],"sm", 1,mgcmd, path_for_me)
#
#        prod_name=decay_tools.compile_fortran_me_production(path_for_me)
#	decay_name = decay_tools.compile_fortran_me_full(path_for_me)
#
#
#        topo_test={1: {'branchings': [{'index_propa': -1, 'type': 's',\
#                'index_d2': 3, 'index_d1': 4}], 'get_id': {}, 'get_momentum': {}, \
#                'get_mass2': {}}, 2: {'branchings': [{'index_propa': -1, 'type': 't', \
#                'index_d2': 3, 'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 4,\
#                 'index_d1': -1}], 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}, \
#                   3: {'branchings': [{'index_propa': -1, 'type': 't', 'index_d2': 4, \
#                'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 3, 'index_d1': -1}],\
#                 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}}
#        
#        self.assertEqual(topo,topo_test)
#  
#
#        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03  \n'
#        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
#        p_string+='0.5000000E+03  0.1040730E+03  0.4173556E+03 -0.1872274E+03 \n'
#        p_string+='0.5000000E+03 -0.1040730E+03 -0.4173556E+03  0.1872274E+03 \n'        
#
#       
#        os.chdir(pjoin(path_for_me,'production_me','SubProcesses',prod_name))
#        executable_prod="./check"
#        external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
# 
#        external.stdin.write(p_string)
#
#        info = int(external.stdout.readline())
#        nb_output = abs(info)+1
#
#
#        prod_values = ' '.join([external.stdout.readline() for i in range(nb_output)])
#
#        prod_values=prod_values.split()
#        prod_values_test=['0.59366146660637686', '7.5713552297679376', '12.386583104018380', '34.882849897228873']
#        self.assertEqual(prod_values,prod_values_test)               
#        external.terminate()
#
#
#        os.chdir(temp_dir)
#        
#        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03 \n'
#        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
#        p_string+='0.8564677E+02 -0.8220633E+01  0.3615807E+02 -0.7706033E+02 \n'
#        p_string+='0.1814001E+03 -0.5785084E+02 -0.1718366E+03 -0.5610972E+01 \n'
#        p_string+='0.8283621E+02 -0.6589913E+02 -0.4988733E+02  0.5513262E+01 \n'
#        p_string+='0.3814391E+03  0.1901552E+03  0.2919968E+03 -0.1550888E+03 \n'
#        p_string+='0.5422284E+02 -0.3112810E+02 -0.7926714E+01  0.4368438E+02\n'
#        p_string+='0.2144550E+03 -0.2705652E+02 -0.9850424E+02  0.1885624E+03\n'
#
#        os.chdir(pjoin(path_for_me,'full_me','SubProcesses',decay_name))
#        executable_decay="./check"
#        external = Popen(executable_decay, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
#        external.stdin.write(p_string)
#
#        nb_output =1 
#        decay_value = ' '.join([external.stdout.readline() for i in range(nb_output)])
#
#        decay_value=decay_value.split()
#        decay_value_test=['3.8420345719455465E-017']
#        for i in range(len(decay_value)): 
#            self.assertAlmostEqual(eval(decay_value[i]),eval(decay_value_test[i]))
#        os.chdir(curr_dir)
#        external.terminate()
#        shutil.rmtree(pjoin(path_for_me,'production_me'))
#        shutil.rmtree(pjoin(path_for_me,'full_me'))
#        os.remove(pjoin(path_for_me,'param_card.dat'))
#        os.environ['GFORTRAN_UNBUFFERED_ALL']='n'


class _FakeDecayFile(object):
    """Minimal stand-in for a decay-pool EventFile: a list-backed iterator that
    also exposes .cross and .name, i.e. exactly the surface _StridedEvents wraps
    and get_decay_from_file reads."""
    def __init__(self, events, cross=1.0, name='fake'):
        self._it = iter(events)
        self._cross = cross
        self._name = name
    def __iter__(self):
        return self
    def __next__(self):
        return next(self._it)
    next = __next__
    @property
    def cross(self):
        return self._cross
    @property
    def name(self):
        return self._name


class TestStridedEvents(unittest.TestCase):
    """Unit tests for the parallel-MadSpin decay-pool striping helper.
    These are pure-Python (no matrix-element / physics), so they validate the
    lock-free disjoint-consumption invariant that the process-parallel
    unweighting relies on."""

    def _drain(self, strided):
        out = []
        while True:
            try:
                out.append(next(strided))
            except StopIteration:
                break
        return out

    def test_partition_is_disjoint_and_complete(self):
        """K workers striping over N events must together consume every event
        exactly once, with no overlap and no loss."""
        for n in (0, 1, 7, 100, 101):
            for k in (1, 2, 3, 5):
                events = list(range(n))
                collected = []
                for shard_id in range(k):
                    src = _FakeDecayFile(list(events))
                    collected.extend(self._drain(
                        interface_madspin._StridedEvents(src, shard_id, k)))
                self.assertEqual(sorted(collected), events,
                                 'n=%s k=%s partition wrong' % (n, k))

    def test_phase_offset(self):
        """Worker `offset` must yield events offset, offset+stride, ..."""
        events = list(range(20))
        for shard_id in range(4):
            src = _FakeDecayFile(list(events))
            got = self._drain(
                interface_madspin._StridedEvents(src, shard_id, 4))
            self.assertEqual(got, list(range(shard_id, 20, 4)))

    def test_single_worker_is_identity(self):
        """stride==1 must reproduce the full sequence unchanged."""
        events = list(range(13))
        src = _FakeDecayFile(list(events))
        got = self._drain(interface_madspin._StridedEvents(src, 0, 1))
        self.assertEqual(got, events)

    def test_cross_and_name_proxied(self):
        """Channel selection reads .cross; reopening reads .name."""
        src = _FakeDecayFile([1, 2, 3], cross=42.5, name='chan0')
        strided = interface_madspin._StridedEvents(src, 0, 2)
        self.assertEqual(strided.cross, 42.5)
        self.assertEqual(strided.name, 'chan0')

    def test_offset_beyond_end_is_empty(self):
        """A worker whose phase is past EOF yields nothing (over-sharding)."""
        src = _FakeDecayFile([0, 1])
        strided = interface_madspin._StridedEvents(src, 3, 4)
        self.assertEqual(self._drain(strided), [])



class TestDensityIdentity(unittest.TestCase):
    """DensityMatrix.identity / normalized: the primitives that let the
    accept/reject be done one decaying particle at a time.

    A particle whose decay has not been drawn yet contributes the average of its
    decay density matrix over the full decay phase space. Rotational invariance
    in the parent rest frame makes that average delta_{hh'}/n, so contracting
    the production density matrix against it must give exactly Tr(rho)/n."""

    # the helicity bases MadSpin builds (hel_dict, MG5 2S+1 convention)
    BASES = {1: [0], 2: [1, -1], 3: [-1, 0, 1]}

    def _random_density(self, hel, seed=0):
        """A hermitian-looking density matrix on that basis, in the packed
        upper-triangular storage the Fortran side produces."""
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(hel)
        arr = (rng.normal(size=n * (n + 1) // 2)
               + 1j * rng.normal(size=n * (n + 1) // 2)).astype('complex64')
        for i in range(n):  # a real diagonal, as a physical density matrix has
            arr[i * (2 * n - i + 1) // 2] = abs(arr[i * (2 * n - i + 1) // 2])
        return madspin.DensityMatrix(arr, 1, hel, n)

    def test_identity_is_flat_diagonal_of_unit_trace(self):
        """delta_{hh'}/n: unit trace, nothing off-diagonal."""
        import numpy as np
        for spin, hel in self.BASES.items():
            n = len(hel)
            I = madspin.DensityMatrix.identity(1, hel, n)
            self.assertAlmostEqual(I.trace().real, 1.0, places=6)
            self.assertEqual(int(np.count_nonzero(I._diag_mask)), n)
            self.assertTrue(np.allclose(I.values[~I._diag_mask], 0))
            self.assertTrue(np.allclose(I.values[I._diag_mask], 1.0 / n))

    def test_contraction_with_identity_is_the_trace(self):
        """The load-bearing property: <rho, I/n> == Tr(rho)/n."""
        import numpy as np
        for spin, hel in self.BASES.items():
            rho = self._random_density(hel, seed=spin)
            I = madspin.DensityMatrix.identity(1, hel, len(hel))
            self.assertTrue(np.allclose(I.scalar_multiplication(rho),
                                        rho.trace() / len(hel)))

    def test_identity_shares_the_cached_map(self):
        """Built like a real density matrix, so the scalar_multiplication fast
        path stays available instead of falling back to sorted alignment."""
        for spin, hel in self.BASES.items():
            rho = self._random_density(hel, seed=spin)
            I = madspin.DensityMatrix.identity(1, hel, len(hel))
            self.assertIsNotNone(I.map_density_matrix_ind)
            self.assertIs(I.map_density_matrix_ind, rho.map_density_matrix_ind)

    def test_normalized_has_unit_trace_and_keeps_direction(self):
        """Dhat = D/Tr(D) puts a drawn decay on the same footing as identity."""
        import numpy as np
        for spin, hel in self.BASES.items():
            rho = self._random_density(hel, seed=spin + 10)
            D = rho.normalized()
            self.assertAlmostEqual(D.trace().real, 1.0, places=5)
            # same matrix up to the scale
            self.assertTrue(np.allclose(D.values * rho.trace(), rho.values,
                                        rtol=1e-4, atol=1e-6))

    def test_scalar_parent_identity_equals_its_density(self):
        """A spin-0 parent has a 1x1 density matrix: normalized() and identity()
        coincide, so its accept/reject ratio is identically 1 -- it can never be
        rejected. This is why the pool ladder must not charge scalars."""
        import numpy as np
        hel = self.BASES[1]
        rho = self._random_density(hel, seed=3)
        I = madspin.DensityMatrix.identity(1, hel, 1)
        self.assertTrue(np.allclose(rho.normalized().values, I.values))


class TestSequentialOrdering(unittest.TestCase):
    """Ordering and pool sizing for the per-particle accept/reject.

    Spins are in the MG5 2S+1 convention: 1 = scalar, 2 = fermion, 3 = vector.
    """

    class _Stub(object):
        _sequential_spin_order = interface_madspin.MadSpinInterface._sequential_spin_order
        _decay_slot_order = interface_madspin.MadSpinInterface._decay_slot_order
        def __init__(self, order='2 3 1'):
            self.options = {'sequential_spin_order': order}

    def test_default_order_is_fermions_vectors_scalars(self):
        s = self._Stub()
        # slots: scalar, vector, fermion, fermion -> fermions, vector, scalar
        self.assertEqual(s._decay_slot_order([1, 3, 2, 2]), [2, 3, 1, 0])
        # scalar last even when it comes first in slot order
        self.assertEqual(s._decay_slot_order([1, 2]), [1, 0])

    def test_ties_broken_by_slot_index(self):
        """Same spin -> keep slot order, so a run is reproducible."""
        s = self._Stub()
        self.assertEqual(s._decay_slot_order([2, 2]), [0, 1])
        self.assertEqual(s._decay_slot_order([3, 3, 3]), [0, 1, 2])

    def test_order_is_configurable(self):
        """The hidden option allows A/B testing the ordering per process."""
        s = self._Stub('3 2 1')  # vectors first
        self.assertEqual(s._decay_slot_order([2, 3, 1]), [1, 0, 2])

    def test_unlisted_spin_goes_last_and_garbage_falls_back(self):
        self.assertEqual(self._Stub('2')._decay_slot_order([3, 2, 1]), [1, 0, 2])
        self.assertEqual(self._Stub('garbage')._decay_slot_order([2, 1, 3]), [0, 2, 1])
        self.assertEqual(self._Stub('')._sequential_spin_order(), [2, 3, 1])

    def test_ladder_grows_with_position(self):
        """1/eff_k: each slot sees a more polarised parent than the last."""
        ladder = interface_madspin.MadSpinInterface._decay_pool_ladder
        self.assertEqual([ladder(k, 2) for k in range(4)], [1.5, 2.0, 2.5, 3.0])
        self.assertEqual([ladder(k, 3) for k in range(4)], [1.5, 2.0, 2.5, 3.0])

    def test_scalar_is_never_charged_the_ladder(self):
        """A spin-0 parent can never be rejected (1x1 density matrix), so it
        burns exactly one decay event wherever it sits in the ordering."""
        ladder = interface_madspin.MadSpinInterface._decay_pool_ladder
        for position in range(4):
            self.assertEqual(ladder(position, 1), 1.1)


class TestDrawOneDecay(unittest.TestCase):
    """_draw_one_decay: drawing a single particle's decay, so the sequential
    accept/reject can redraw one particle without touching the others.

    get_decay_from_file is now a loop over it and must behave exactly as before.
    """

    class _Part(object):
        def __init__(self, pid):
            self.pid = pid
            self.pdg = pid
            self.status = 1

    class _Pool(object):
        """Stands in for an lhe_parser.EventFile of decay events."""
        def __init__(self, tag, n=50, cross=1.0):
            self.tag = tag
            self._it = iter(range(n))
            self.cross = cross
        def __next__(self):
            return '%s:%s' % (self.tag, next(self._it))

    class _Stub(object):
        get_decay_from_file = interface_madspin.MadSpinInterface.get_decay_from_file
        _draw_all_decays = interface_madspin.MadSpinInterface._draw_all_decays
        _draw_one_decay = interface_madspin.MadSpinInterface._draw_one_decay
        efficiency = 0.5

    def _setup(self):
        # t t~ t plus a gluon that never decays; two decay files for each pdg so
        # both the one-file-per-parent and the cross-section-weighted branches
        # are exercised.
        production = [self._Part(6), self._Part(-6), self._Part(6), self._Part(21)]
        evt_decayfile = {6: {0: self._Pool('t0'), 1: self._Pool('t1')},
                         -6: {0: self._Pool('tx0', cross=2.0),
                              1: self._Pool('tx1', cross=3.0)}}
        return production, evt_decayfile

    def test_non_decaying_particle_gives_none(self):
        production, evt_decayfile = self._setup()
        gluon = production[3]
        self.assertIsNone(self._Stub()._draw_one_decay(
            gluon, 3, [p.pid for p in production], evt_decayfile, 10))

    def test_empty_pool_dict_gives_none(self):
        production, evt_decayfile = self._setup()
        evt_decayfile[6] = {}
        self.assertIsNone(self._Stub()._draw_one_decay(
            production[0], 0, [p.pid for p in production], evt_decayfile, 10))

    def test_slots_come_in_production_order(self):
        """The density matrix slots are built in production order, so the draw
        must walk the particles in that same order."""
        production, evt_decayfile = self._setup()
        got = [(i, part.pid) for i, part, _ in
               self._Stub()._draw_all_decays(production, evt_decayfile, 10)]
        self.assertEqual(got, [(0, 6), (1, -6), (2, 6)])

    def test_identical_parents_read_their_own_file(self):
        """Two tops with two decay files: one file each, in order."""
        production, evt_decayfile = self._setup()
        out = self._Stub().get_decay_from_file(production, evt_decayfile, 10)
        self.assertEqual(out[6], ['t0:0', 't1:0'])

    def test_joint_path_is_unchanged(self):
        """get_decay_from_file must still consume the RNG in the same order and
        return the same draws as before the refactor."""
        import random
        for seed in range(50):
            random.seed(seed)
            production, evt_decayfile = self._setup()
            got = dict(self._Stub().get_decay_from_file(production, evt_decayfile, 10))
            random.seed(seed)
            production, evt_decayfile = self._setup()
            want = dict(self._reference(production, evt_decayfile))
            self.assertEqual(got, want)

    @staticmethod
    def _reference(production, evt_decayfile):
        """The implementation as it was before _draw_one_decay was extracted."""
        import random
        out = collections.defaultdict(list)
        particles = [p for p in production if int(p.status) == 1.0]
        ids = [particle.pid for particle in particles]
        for i, particle in enumerate(particles):
            if particle.pdg not in evt_decayfile:
                continue
            nb_decay = len(evt_decayfile[particle.pdg])
            if nb_decay == 0:
                continue
            if nb_decay == 1:
                decay_file = evt_decayfile[particle.pdg][0]
            elif ids.count(particle.pdg) == nb_decay:
                decay_file = evt_decayfile[particle.pdg][ids[:i].count(particle.pdg)]
            else:
                r = random.random()
                tot = sum(evt_decayfile[particle.pdg][k].cross
                          for k in evt_decayfile[particle.pdg])
                r = r * tot
                cumul = 0
                for j, events in evt_decayfile[particle.pdg].items():
                    cumul += events.cross
                    if r < cumul:
                        decay_file = events
                        break
                else:
                    raise Exception
            out[particle.pdg].append(next(decay_file))
        return out


class TestDrawOffshellMass(unittest.TestCase):
    """_draw_offshell_mass: one resonance virtuality, owned by the decay that
    carries it.

    The draw used to be a single pass over every decay, closing over a shared
    sqrt(shat) budget. The sequential accept/reject needs to draw one slot at a
    time and redraw a single mass on a reshuffling failure, so the budget is now
    passed in and out and the caller owns the order.
    """

    class _Val(object):
        def __init__(self, value):
            self.value = value

    class _Banner(object):
        def get(self, card, kind, pdg):
            return TestDrawOffshellMass._Val(173.0 if kind == 'mass' else 1.5)

    class _Dec(object):
        pass

    class _Stub(object):
        _draw_mass_value = interface_madspin.MadSpinInterface._draw_mass_value
        _draw_offshell_mass = interface_madspin.MadSpinInterface._draw_offshell_mass
        def __init__(self, bw_cut=-1):
            self.banner = TestDrawOffshellMass._Banner()
            self.options = {'BW_cut': bw_cut}

    def _reference(self, pdg, dec, budget, banner, options):
        """The inline draw block, mirroring the extracted _draw_mass_value: the
        Breit-Wigner sampling jacobian gap/pi with the atan argument divided by
        the width (not multiplied)."""
        pole = banner.get('param', 'mass', abs(pdg)).value
        width = banner.get('param', 'decay', abs(pdg)).value
        if options['BW_cut'] < 0:
            bw_cut = 15
        else:
            bw_cut = options['BW_cut']
        min_mass = pole - bw_cut * width
        max_mass = min(pole + bw_cut * width, budget)
        dec[0].new_mass = lhe_parser.Event.generate_random_mass(
                                    pole, width, min_mass, max_mass)
        dec[0].reshuffle_info = (pole, width, min_mass, max_mass)
        budget -= dec[0].new_mass
        gap = math.atan((pole ** 2 - min_mass ** 2) / pole / width)
        gap += math.atan((max_mass ** 2 - pole ** 2) / pole / width)
        return budget, gap / math.pi

    def test_identical_to_the_previous_inline_draw(self):
        """Same masses, same jacobians, same budget, same random sequence."""
        import random
        stub = self._Stub()
        for seed in range(50):
            random.seed(seed)
            budget, got = 500.0, []
            for _ in range(2):   # two resonances off one shared budget, as t t~
                dec = [self._Dec()]
                budget, jac = stub._draw_offshell_mass(6, dec, budget)
                got.append((dec[0].new_mass, jac, dec[0].reshuffle_info))
            random.seed(seed)
            ref_budget, want = 500.0, []
            for _ in range(2):
                dec = [self._Dec()]
                ref_budget, jac = self._reference(6, dec, ref_budget,
                                                  stub.banner, stub.options)
                want.append((dec[0].new_mass, jac, dec[0].reshuffle_info))
            self.assertEqual(got, want)
            self.assertEqual(budget, ref_budget)

    def test_mass_and_resample_info_land_on_the_decay(self):
        """The decay owns its mass: new_mass plus what is needed to redraw it."""
        import random
        random.seed(0)
        dec = [self._Dec()]
        _, jac = self._Stub()._draw_offshell_mass(6, dec, 500.0)
        self.assertTrue(hasattr(dec[0], 'new_mass'))
        pole, width, min_mass, max_mass = dec[0].reshuffle_info
        self.assertEqual((pole, width), (173.0, 1.5))
        self.assertTrue(min_mass <= dec[0].new_mass <= max_mass)
        self.assertTrue(jac > 0)

    def test_budget_shrinks_so_the_draw_is_order_dependent(self):
        """Each resonance eats into what the next one may take -- which is why
        the slot has to own the draw rather than inherit one pass over all."""
        import random
        random.seed(1)
        stub = self._Stub()
        dec = [self._Dec()]
        left, _ = stub._draw_offshell_mass(6, dec, 500.0)
        self.assertAlmostEqual(left, 500.0 - dec[0].new_mass)
        self.assertLess(left, 500.0)

    def test_bw_cut_option_is_honoured(self):
        """BW_cut < 0 means the default 15 widths; otherwise the option wins."""
        import random
        random.seed(2)
        dec = [self._Dec()]
        self._Stub(bw_cut=2)._draw_offshell_mass(6, dec, 500.0)
        pole, width, min_mass, max_mass = dec[0].reshuffle_info
        self.assertAlmostEqual(min_mass, 173.0 - 2 * 1.5)
        self.assertAlmostEqual(max_mass, 173.0 + 2 * 1.5)


class TestPartialDensityContraction(unittest.TestCase):
    """_partial_density_contraction: N_k, the production density matrix
    contracted with the decays drawn so far, the rest replaced by I/n.

    This is the heart of the per-particle accept/reject, so it is pinned against
    the two endpoints it has to reproduce and against the telescoping identity
    the method's exactness rests on.
    """

    class _Stub(object):
        _slot_identity = interface_madspin.MadSpinInterface._slot_identity
        _partial_density_contraction = \
            interface_madspin.MadSpinInterface._partial_density_contraction

    def _density(self, hel, seed):
        """A density matrix on a single-particle basis, packed as Fortran gives it."""
        import numpy as np
        rng = np.random.default_rng(seed)
        n = len(hel)
        arr = (rng.normal(size=n * (n + 1) // 2)
               + 1j * rng.normal(size=n * (n + 1) // 2)).astype('complex64')
        for i in range(n):
            arr[i * (2 * n - i + 1) // 2] = abs(arr[i * (2 * n - i + 1) // 2])
        return madspin.DensityMatrix(arr, 1, hel, n)

    def _production(self, hels, seed=7):
        """A production density matrix over the joint helicity index."""
        import numpy as np
        import itertools
        dim = 1
        for h in hels:
            dim *= len(h)
        allowed = []
        for combo in itertools.product(*hels):
            allowed.extend(combo)
        rng = np.random.default_rng(seed)
        arr = (rng.normal(size=dim * (dim + 1) // 2)
               + 1j * rng.normal(size=dim * (dim + 1) // 2)).astype('complex64')
        for i in range(dim):
            arr[i * (2 * dim - i + 1) // 2] = abs(arr[i * (2 * dim - i + 1) // 2])
        return madspin.DensityMatrix(arr, len(hels), allowed, dim), dim

    CASES = [[[1, -1], [1, -1]],            # t t~
             [[-1, 0, 1], [-1, 0, 1]],      # W W
             [[1, -1], [0]],                # fermion + scalar
             [[1, -1], [-1, 0, 1], [0]]]    # fermion + vector + scalar

    def test_nothing_drawn_is_the_trace(self):
        """N_0 = Tr(rho) / prod n_i: every slot contributes I/n."""
        import numpy as np
        for hels in self.CASES:
            rho, dim = self._production(hels)
            got = self._Stub()._partial_density_contraction(rho, hels, {})
            self.assertTrue(np.allclose(got, rho.trace() / dim))

    def test_everything_drawn_is_the_joint_contraction(self):
        """N_n = <rho, (x)_i Dhat_i>: the weight the joint accept/reject uses."""
        import numpy as np
        for hels in self.CASES:
            rho, _ = self._production(hels)
            densities = {i: self._density(h, 10 + i) for i, h in enumerate(hels)}
            got = self._Stub()._partial_density_contraction(rho, hels, densities)
            joint = None
            for i, h in enumerate(hels):
                d = densities[i].normalized()
                joint = d if joint is None else joint.tensor_product(d)
            self.assertTrue(np.allclose(got, joint.scalar_multiplication(rho)))

    def test_ratios_telescope_whatever_the_fill_order(self):
        """prod_k N_k/N_{k-1} == N_n/N_0 -- why the per-slot test targets the
        same distribution as the joint one, for any decay ordering."""
        import numpy as np
        hels = [[1, -1], [-1, 0, 1], [0]]
        rho, _ = self._production(hels, seed=3)
        densities = {i: self._density(h, 20 + i) for i, h in enumerate(hels)}
        stub = self._Stub()
        n_0 = stub._partial_density_contraction(rho, hels, {})
        n_n = stub._partial_density_contraction(rho, hels, densities)
        for order in ([0, 1, 2], [2, 1, 0], [1, 0, 2]):
            filled, previous, product = {}, n_0, 1.0
            for slot in order:
                filled[slot] = densities[slot]
                current = stub._partial_density_contraction(rho, hels, filled)
                product *= current / previous
                previous = current
            self.assertTrue(np.allclose(product, n_n / n_0))

    def test_scalar_slot_cannot_be_rejected(self):
        """Filling a spin-0 slot leaves N_k untouched: its ratio is exactly 1."""
        import numpy as np
        hels = [[1, -1], [-1, 0, 1], [0]]
        rho, _ = self._production(hels, seed=3)
        densities = {i: self._density(h, 20 + i) for i, h in enumerate(hels)}
        stub = self._Stub()
        without = stub._partial_density_contraction(rho, hels, {0: densities[0],
                                                                1: densities[1]})
        with_scalar = stub._partial_density_contraction(rho, hels, densities)
        self.assertTrue(np.allclose(with_scalar / without, 1.0))


class TestSequentialSlots(unittest.TestCase):
    """_decaying_pdgs / _sequential_slots: which density matrix slot belongs to
    which production particle.

    The sequential accept/reject must know this *before* drawing anything (it
    needs the basis to build N_0), and it must agree exactly with the slot order
    _density_basis lays out -- for pdg in decays_key, in production order -- or
    the tensor product and the production helicity index stop lining up.
    """

    class _Part(object):
        def __init__(self, pid, status=1):
            self.pid = pid
            self.pdg = pid
            self.status = status

    def _production(self):
        # t t~ t plus a gluon spectator, off an initial state
        return [self._Part(2, -1), self._Part(-2, -1), self._Part(6),
                self._Part(-6), self._Part(6), self._Part(21)]

    def _pools(self):
        # two files for the tops, one for the anti-top; 23 never appears
        return {6: {0: 'f', 1: 'f'}, -6: {0: 'f'}, 23: {0: 'f'}}

    def test_decays_key_is_first_appearance_order(self):
        """The order get_decay_from_file fills its dict in."""
        got = interface_madspin.MadSpinInterface._decaying_pdgs(
                        self._production(), self._pools())
        self.assertEqual(got, (6, -6))

    def test_pdg_without_a_pool_is_not_a_slot(self):
        """Same 'does this particle decay' test as _draw_one_decay."""
        production = self._production()
        interface = interface_madspin.MadSpinInterface
        self.assertEqual(interface._decaying_pdgs(production, {6: {}, -6: {0: 'f'}}),
                         (-6,))
        self.assertEqual(interface._decaying_pdgs(production, {}), ())

    def test_slots_match_the_basis_init_part(self):
        """The mapping must resolve to exactly the particles _density_basis
        puts in init_part, in the same order and as the same objects -- they are
        the parents the decays get boosted to."""
        production = self._production()
        interface = interface_madspin.MadSpinInterface
        decays_key = interface._decaying_pdgs(production, self._pools())
        particles, slots = interface._sequential_slots(production, decays_key)

        init_part = [part for pdg in decays_key for part in production
                     if part.pid == pdg and part.status == 1]
        self.assertEqual([particles[i].pid for i in slots],
                         [p.pid for p in init_part])
        for slot, index in enumerate(slots):
            self.assertIs(particles[index], init_part[slot])

    def test_slots_are_grouped_by_pdg_not_production_order(self):
        """t t~ t -> slots are (t, t, t~): grouped by pdg, production order
        inside a group. The indices therefore are not sorted."""
        production = self._production()
        interface = interface_madspin.MadSpinInterface
        decays_key = interface._decaying_pdgs(production, self._pools())
        particles, slots = interface._sequential_slots(production, decays_key)
        self.assertEqual(slots, [0, 2, 1])
        self.assertEqual([p.pid for p in particles], [6, -6, 6, 21])


class TestProductionJacobianForSlots(unittest.TestCase):
    """_production_jacobian_for: J_k, the production jacobian with the slots
    drawn so far offshell and the rest nominal. Each slot carries J_k/J_{k-1}.
    """

    EVENT = """ <event>
 12      1 +4.8368719e+02 1.76709900e+02 7.54677100e-03 1.17102600e-01
         2 -1    0    0  502    0 +0.0000000000e+00 +0.0000000000e+00 +1.6801959055e+02 1.6801959055e+02 0.0000000000e+00  0.0000e+00 1.0000e+00
        -2 -1    0    0    0  501 -0.0000000000e+00 -0.0000000000e+00 -3.6057100553e+02 3.6057100553e+02 0.0000000000e+00  0.0000e+00 -1.0000e+00
         6  2    1    2  502    0 -1.0742571918e+01 -3.4379861756e+01 -2.8025420328e+02 3.3131374285e+02 1.7300000000e+02  0.0000e+00 9.0000e+00
        -6  1    1    2    0  501 +1.0742571918e+01 +3.4379861756e+01 +8.7702788293e+01 1.9727685323e+02 1.7300000000e+02  0.0000e+00 9.0000e+00
         5  1    3    3  502    0 -6.3369583864e+00 +5.5362090397e+01 -7.6229914475e+01 9.4542096209e+01 4.7000000000e+00  0.0000e+00 -1.0000e+00
        24  1    3    3    0    0 -4.4056135319e+00 -8.9741952154e+01 -2.0402428881e+02 2.3677164665e+02 7.9761361725e+01  0.0000e+00 9.0000e+00
        </event>"""

    INFO = (173.0, 1.5, 150.0, 200.0)

    def _production(self):
        evt = lhe_parser.Event()
        evt.parse(self.EVENT)
        return evt

    def test_nothing_offshell_is_unit_jacobian(self):
        """J_0: no mass moved, so the reshuffling is the identity."""
        jac = interface_madspin.MadSpinInterface._production_jacobian_for(
                        self._production(), {0: 0}, {})
        self.assertAlmostEqual(jac, 1.0, places=6)

    def test_leaves_the_production_untouched(self):
        """J_k is needed at every slot; the reshuffling happens once, later."""
        production = self._production()
        before = str(production)
        interface_madspin.MadSpinInterface._production_jacobian_for(
                        production, {0: 0}, {0: (180.0, self.INFO)})
        self.assertEqual(str(production), before)

    def test_offshell_slot_changes_the_jacobian(self):
        """J_1 != J_0 once a virtuality is sampled -- the ratio is what the slot
        carries in its accept/reject weight."""
        interface = interface_madspin.MadSpinInterface
        j_0 = interface._production_jacobian_for(self._production(), {0: 0}, {})
        j_1 = interface._production_jacobian_for(self._production(), {0: 0},
                                                {0: (180.0, self.INFO)})
        self.assertNotAlmostEqual(j_1, j_0, places=3)
        self.assertTrue(0 < j_1 / j_0 < 2)

    def test_impossible_mass_set_is_reported(self):
        """The production-side kinematic failure: the caller trashes the set."""
        jac = interface_madspin.MadSpinInterface._production_jacobian_for(
                        self._production(), {0: 0},
                        {0: (1e6, (173.0, 1.5, 150.0, 2e6))})
        self.assertEqual(jac, -1)


class TestSequentialAcceptReject(unittest.TestCase):
    """sequential_accept_reject: accepting one decaying particle at a time must
    sample the *same* distribution as the joint accept/reject, i.e. p(decays)
    proportional to N_n.

    Driven with synthetic density matrices so no matrix element is needed. The
    decay pool has to be physical for the claim to hold: Dhat = (I + a n.sigma)/2
    for a spin-1/2 parent, and antipodal directions so the pool average is
    exactly I/2 -- the trace property the method rests on. (With a pool that
    violates it the method is genuinely biased; that is the caveat the opt-out
    flag exists for.)
    """

    POOL = 4
    HELS = [[1, -1], [1, -1]]      # t t~

    class _Part(object):
        def __init__(self, pid, status=1):
            self.pid = pid
            self.pdg = pid
            self.status = status

    class _Prod(list):
        sqrts = 1000.0

    def _fermion_decay(self, nhat, alpha=0.9):
        import numpy as np
        nx, ny, nz = nhat
        matrix = 0.5 * np.array([[1 + alpha * nz, alpha * (nx - 1j * ny)],
                                 [alpha * (nx + 1j * ny), 1 - alpha * nz]],
                                dtype=complex)
        arr = np.array([matrix[0, 0], matrix[0, 1], matrix[1, 1]],
                       dtype='complex64')
        return madspin.DensityMatrix(arr, 1, [1, -1], 2)

    def _pool(self, seed):
        import numpy as np
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(self.POOL // 2):
            vec = rng.normal(size=3)
            vec /= np.linalg.norm(vec)
            out.append(self._fermion_decay(vec))
            out.append(self._fermion_decay(-vec))
        return out

    def _production_density(self, seed=11, rank=3, dim=4):
        import numpy as np
        import itertools
        rng = np.random.default_rng(seed)
        matrix = np.zeros((dim, dim), dtype=complex)
        for _ in range(rank):
            vec = rng.normal(size=dim) + 1j * rng.normal(size=dim)
            matrix += np.outer(vec, vec.conj())
        arr = np.array([matrix[i, j] for i in range(dim) for j in range(i, dim)],
                       dtype='complex64')
        allowed = []
        for combo in itertools.product(*self.HELS):
            allowed.extend(combo)
        return madspin.DensityMatrix(arr, 2, allowed, dim)

    def _stub(self, rho, pools):
        interface = interface_madspin.MadSpinInterface
        hels = self.HELS
        pool = self.POOL

        class Stub(object):
            _decaying_pdgs = staticmethod(interface._decaying_pdgs)
            _sequential_slots = staticmethod(interface._sequential_slots)
            _slot_identity = interface._slot_identity
            _partial_density_contraction = interface._partial_density_contraction
            _sequential_spin_order = interface._sequential_spin_order
            _decay_slot_order = interface._decay_slot_order
            sequential_accept_reject = interface.sequential_accept_reject
            _scan_maxwgt_range = interface._scan_maxwgt_range
            _sequential_offshell = interface._sequential_offshell
            def __init__(self):
                self.options = {'spinmode': 'onshell',
                                'sequential_spin_order': '2 3 1',
                                'sequential_exact': False}
            def _density_basis(self, production, decays_key):
                particles, slots = interface._sequential_slots(production, decays_key)
                return {'decays_key': decays_key, 'helicities': hels,
                        'init_part': [particles[i] for i in slots],
                        'decaying_spins': [2, 2], 'position': [1, 2],
                        'allowed_hel': [], 'ncomb': 0, 'dimension': 4}
            def create_and_initialise_f2py_modules(self, all_prefix, all_pdg, all_procid):
                pass
            def get_density(self, *args, **opts):
                return rho
            def _draw_one_decay(self, particle, index, ids, evt_decayfile, nb_remain):
                import random
                return ('cand', self._slot_of[index], random.randrange(pool))
            def _slot_density(self, decay, parent, hel):
                return pools[decay[1]][decay[2]]
        return Stub()

    def test_pool_average_is_the_identity(self):
        """The property the method needs from the decay sample."""
        import numpy as np
        for seed in (100, 200):
            pool = self._pool(seed)
            average = sum(np.array([[d.values[0], d.values[1]],
                                    [np.conj(d.values[1]), d.values[2]]])
                          for d in pool) / self.POOL
            self.assertTrue(np.allclose(average, np.eye(2) / 2))

    def test_reproduces_the_joint_distribution(self):
        """The whole claim: p(decays) proportional to N_n, i.e. the same target
        the joint accept/reject samples -- while only ever redrawing one
        particle at a time."""
        import random
        rho = self._production_density()
        pools = {0: self._pool(100), 1: self._pool(200)}
        stub = self._stub(rho, pools)
        production = self._Prod([self._Part(2, -1), self._Part(-2, -1),
                                 self._Part(6), self._Part(-6)])
        evt_decayfile = {6: {0: 'f'}, -6: {0: 'f'}}
        particles, slots = interface_madspin.MadSpinInterface._sequential_slots(
                                                    production, (6, -6))
        stub._slot_of = {index: slot for slot, index in enumerate(slots)}

        exact = {(a, b): stub._partial_density_contraction(
                            rho, self.HELS, {0: pools[0][a], 1: pools[1][b]}).real
                 for a in range(self.POOL) for b in range(self.POOL)}
        total = sum(exact.values())
        exact = {k: v / total for k, v in exact.items()}
        self.assertTrue(all(v > 0 for v in exact.values()))

        random.seed(0)
        counts = collections.Counter()
        nb_run = 20000
        for _ in range(nb_run):
            decays = stub.sequential_accept_reject(production, evt_decayfile,
                                                   [4.0, 4.0], 10)
            counts[(decays[6][0][2], decays[-6][0][2])] += 1

        for combo, want in exact.items():
            got = counts[combo] / float(nb_run)
            self.assertLess(abs(got / want - 1), 0.15,
                            'combo %s: got %.4f, expected %.4f' % (combo, got, want))


class TestSequentialPoolLadder(unittest.TestCase):
    """_sequential_pool_ladder / _sequential_active: how many decay events a
    production event burns per pdg once the accept/reject is per particle, and
    when that regime applies at all.
    """

    class _Part(object):
        def __init__(self, spin):
            self._spin = spin
        def get(self, key):
            return self._spin

    class _Model(object):
        def __init__(self, spins):
            self.spins = spins
        def get_particle(self, pdg):
            return TestSequentialPoolLadder._Part(self.spins[pdg])

    def _stub(self, spins, **options):
        interface = interface_madspin.MadSpinInterface
        class Stub(object):
            _sequential_pool_ladder = interface._sequential_pool_ladder
            _sequential_active = interface._sequential_active
            _sequential_spin_order = interface._sequential_spin_order
            _decay_pool_ladder = staticmethod(interface._decay_pool_ladder)
        stub = Stub()
        stub.model = self._Model(spins)
        stub.options = {'sequential_decay': True, 'fixed_order': False,
                        'spinmode': 'PA', 'sequential_spin_order': '2 3 1'}
        stub.options.update(options)
        return stub

    NB = 1000

    def test_ladder_follows_the_ordering_and_spares_scalars(self):
        """t, t~ take the first two rungs; the higgs is last and stays at 1.1
        because it can never be rejected."""
        stub = self._stub({6: 2, -6: 2, 25: 1})
        got = stub._sequential_pool_ladder({6: self.NB, -6: self.NB, 25: self.NB},
                                           self.NB, True)
        self.assertEqual(got, {-6: 1.5, 6: 2.0, 25: 1.1})

    def test_identical_parents_share_a_pool_at_their_largest_rung(self):
        """Two tops occupy slots 0 and 1 but read the same pool, so it must be
        sized for the hungrier of the two; the vector follows at slot 2."""
        stub = self._stub({6: 2, 24: 3})
        got = stub._sequential_pool_ladder({6: 2 * self.NB, 24: self.NB},
                                           self.NB, True)
        self.assertEqual(got, {6: 2.0, 24: 2.5})

    def test_no_ladder_when_the_joint_test_is_used(self):
        """Empty dict -> the caller keeps the historical 1.1 / 2.0."""
        interface = interface_madspin.MadSpinInterface
        spins = {6: 2, -6: 2}
        pools = {6: self.NB, -6: self.NB}
        # opted out
        self.assertEqual(self._stub(spins, sequential_decay=False)
                             ._sequential_pool_ladder(pools, self.NB, True), {})
        # not density mode
        self.assertEqual(self._stub(spins)
                             ._sequential_pool_ladder(pools, self.NB, False), {})
        # fixed_order keeps the joint test
        self.assertEqual(self._stub(spins, fixed_order=True)
                             ._sequential_pool_ladder(pools, self.NB, True), {})
        # a spinmode outside the supported set keeps the joint test
        self.assertEqual(self._stub(spins, spinmode='none')
                             ._sequential_pool_ladder(pools, self.NB, True), {})

    def test_sequential_active_gate(self):
        stub = self._stub({6: 2})
        self.assertTrue(stub._sequential_active(True))
        self.assertFalse(stub._sequential_active(False))  # not density mode
        # onshell is supported just like PA
        self.assertTrue(self._stub({6: 2}, spinmode='onshell')._sequential_active(True))

    def test_sequential_active_auto(self):
        """'auto' (the default) resolves per spinmode: sequential for the
        PA/onshell pole approximations, joint for madspin/full."""
        for mode, expected in [('PA', True), ('onshell', True),
                               ('madspin', False), ('full', False),
                               ('none', False)]:
            stub = self._stub({6: 2}, sequential_decay='auto', spinmode=mode)
            self.assertEqual(stub._sequential_active(True), expected,
                             'auto + spinmode=%s' % mode)
        # fixed_order still forces the joint test
        stub = self._stub({6: 2}, sequential_decay='auto', fixed_order=True)
        self.assertFalse(stub._sequential_active(True))

    def test_madspin_option_defaults(self):
        """The shipped defaults: spinmode=madspin, jacobian in the weight,
        sequential_decay on auto (and switchable off/back by the user)."""
        options = interface_madspin.MadSpinOptions()
        self.assertEqual(options['spinmode'], 'madspin')
        self.assertEqual(options['density_keep_jacobian'], True)
        self.assertEqual(options['sequential_decay'], 'auto')
        options['sequential_decay'] = 'False'
        self.assertEqual(options['sequential_decay'], False)
        options['sequential_decay'] = 'auto'
        self.assertEqual(options['sequential_decay'], 'auto')


class TestScanMaxwgtDecomposition(unittest.TestCase):
    """The parallel max-weight scan splits the probe events across workers and
    concatenates their per-event vectors. That is only valid if scanning a range
    of events, then another, gives exactly the per-event vectors of scanning the
    whole -- which is what _scan_maxwgt_parallel relies on. Tested here without
    fork, on the synthetic densities of TestSequentialAcceptReject.
    """

    def _fixture(self):
        base = TestSequentialAcceptReject()
        rho = base._production_density()
        pools = {0: base._pool(100), 1: base._pool(200)}
        stub = base._stub(rho, pools)
        production = base._Prod([base._Part(2, -1), base._Part(-2, -1),
                                 base._Part(6), base._Part(-6)])
        _, slots = interface_madspin.MadSpinInterface._sequential_slots(
                                                    production, (6, -6))
        stub._slot_of = {index: slot for slot, index in enumerate(slots)}
        return stub, [production] * 6, {6: {0: 'f'}, -6: {0: 'f'}}

    def test_range_split_matches_the_whole(self):
        """scan[0:6] == scan[0:2] + scan[2:6], event for event, at fixed seed."""
        import random
        stub, events, evt_decayfile = self._fixture()

        random.seed(5)
        whole, _ = stub._scan_maxwgt_range(events, 0, 6, evt_decayfile, 6, 30)

        random.seed(5)
        first, _ = stub._scan_maxwgt_range(events, 0, 2, evt_decayfile, 6, 30)
        second, _ = stub._scan_maxwgt_range(events, 2, 6, evt_decayfile, 6, 30)

        self.assertEqual(len(whole), 6)
        self.assertEqual(first + second, whole)

    def test_one_vector_per_event_one_entry_per_slot(self):
        stub, events, evt_decayfile = self._fixture()
        import random
        random.seed(1)
        #Ignore the message "Error while creating the f2py modules for the production/decay part"
        per_event, _ = stub._scan_maxwgt_range(events, 0, 6, evt_decayfile, 6, 20)
        self.assertEqual(len(per_event), 6)
        for vec in per_event:
            self.assertEqual(len(vec), 2)          # two decaying particles
            self.assertTrue(all(w >= 0 for w in vec))


class TestOffshellRateFactor(unittest.TestCase):
    """Z_k(m): the normalisation the per-angle stage of the offshell sequential
    accept/reject divides out, and which therefore has to be put back into the
    mass-set weight.

    Z_k is the offshell decay rate at the sampled virtuality over the onshell
    one -- for a two-body decay of a spin-1/2 parent it is (m/M) Gamma(m)/Gamma(M),
    a smooth function of that slot's virtuality alone. These tests check that the
    tabulation recovers a known one from the samples the max-weight probe
    collects, that it interpolates and clips as advertised, and that it lands in
    the two weights the way sequential_exact needs.
    """

    class _Val(object):
        def __init__(self, value):
            self.value = value

    class _Banner(object):
        def get(self, card, kind, pdg):
            return TestOffshellRateFactor._Val(173.0)

    class _Stub(object):
        _build_z_tables = interface_madspin.MadSpinInterface._build_z_tables
        _weighted_polyfit2 = staticmethod(
                        interface_madspin.MadSpinInterface._weighted_polyfit2)
        _z_slot_keys = staticmethod(
                        interface_madspin.MadSpinInterface._z_slot_keys)
        _zhat = interface_madspin.MadSpinInterface._zhat
        _complete_offshell_probe = \
                        interface_madspin.MadSpinInterface._complete_offshell_probe
        def __init__(self, exact=False):
            self.banner = TestOffshellRateFactor._Banner()
            self.options = {'sequential_exact': exact}
            self._z_tables = {}

    @staticmethod
    def _truth(mass):
        """A stand-in running width: (m/M) Gamma(m)/Gamma(M) for t > W b."""
        pole, mw = 173.0, 80.419
        def rate(m):
            x = (mw / m) ** 2
            return m ** 4 * (1 - x) ** 2 * (1 + 2 * x)
        return rate(mass) / rate(pole)

    def _samples(self, nb=20000, seed=3, spread=1.0, threshold=0.0):
        """(virtuality, rate factor) pairs as the probe records them: one draw
        each, the rate factor fluctuating around Z(m) with a large spread -- the
        table is an average, not a fit through clean points."""
        import random
        rng = random.Random(seed)
        out = []
        for _ in range(nb):
            mass = rng.uniform(150.0, 196.0)
            if mass < threshold:
                out.append((mass, 0.0))
                continue
            # lognormal noise, mean one: E[value | m] = Z(m)
            noise = math.exp(rng.gauss(0, spread) - 0.5 * spread ** 2)
            out.append((mass, self._truth(mass) * noise))
        return out

    def test_polyfit_recovers_a_quadratic(self):
        xs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        ys = [0.5 - 2 * x + 3 * x ** 2 for x in xs]
        coeff = self._Stub()._weighted_polyfit2(xs, ys, [1.0] * len(xs))
        for got, want in zip(coeff, [0.5, -2.0, 3.0]):
            self.assertAlmostEqual(got, want, places=6)

    def test_polyfit_is_degenerate_below_three_points(self):
        stub = self._Stub()
        self.assertIsNone(stub._weighted_polyfit2([1.0, 2.0], [1.0, 2.0], [1, 1]))
        self.assertIsNone(stub._weighted_polyfit2([1.0] * 4, [1.0] * 4, [1] * 4))

    def test_tabulation_recovers_the_running_width(self):
        """The whole point: an average over noisy per-draw samples reproduces
        the underlying Z(m) across the Breit-Wigner range.

        The tolerance that matters is on the *slope* of ln Z -- a fractional
        error there survives as the same fractional error on the lineshape shift
        the factor corrects -- so it is checked directly, and much more tightly
        than the 10% the physics needs.
        """
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples()})
        self.assertIn('6_0', stub._z_tables)
        for mass in (152.0, 160.0, 173.0, 185.0, 195.0):
            self.assertLess(abs(stub._zhat('6_0', mass) / self._truth(mass) - 1),
                            0.05, 'Z(%s) off' % mass)
        slope = ((math.log(stub._zhat('6_0', 190.0) / stub._zhat('6_0', 156.0)))
                 / (math.log(self._truth(190.0) / self._truth(156.0))))
        self.assertLess(abs(slope - 1), 0.05)

    def test_normalised_at_the_pole(self):
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples()})
        self.assertAlmostEqual(stub._zhat('6_0', 173.0), 1.0, places=6)

    def test_held_constant_outside_the_probed_range(self):
        """Beyond the samples the fit is unconstrained, so it is frozen at the
        edge rather than extrapolated."""
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples()})
        self.assertEqual(stub._zhat('6_0', 400.0), stub._zhat('6_0', 196.0))
        self.assertEqual(stub._zhat('6_0', 100.0), stub._zhat('6_0', 150.0))

    def test_threshold_is_recorded_as_a_hard_zero(self):
        """Virtualities no decay of the pool can be reshuffled onto give a rate
        factor of exactly zero, and the mass-set stage must reject them: Z there
        is 0, not the low edge of the fit."""
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables(
                            {'6_0': self._samples(threshold=165.0)})
        self.assertEqual(stub._zhat('6_0', 160.0), 0.0)
        self.assertGreater(stub._zhat('6_0', 180.0), 0.0)

    def test_no_table_is_a_factor_one(self):
        """The max-weight probe itself runs before any table exists, and every
        non-offshell mode never builds one."""
        stub = self._Stub()
        self.assertEqual(stub._zhat('6_0', 160.0), 1.0)
        stub._z_tables = stub._build_z_tables({'6_0': self._samples(nb=10)})
        self.assertEqual(stub._z_tables, {})
        self.assertEqual(stub._zhat('6_0', 160.0), 1.0)

    def test_slot_keys_separate_identical_parents(self):
        Part = TestSequentialAcceptReject._Part
        particles = [Part(6), Part(-6), Part(6)]
        keys = self._Stub()._z_slot_keys(particles, [0, 2, 1])
        self.assertEqual(keys, ['6_0', '6_1', '-6_0'])

    def _probe_event(self):
        return {'keys': ['6_0', '-6_0'], 'order': [1, 0],
                'chains': [[[2.0, 3.0, 5.0], [160.0, 180.0]],
                           [[1.0, 7.0, 4.0], [173.0, 173.0]]]}

    def test_probe_completion_puts_z_in_the_mass_weight(self):
        """The probe records the mass-set weight before Z exists; completing it
        multiplies in one Z per slot, and leaves the per-angle weights alone."""
        stub = self._Stub()
        stub._z_tables = stub._build_z_tables({'6_0': self._samples(),
                                               '-6_0': self._samples(seed=9)})
        z0, z1 = stub._zhat('6_0', 160.0), stub._zhat('-6_0', 180.0)
        best = stub._complete_offshell_probe(self._probe_event())
        self.assertAlmostEqual(best[0], max(2.0 * z0 * z1, 1.0), places=10)
        self.assertEqual(best[1], 7.0)      # position 0 = slot 1
        self.assertEqual(best[2], 5.0)

    def test_probe_completion_divides_z_out_per_slot_when_exact(self):
        """Under sequential_exact the mass stage pays Z and the slot takes it
        back, so the bound each slot is tested against is the one of w_k/Z_k --
        mapped through the ordering, not the slot order."""
        stub = self._Stub(exact=True)
        stub._z_tables = stub._build_z_tables({'6_0': self._samples(),
                                               '-6_0': self._samples(seed=9)})
        z_by_slot = [stub._zhat('6_0', 160.0), stub._zhat('-6_0', 180.0)]
        best = stub._complete_offshell_probe(self._probe_event())
        # order [1, 0]: probe position 0 is slot 1, position 1 is slot 0
        self.assertAlmostEqual(best[1], max(3.0 / z_by_slot[1], 7.0), places=10)
        self.assertAlmostEqual(best[2], max(5.0 / z_by_slot[0], 4.0), places=10)


class TestTwoStageMassDistribution(unittest.TestCase):
    """The bias the Z_k factor exists to remove, and the two cures, on a model
    of the offshell chain small enough to solve exactly.

    The chain is: draw a virtuality m from a prior, accept the mass set with
    probability w_mass(m) Z_hat(m) / C, then draw decay angles until one is
    accepted with probability w(m, angle) / C_ang. The target -- what the joint
    accept/reject samples -- is p(m) w_mass(m) E_angle[w(m, .)]. Because the
    per-angle stage redraws until it accepts, it divides its own normalisation
    Z(m) = E_angle[w(m, .)] out again, so with Z_hat = 1 the accepted
    virtualities come out proportional to p(m) w_mass(m) instead: the
    Breit-Wigner shape rather than the offshell one. This is the measured
    ttbar bias in miniature.
    """

    ANGLES = [0.3, 0.8, 1.6, 2.5]      # the decay "pool"
    MASSES = [160.0, 170.0, 180.0, 190.0]

    def _w(self, mass, angle):
        """Per-angle weight; its angle average rises steeply with the mass, as
        the offshell rate factor does."""
        return (mass / 170.0) ** 6 * angle

    def _w_mass(self, mass):
        return 1.0 + (mass - 160.0) / 100.0

    def _z(self, mass):
        return sum(self._w(mass, a) for a in self.ANGLES) / len(self.ANGLES)

    def _target(self):
        raw = {m: self._w_mass(m) * self._z(m) for m in self.MASSES}
        total = sum(raw.values())
        return {m: v / total for m, v in raw.items()}

    def _run(self, zhat, exact, nb=200000, seed=7):
        """The two-stage chain, mirroring sequential_accept_reject's offshell
        branch: mass-set accept/reject, then one slot redrawn to acceptance
        (or, under ``exact``, a rejected angle killing the mass set)."""
        import random
        rng = random.Random(seed)
        c_mass = max(self._w_mass(m) * zhat(m) for m in self.MASSES) * 1.01
        c_ang = max(self._w(m, a) / zhat(m)
                    for m in self.MASSES for a in self.ANGLES) * 1.01
        counts = collections.Counter()
        for _ in range(nb):
            while True:
                mass = rng.choice(self.MASSES)
                if rng.random() * c_mass >= self._w_mass(mass) * zhat(mass):
                    continue
                restart = False
                while True:
                    angle = rng.choice(self.ANGLES)
                    if rng.random() * c_ang < self._w(mass, angle) / zhat(mass):
                        break
                    if exact:
                        restart = True
                        break
                if not restart:
                    break
            counts[mass] += 1
        return {m: counts[m] / float(nb) for m in self.MASSES}

    def _assert_close(self, got, want, tolerance):
        for mass in self.MASSES:
            self.assertLess(abs(got[mass] / want[mass] - 1), tolerance,
                            'mass %s: got %.4f, want %.4f' % (mass, got[mass],
                                                              want[mass]))

    def test_without_the_factor_the_mass_distribution_is_biased(self):
        """The bug: the accepted virtualities follow p(m) w_mass(m), the shape
        the per-angle stage was supposed to reweight."""
        got = self._run(lambda m: 1.0, exact=False)
        prior = {m: self._w_mass(m) for m in self.MASSES}
        total = sum(prior.values())
        self._assert_close(got, {m: v / total for m, v in prior.items()}, 0.03)
        self.assertGreater(abs(got[190.0] / self._target()[190.0] - 1), 0.3)

    def test_the_exact_factor_restores_the_target(self):
        self._assert_close(self._run(self._z, exact=False), self._target(), 0.03)

    def test_a_wrong_factor_biases_by_exactly_its_error(self):
        """Why the tabulation has to be accurate: the per-angle stage divides
        out the true Z whatever weight it is given, so the residual bias is
        Z_hat/Z -- it does not cancel."""
        skew = lambda m: self._z(m) * (m / 170.0) ** 2
        got = self._run(skew, exact=False)
        want = {m: v * (m / 170.0) ** 2 for m, v in self._target().items()}
        total = sum(want.values())
        self._assert_close(got, {m: v / total for m, v in want.items()}, 0.03)

    def test_sequential_exact_is_right_for_any_factor(self):
        """And why the switch exists: rejecting the mass set instead of
        redrawing the angle stops the per-angle stage from normalising, so
        Z_hat cancels from the chain and only sets the efficiency."""
        for zhat in (lambda m: 1.0,
                     lambda m: self._z(m),
                     lambda m: self._z(m) * (m / 170.0) ** 2):
            self._assert_close(self._run(zhat, exact=True), self._target(), 0.03)


class TestOffshellCache(unittest.TestCase):
    """The offshell sequential bounds travel with the Z_k tables that complete
    them, so the cache holds two coupled objects and must not be read back under
    a schema it was not written with.  A mismatch is ignored rather than raised
    on: the scan is reproducible, so re-measuring is always available, whereas a
    table dereferenced under the wrong schema would either crash inside the
    accept/reject or silently weight the virtualities with the wrong fit.
    """

    class _Stub(object):
        _OFFSHELL_CACHE_FORMAT = \
            interface_madspin.MadSpinInterface._OFFSHELL_CACHE_FORMAT
        _read_offshell_cache = \
            interface_madspin.MadSpinInterface._read_offshell_cache

    def _write(self, payload):
        import json, tempfile
        handle, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(handle, 'w') as f:
            json.dump(payload, f)
        self.addCleanup(os.remove, path)
        return path

    def _good(self):
        return {'format': self._Stub._OFFSHELL_CACHE_FORMAT,
                'maxwgts': [17.0, 2.3, 3.9],
                'z_tables': {'6_0': {'pole': 173.0, 'coeff': [0.0, 2.0, -1.0],
                                     'zero_below': 0.0, 'range': [150.0, 195.0]}}}

    def test_round_trip(self):
        got = self._Stub()._read_offshell_cache(self._write(self._good()))
        self.assertEqual(got['maxwgts'], [17.0, 2.3, 3.9])
        self.assertEqual(got['z_tables']['6_0']['pole'], 173.0)

    def test_missing_file_is_not_an_error(self):
        self.assertIsNone(self._Stub()._read_offshell_cache('/no/such/file'))
        self.assertIsNone(self._Stub()._read_offshell_cache(''))

    def test_every_malformed_shape_is_ignored(self):
        """Each of these would otherwise surface as a KeyError, an IndexError or
        a wrong weight somewhere inside the unweighting loop."""
        cases = {}
        cases['no format tag'] = {k: v for k, v in self._good().items()
                                  if k != 'format'}
        cases['older format'] = dict(self._good(), format=0)
        cases['no bounds'] = dict(self._good(), maxwgts=[])
        cases['no tables'] = {k: v for k, v in self._good().items()
                              if k != 'z_tables'}
        short = self._good()
        short['z_tables']['6_0'].pop('zero_below')
        cases['table missing a field'] = short
        degree = self._good()
        degree['z_tables']['6_0']['coeff'] = [0.0, 2.0, -1.0, 0.5]
        cases['a cubic fit'] = degree
        window = self._good()
        window['z_tables']['6_0']['range'] = [150.0]
        cases['a malformed range'] = window
        for name, payload in cases.items():
            self.assertIsNone(
                self._Stub()._read_offshell_cache(self._write(payload)), name)

    def test_garbage_is_ignored(self):
        import tempfile
        handle, path = tempfile.mkstemp(suffix='.json')
        with os.fdopen(handle, 'w') as f:
            f.write('not json at all')
        self.addCleanup(os.remove, path)
        self.assertIsNone(self._Stub()._read_offshell_cache(path))


class TestPolyfitConditioning(unittest.TestCase):
    """_weighted_polyfit2 solves the normal equations of a fit in
    u = ln(m/pole), which spans about +-0.13 over a Breit-Wigner window.  The
    moments therefore range over four orders of magnitude before any data is
    seen, which is why the degeneracy test has to be relative to the size of the
    matrix rather than an absolute floor.
    """

    def _fit(self, xs, ys, ws=None):
        return interface_madspin.MadSpinInterface._weighted_polyfit2(
                        xs, ys, ws or [1.0] * len(xs))

    def _window(self, pole=173.0, lo=150.7, hi=195.4, nb=20):
        return [math.log((lo + i * (hi - lo) / (nb - 1)) / pole)
                for i in range(nb)]

    def test_recovers_a_quadratic_on_the_real_fit_variable(self):
        """Exact data over the actual window, with the actual weights (bin
        counts in the thousands): the conditioning of this fit is ~3e4, so
        double precision must return the coefficients essentially exactly."""
        xs = self._window()
        ys = [0.3 + 2.0 * x - 1.0 * x ** 2 for x in xs]
        got = self._fit(xs, ys, [1000.0] * len(xs))
        for value, want in zip(got, [0.3, 2.0, -1.0]):
            self.assertAlmostEqual(value, want, places=8)

    def test_all_bins_at_one_virtuality_is_degenerate(self):
        """The case the tolerance exists for.  An absolute floor of 1e-30 would
        let this through -- the moments are O(n) -- and return a quadratic
        fitted to nothing."""
        self.assertIsNone(self._fit([0.1] * 6, [1.0, 2.0, 1.5, 1.2, 1.8, 1.4]))

    def test_a_needle_narrow_window_is_degenerate(self):
        """A resonance whose samples all land within rounding of each other:
        the moments are tiny in absolute terms but the matrix is still
        singular relative to itself."""
        xs = [1e-13 * i for i in range(6)]
        self.assertIsNone(self._fit(xs, [1.0, 2.0, 1.5, 1.2, 1.8, 1.4],
                                    [1000.0] * 6))

    def test_two_distinct_points_cannot_fix_a_quadratic(self):
        self.assertIsNone(self._fit([0.0, 0.0, 0.1, 0.1], [1.0, 1.0, 2.0, 2.0]))
