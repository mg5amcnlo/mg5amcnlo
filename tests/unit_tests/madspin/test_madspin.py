################################################################################
#
# Copyright (c) 2009 The MadGraph Development team and Contributors
#
# This file is a part of the MadGraph 5 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph license which should accompany this
# distribution.
#
# For more information, please visit: http://madgraph.phys.ucl.ac.be
#
################################################################################

"""Unit test library for the spin correlated decay routines
in the madspin directory"""

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

import madgraph.core.base_objects as MG
import madgraph.various.misc as misc
import MadSpin.decay as madspin 
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

class TestDecayChannels(unittest.TestCase):
    """ Test that the different decay channels are correclty generated"""

    def test_symmetrize_decay(self):

        decay_tools=madspin.decay_misc()


        multi_channel_check=[('|z>bb~', '|z>e+e-', '|z>e+e-'), \
            ('|z>cc~', '|z>e+e-', '|z>e+e-'), ('|z>dd~', '|z>e+e-', '|z>e+e-'),\
            ('|z>e+e-', '|z>bb~', '|z>e+e-'), ('|z>e+e-', '|z>cc~', '|z>e+e-'),\
            ('|z>e+e-', '|z>dd~', '|z>e+e-'), ('|z>e+e-', '|z>e+e-', '|z>bb~'),\
            ('|z>e+e-', '|z>e+e-', '|z>cc~'), ('|z>e+e-', '|z>e+e-', '|z>dd~'),\
            ('|z>e+e-', '|z>e+e-', '|z>e+e-'), ('|z>e+e-', '|z>e+e-', '|z>mu+mu-'),\
            ('|z>e+e-', '|z>e+e-', '|z>ss~'), ('|z>e+e-', '|z>e+e-', '|z>ta+ta-'),\
            ('|z>e+e-', '|z>e+e-', '|z>uu~'), ('|z>e+e-', '|z>e+e-', '|z>veve~'), \
            ('|z>e+e-', '|z>e+e-', '|z>vmvm~'), ('|z>e+e-', '|z>e+e-', '|z>vtvt~'), \
            ('|z>e+e-', '|z>mu+mu-', '|z>e+e-'), ('|z>e+e-', '|z>ss~', '|z>e+e-'), \
            ('|z>e+e-', '|z>ta+ta-', '|z>e+e-'), ('|z>e+e-', '|z>uu~', '|z>e+e-'), \
            ('|z>e+e-', '|z>veve~', '|z>e+e-'), ('|z>e+e-', '|z>vmvm~', '|z>e+e-'), \
            ('|z>e+e-', '|z>vtvt~', '|z>e+e-'), ('|z>mu+mu-', '|z>e+e-', '|z>e+e-'), \
            ('|z>ss~', '|z>e+e-', '|z>e+e-'), ('|z>ta+ta-', '|z>e+e-', '|z>e+e-'), \
            ('|z>uu~', '|z>e+e-', '|z>e+e-'), ('|z>veve~', '|z>e+e-', '|z>e+e-'), \
            ('|z>vmvm~', '|z>e+e-', '|z>e+e-'), ('|z>vtvt~', '|z>e+e-', '|z>e+e-')] 

        decay_processes={0: 'z  >  e+ e- ', 1: 'z  >  e+ e- ', 2: 'z  >  all all '}

        branching_per_channel={0: {'|z>e+e-': {'config': ' z > e+ e- ', 'br': 0.03438731}},\
             1: {'|z>e+e-': {'config': ' z > e+ e- ', 'br': 0.03438731}}, \
             2: {'|z>ss~': {'config': ' z > s s~ ', 'br': 0.1523651}, \
                 '|z>e+e-': {'config': ' z > e- e+ ', 'br': 0.03438731}, \
                 '|z>veve~': {'config': ' z > ve ve~ ', 'br': 0.06793735}, \
                 '|z>cc~': {'config': ' z > c c~ ', 'br': 0.1188151}, \
                 '|z>bb~': {'config': ' z > b b~ ', 'br': 0.150743}, \
                 '|z>mu+mu-': {'config': ' z > mu- mu+ ', 'br': 0.03438731}, \
                 '|z>uu~': {'config': ' z > u u~ ', 'br': 0.1188151}, \
                 '|z>ta+ta-': {'config': ' z > ta- ta+ ', 'br': 0.03430994}, \
                 '|z>vmvm~': {'config': ' z > vm vm~ ', 'br': 0.06793735}, \
                 '|z>vtvt~': {'config': ' z > vt vt~ ', 'br': 0.06793735}, \
                 '|z>dd~': {'config': ' z > d d~ ', 'br': 0.1523651}}}

        list_particle_to_decay=decay_processes.keys()
        list_particle_to_decay.sort()

        # also determine which particles are identical
        identical_part=decay_tools.get_identical(decay_processes)

#       2. then use a tuple to identify a decay channel
#       (tag1 , tag2 , tag3, ...)  
#       where the number of entries is the number of branches, 
#       tag1 is the tag that identifies the final state of branch 1, 
#       tag2 is the tag that identifies the final state of branch 2, 
#       etc ...

#       2.a. first get decay_tags = the list of all the tuples

        decay_tags=[]
        for part in list_particle_to_decay:  # loop over particle to decay in the production process
            branch_tags=[ fs for fs in branching_per_channel[part]] # list of tags in a given branch
            decay_tags=decay_tools.update_tag_decays(decay_tags, branch_tags)
#      Now here is how we account for symmetry if identical particles:
#      EXAMPLE:  (Z > mu+ mu- ) (Z> e+ e-)
#      currently in decay_tags, the first Z is mapped onto the channel mu+ mu- 
#                               the second Z is mapped onto the channel e+ e-
#                              => we need to add the symmetric channel obtained by 
#                                 swapping the role of the first Z and the role of the snd Z 
        decay_tags, map_tag2branch =\
            decay_tools.symmetrize_tags(decay_tags, identical_part)

#       2.b. then build the dictionary multi_decay_processes = multi dico
#       first key = a tag in decay_tags
#       second key :  ['br'] = float with the branching fraction for the FS associated with 'tag'   
#                     ['config'] = a list of strings, each of them giving the definition of a branch    

        multi_decay_processes={}
        for tag in decay_tags:
#           compute br + get the config
            br=1.0
            list_branches={}
            for index, part in enumerate(list_particle_to_decay):
                br=br*branching_per_channel[map_tag2branch[tag][part]][tag[index]]['br']
                list_branches[part]=branching_per_channel[map_tag2branch[tag][part]][tag[index]]['config']
            multi_decay_processes[tag]={}
            multi_decay_processes[tag]['br']=br
            multi_decay_processes[tag]['config']=list_branches

#       compute the cumulative probabilities associated with the branching fractions
#       keep track of sum of br over the decay channels at work, since we need to rescale
#       the event weight by this number
        sum_br=decay_tools.set_cumul_proba_for_tag_decay(multi_decay_processes,decay_tags)

        multi_decay_list=sorted(multi_decay_processes.keys())

        self.assertEqual(multi_decay_list,multi_channel_check)


class Testtopo(unittest.TestCase):
    """Test the extraction of the topologies for the undecayed process"""

    def test_topottx(self):

        os.environ['GFORTRAN_UNBUFFERED_ALL']='y'
        path_for_me=pjoin(MG5DIR, 'tests','unit_tests','madspin')
        shutil.copyfile(pjoin(MG5DIR, 'tests','input_files','param_card_sm.dat'),\
		pjoin(path_for_me,'param_card.dat'))
        curr_dir=os.getcwd()
        os.chdir('/tmp')
        temp_dir=os.getcwd()
        mgcmd=Cmd.MasterCmd()
        process_prod=" g g > t t~ "
        process_full=process_prod+", ( t > b w+ , w+ > mu+ vm ), "
        process_full+="( t~ > b~ w- , w- > mu- vm~ ) "
        decay_tools=madspin.decay_misc()
        topo=decay_tools.generate_fortran_me([process_prod],"sm",0, mgcmd, path_for_me)
        decay_tools.generate_fortran_me([process_full],"sm", 1,mgcmd, path_for_me)

        prod_name=decay_tools.compile_fortran_me_production(path_for_me)
	decay_name = decay_tools.compile_fortran_me_full(path_for_me)


        topo_test={1: {'branchings': [{'index_propa': -1, 'type': 's',\
                'index_d2': 3, 'index_d1': 4}], 'get_id': {}, 'get_momentum': {}, \
                'get_mass2': {}}, 2: {'branchings': [{'index_propa': -1, 'type': 't', \
                'index_d2': 3, 'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 4,\
                 'index_d1': -1}], 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}, \
                   3: {'branchings': [{'index_propa': -1, 'type': 't', 'index_d2': 4, \
                'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 3, 'index_d1': -1}],\
                 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}}
        
        self.assertEqual(topo,topo_test)
  

        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03  \n'
        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
        p_string+='0.5000000E+03  0.1040730E+03  0.4173556E+03 -0.1872274E+03 \n'
        p_string+='0.5000000E+03 -0.1040730E+03 -0.4173556E+03  0.1872274E+03 \n'        

       
        os.chdir(pjoin(path_for_me,'production_me','SubProcesses',prod_name))
        executable_prod="./check"
        external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
 
        external.stdin.write(p_string)

        info = int(external.stdout.readline())
        nb_output = abs(info)+1


        prod_values = ' '.join([external.stdout.readline() for i in range(nb_output)])

        prod_values=prod_values.split()
        prod_values_test=['0.59366146660637686', '7.5713552297679376', '12.386583104018380', '34.882849897228873']
        self.assertEqual(prod_values,prod_values_test)               
        external.terminate()


        os.chdir(temp_dir)
        
        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03 \n'
        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
        p_string+='0.8564677E+02 -0.8220633E+01  0.3615807E+02 -0.7706033E+02 \n'
        p_string+='0.1814001E+03 -0.5785084E+02 -0.1718366E+03 -0.5610972E+01 \n'
        p_string+='0.8283621E+02 -0.6589913E+02 -0.4988733E+02  0.5513262E+01 \n'
        p_string+='0.3814391E+03  0.1901552E+03  0.2919968E+03 -0.1550888E+03 \n'
        p_string+='0.5422284E+02 -0.3112810E+02 -0.7926714E+01  0.4368438E+02\n'
        p_string+='0.2144550E+03 -0.2705652E+02 -0.9850424E+02  0.1885624E+03\n'

        os.chdir(pjoin(path_for_me,'full_me','SubProcesses',decay_name))
        executable_decay="./check"
        external = Popen(executable_decay, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        external.stdin.write(p_string)

        nb_output =1 
        decay_value = ' '.join([external.stdout.readline() for i in range(nb_output)])

        decay_value=decay_value.split()
        decay_value_test=['3.8420345719455465E-017']
        for i in range(len(decay_value)): 
            self.assertAlmostEqual(eval(decay_value[i]),eval(decay_value_test[i]))
        os.chdir(curr_dir)
        external.terminate()
        shutil.rmtree(pjoin(path_for_me,'production_me'))
        shutil.rmtree(pjoin(path_for_me,'full_me'))
        os.remove(pjoin(path_for_me,'param_card.dat'))
        os.environ['GFORTRAN_UNBUFFERED_ALL']='n'

        
