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

import copy
import array

import madgraph.core.base_objects as MG
import madgraph.various.misc as misc
import MadSpin.decay as madspin 

from madgraph import MG5DIR
#
class TestBanner(unittest.TestCase):
    """Test class for the reading of the banner"""

    def test_extract_info(self):
        """Test that the banner is read properly"""

        banner=pjoin(MG5DIR, 'tests', 'input_files', 'tt_banner.txt')
        inputfile = open(banner, 'r')
        mybanner=madspin.Banner(inputfile)
        mybanner.ReadBannerFromFile()
        process=mybanner.proc["generate"]
        model=mybanner.proc["model"]
        self.assertEqual(process,"p p > t t~ @1")
        self.assertEqual(model,"sm")

class Testtopo(unittest.TestCase):
    """Test the extraction of the topologies for the undecayed process"""

    def test_topottx(self):

        curr_dir=os.getcwd()
        os.chdir('/tmp')
        temp_dir=os.getcwd()

        mgcmd=Cmd.MasterCmd()
        process_prod=" g g > t t~ "
        process_full=process_prod+", ( t > b w+ , w+ > mu+ vm ), "
        process_full+="( t~ > b~ w- , w- > mu- vm~ ) "
        decay_tools=madspin.decay_misc()
        topo=decay_tools.generate_fortran_me([process_prod],"sm",0, mgcmd)
        decay_tools.generate_fortran_me([process_full],"sm", 1,mgcmd)

        topo_test={1: {'branchings': [{'index_propa': -1, 'type': 's',\
                'index_d2': 3, 'index_d1': 4}], 'get_id': {}, 'get_momentum': {}, \
                'get_mass2': {}}, 2: {'branchings': [{'index_propa': -1, 'type': 't', \
                'index_d2': 3, 'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 4,\
                 'index_d1': -1}], 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}, \
                   3: {'branchings': [{'index_propa': -1, 'type': 't', 'index_d2': 4, \
                'index_d1': 1}, {'index_propa': -2, 'type': 't', 'index_d2': 3, 'index_d1': -1}],\
                 'get_id': {}, 'get_momentum': {}, 'get_mass2': {}}}
        
        self.assertEqual(topo,topo_test)
               
        list_prod=os.listdir("production_me/SubProcesses")
        counter=0
        for direc in list_prod:
            if direc[0]=="P":
                counter+=1
                prod_name=direc[string.find(direc,"_")+1:]
                old_path=pjoin(temp_dir,'production_me','SubProcesses',direc)
                new_path=pjoin(temp_dir,'production_me','SubProcesses',prod_name)
                if os.path.isdir(new_path): shutil.rmtree(new_path)
                os.rename(old_path, new_path)
                #shutil.rmtree('production_me/SubProcesses/'+direc)
                
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_prod.f')
                shutil.copyfile(file_madspin, new_path+"/check_sa.f")  
                
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_ms')
                shutil.copyfile(file_madspin, new_path+"/makefile") 
                
                file=pjoin(MG5DIR, 'tests', 'input_files', 'param_card_sm.dat')
                shutil.copyfile(file,"production_me/Cards/param_card.dat",) 

                if not os.path.isfile("parameters.inc"):
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'initialize.f')
                    shutil.copyfile(file_madspin,new_path+"/initialize.f")
                    
                    file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'lha_read_ms.f')
                    shutil.copyfile(file_madspin, "production_me/Source/MODEL/lha_read.f" )

                    os.chdir(pjoin(temp_dir,'production_me','Source','MODEL'))
                    try:
                       os.remove('*.o')
                    except:
                       pass
                    misc.compile(cwd=pjoin(temp_dir,'production_me','Source','MODEL'), mode='fortran')
#                    misc.call(' make > /dev/null ')
                    os.chdir(new_path)
                    misc.compile(arg=['init'],cwd=new_path,mode='fortran')
                    misc.call('./init')
                    shutil.copyfile('parameters.inc', '../../../parameters.inc')
                    os.chdir(temp_dir)
                    

                shutil.copyfile('production_me/Source/MODEL/input.inc',new_path+'/input.inc') 
                misc.compile(cwd=new_path, mode='fortran')
                
                
        list_full=os.listdir("full_me/SubProcesses")

        first=1
        for direc in list_full:
            if direc[0]=="P":
                
                counter+=1
                decay_name=direc[string.find(direc,"_")+1:]
                
                old_path=pjoin(temp_dir,'full_me','SubProcesses',direc)
                new_path=pjoin(temp_dir,'full_me','SubProcesses',decay_name)
                if os.path.isdir(new_path): shutil.rmtree(new_path)
                os.rename(old_path, new_path)               
                
                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'driver_full.f')
                shutil.copyfile(file_madspin, new_path+"/check_sa.f")  

                file_madspin=pjoin(MG5DIR, 'MadSpin', 'src', 'makefile_ms')
                shutil.copyfile(file_madspin, new_path+"/makefile") 
                
                shutil.copyfile('full_me/Source/MODEL/input.inc',new_path+'/input.inc') 
                misc.compile(arg=['check'], cwd=new_path, mode='fortran')

                file=pjoin(MG5DIR, 'tests', 'input_files', 'param_card_sm.dat')
                shutil.copyfile(file,"full_me/Cards/param_card.dat")                 
                
                if(os.path.getsize("parameters.inc")<10): 
                    print "Parameters of the model were not written correctly !"
                    os.system("rm parameters.inc")
                if first:
                    first=0
                    decay_pattern=direc[string.find(direc,"_")+1:]
                    decay_pattern=decay_pattern[string.find(decay_pattern,"_")+1:]
                    decay_pattern=decay_pattern[string.find(decay_pattern,"_")+1:]


        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03  \n'
        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
        p_string+='0.5000000E+03  0.1040730E+03  0.4173556E+03 -0.1872274E+03 \n'
        p_string+='0.5000000E+03 -0.1040730E+03 -0.4173556E+03  0.1872274E+03 \n'        

        os.chdir("production_me/SubProcesses/"+prod_name)
        executable_prod="./check"
        external = Popen(executable_prod, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        prod_values=external.communicate(input=p_string)[0] 
        prod_values=prod_values.split()
        prod_values_test=['0.59366146660637686', '7.5713552297679376', '12.386583104018380', '34.882849897228873']
        self.assertEqual(prod_values,prod_values_test)               
        os.chdir(temp_dir)
        
        p_string='0.5000000E+03  0.0000000E+00  0.0000000E+00  0.5000000E+03 \n'
        p_string+='0.5000000E+03  0.0000000E+00  0.0000000E+00 -0.5000000E+03 \n'
        p_string+='0.8564677E+02 -0.8220633E+01  0.3615807E+02 -0.7706033E+02 \n'
        p_string+='0.1814001E+03 -0.5785084E+02 -0.1718366E+03 -0.5610972E+01 \n'
        p_string+='0.8283621E+02 -0.6589913E+02 -0.4988733E+02  0.5513262E+01 \n'
        p_string+='0.3814391E+03  0.1901552E+03  0.2919968E+03 -0.1550888E+03 \n'
        p_string+='0.5422284E+02 -0.3112810E+02 -0.7926714E+01  0.4368438E+02\n'
        p_string+='0.2144550E+03 -0.2705652E+02 -0.9850424E+02  0.1885624E+03\n'

        os.chdir("full_me/SubProcesses/"+decay_name)
        executable_decay="./check"
        external = Popen(executable_decay, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        decay_value=external.communicate(input=p_string)[0] 
        decay_value=decay_value.split()
        decay_value_test=['3.8420345719455465E-017']
        self.assertEqual(decay_value,decay_value_test)
        os.chdir(temp_dir)
#        shutil.rmtree('production_me')
#        shutil.rmtree('full_me')

        