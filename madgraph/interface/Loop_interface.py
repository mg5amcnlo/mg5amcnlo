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
"""A user friendly command line interface to access all MadGraph features.
   Uses the cmd package for command interpretation and tab completion.
"""

import os

import madgraph
from madgraph import MG4DIR, MG5DIR, MadGraph5Error
import madgraph.interface.madgraph_interface as mg_interface

#useful shortcut
pjoin = os.path.join

class CheckLoop(mg_interface.CheckValidForCmd):
    pass

class CheckLoopWeb(mg_interface.CheckValidForCmdWeb, CheckLoop):
    pass

class CompleteLoop(mg_interface.CompleteForCmd):
    pass

class HelpLoop(mg_interface.HelpToCmd):
    pass

class LoopInterface(CheckLoop, CompleteLoop, HelpLoop, mg_interface.MadGraphCmd):
    
    def do_generate(self, *args,**opt):
        mg_interface.MadGraphCmd.do_generate(self, *args,**opt)
    
    def do_add(self, *args,**opt):
        if self._curr_model['perturbation_couplings']=={}:
            if self._curr_model['name']=='sm':
                mg_interface.logger.warning(\
                  "The default sm model does not allow to generate"+
                  " loop processes. MG5 now loads 'loop_sm' instead.")
                mg_interface.MadGraphCmd.do_import(self,"model loop_sm")
            else:
                raise MadGraph5Error(
                  "The model %s cannot generate loop processes"\
                  %self._curr_model['name'])
                
        mg_interface.MadGraphCmd.do_add(self, *args,**opt)
   
class LoopInterfaceWeb(mg_interface.CheckValidForCmdWeb, LoopInterface):
    pass

