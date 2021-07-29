##############################################################################
#
# Copyright (c) 2010 The MadGraph5_aMC@NLO Development team and Contributors
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
from __future__ import absolute_import
from cmd import Cmd
""" Basic test of the command interface """

import unittest
import madgraph

import madgraph.interface.master_interface as master_cmd
import madgraph.interface.madgraph_interface  as mg_cmd
import madgraph.interface.extended_cmd as ext_cmd
import madgraph.interface.amcatnlo_interface as mecmd
import madgraph.various.misc as misc
import os
from six import StringIO
import logging

root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.dirname(root_path)
# root_path is ./tests
pjoin = os.path.join

class MGerror(Exception): pass

class TestMadEventCmd(unittest.TestCase):
    """ check if the ValidCmd works correctly """
    
        
    def test_v31_syntax_crash(self):
        """Check that process with ambiguous syntax correctly crashes if the flag is not set correctly
        """
        #cmd = mecmd.aMCatNLOInterface()
        #category = set()
        #valid_command = [c for c in dir(cmd) if c.startswith('do_')]

        interface = master_cmd.MasterCmd()
        interface.no_notification()
        interface.do_import('model loop_qcd_qed_sm')
        
        #run_cmd('import model %s' % model)

        stream = StringIO()
        handler_stream = logging.StreamHandler(stream)
        log = logging.getLogger('cmdprint')
        log.setLevel(logging.CRITICAL)

        for handler in log.handlers: 
            log.removeHandler(handler)
        log.addHandler(handler_stream)


        def check_message(line):
            """return False if not warning is raised, return True is a warning is raised"""
            
            stream.seek(0)
            stream.truncate(0)
            myprocdef = interface.extract_process(line)
            try: 
                interface.proc_validity(myprocdef,'aMCatNLO_all')
            except Exception as error:
                if '1804.10017' in str(error):
                    raise MGerror

            text = stream.getvalue()
            if '1804.10017' in text:
                return True
            else:
                return False

        # force the option to not by bypassed
        interface.options['acknowledged_v3.1_syntax'] = False

        # check case where the code crash
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=1 [QED]")
        self.assertRaises(MGerror,check_message, "p p > t t~ QCD=1 [QED QCD]")

        # check case where the code write a warning (critical level)
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=1 [QCD]")
        self.assertRaises(MGerror, check_message, "p p > t t~ QCD=1 QED=0 [QCD]")
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=98 [QCD]")
        self.assertRaises(MGerror, check_message, "p p > t t~ QED=99 QCD=1 [QCD]")
        # check case where the code does not complain
        self.assertFalse(check_message("p p > t t~ QED=0 [QCD]"))
        self.assertFalse(check_message("p p > t t~ / z QCD=0 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QCD=1 QED^2==2 [QCD]"))
        self.assertFalse(check_message("p p > w+ w- QED=99 [QCD]"))
        self.assertFalse(check_message("p p > w+ w- j j $ h QED<=99 [QCD]"))

        # force the option to not  be by bypassed
        interface.options['acknowledged_v3.1_syntax'] = True 
        # and check that no crash/warning is raised anymore
        self.assertFalse(check_message( "p p > t t~ QED=1 [QED]"))
        self.assertFalse(check_message( "p p > t t~ QCD=1 [QED QCD]"))
        self.assertFalse(check_message("p p > t t~ QED=1 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QCD=1 QED=0 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QED=98 [QCD]"))
        self.assertFalse(check_message("p p > t t~ QED=99 QCD=1 [QCD]"))
