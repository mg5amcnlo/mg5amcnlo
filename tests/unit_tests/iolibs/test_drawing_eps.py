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
"""Unit test library for the routine creating the points position for the 
    diagram drawing and for the creation of the EPS file."""

from __future__ import division

# The following lines are needed to run the
# diagram generation using __main__
import os, sys
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
root_path = os.path.realpath(os.path.join(root_path,'..','..'))
sys.path.insert(0, root_path)


import os
import pickle

import madgraph.core.base_objects as base_objects
import madgraph.core.drawing as drawing
import tests.unit_tests.core.test_drawing as test_drawing 
import madgraph.iolibs.drawing_eps as draw_eps
import madgraph.iolibs.import_v4 as import_v4
import madgraph.iolibs.files as files
import madgraph.various.misc as misc

import tests.unit_tests as unittest
_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

def define_model():
    global _model
    
    if test_drawing._model:
        _model = test_drawing._model
    else:
        test_drawing.define_model()
        _model = test_drawing._model

#===============================================================================
# TestDrawingEPS
#===============================================================================
class TestDrawingOption(unittest.TestCase):
    """Sanity check for all combination of option. This check on a small sample
    of diagram that no line have zero lenght and that we don't have any line 
    crossing for any combination of option."""

    def setUp(self):
        """Basic building of the object needed to build the test"""
        
        if not hasattr(self, 'store_diagram'):
            filehandler = open(os.path.join(_file_path, \
                                '../input_files/test_draw.obj'), 'r')
            TestDrawingOption.store_diagram = pickle.load(filehandler)
        try:
            _model
        except:
            define_model()
            
    
    def schedular(self, diagram):
        """Test that the DrawingEPS returns valid result"""
        
        horizontal_list = [True, False]
        external_list = [0, 1, 1.5]
        contract_unpropa_list = [True, False]
        max_size_list = [0, 1.8]
        
        opt = drawing.DrawOption()
        for horizontal in horizontal_list:
            opt.set('horizontal', horizontal)
            for external in external_list:
                opt.set('external', external)
                for contract_unpropa in contract_unpropa_list:
                    opt.set('contract_non_propagating', contract_unpropa)
                    for max_size in max_size_list:
                        opt.set('max_size', max_size)
                        
                        plot = draw_eps.EpsDiagramDrawer(diagram, \
                                        '__testdiag__.eps', model=_model, \
                                         amplitude='')
                        plot.draw(opt)
                        self.assertFalse(\
                                    plot.diagram._debug_has_intersection())
                        for line in plot.diagram.lineList:
                            self.assertNotAlmostEquals(line.get_length(), 0)
                            
    def test_option_6g(self):
        """Test that gg>6g is fine with all options"""
        diagram = self.store_diagram['g g > g g g g g g'][73]
        self.schedular(diagram)
        
    def test_option_6g_second(self):
        """Test that gg>6g is fine with all options"""
        diagram = self.store_diagram['g g > g g g g g g'][2556]
        self.schedular(diagram)   
        
    def test_option_multi_type(self):
        """Test that t h > t g W+ W-  is fine with all options"""
        diagram = self.store_diagram['t h > t g w+ w-'][0] 
        self.schedular(diagram)        
          
#===============================================================================
# TestDrawingS_EPS
#===============================================================================
class TestDrawingS_EPS(unittest.TestCase):
    """ Class testing if we can create the files in the EPS mode for a set
        of diagrams.
    
    This test the following two points:
    1) can we create the output file?
    2) can we convert him in pdf? (Imagemagick is needed for this)
        checking that the file is valid."""

    
    def setUp(self):
        """Charge a diagram to draw"""
        
        if not hasattr(self, 'store_diagram'):
            filehandler = open(os.path.join(_file_path, \
                                '../input_files/test_draw.obj'), 'r')
            TestDrawingS_EPS.store_diagram = pickle.load(filehandler)
        try:
            _model
        except:
            define_model()

        self.diagram = base_objects.DiagramList()
        for i in range(7):
            self.diagram.append(self.store_diagram['t h > t g w+ w-'][i])

        self.plot = draw_eps.MultiEpsDiagramDrawer(self.diagram, '__testdiag__.eps', \
                                          model=_model, amplitude='')
        
    def test_blob(self):
        """ test if the blob are written correctly """
        #prepare everything
        diagram = self.store_diagram['u~ u~ > e+ e- u~ u~ g'][1]
        plot = draw_eps.EpsDiagramDrawer(diagram, \
                                        '__testdiag__.eps', model=_model, \
                                         amplitude='')

        plot.convert_diagram()
        plot.initialize()
        
        nb_blob =0
        for i, vertex in enumerate(plot.diagram.vertexList): 
            plot.text = ''
            plot.draw_vertex(vertex, bypass= ['QCD'])
            if '1.0 Fblob' in plot.text:
                nb_blob += 1
        self.assertEqual(nb_blob, 4)
            
    
    def output_is_valid(self, position, pdf_check=True):
        """Test if the output files exist. 
        Additionally if pdf_check is on True
        check if we can convert the output file in pdf. Finally delete files."""

        # Check if exist
        self.assertTrue(os.path.isfile(position))

        # Check if the file is valid
        if pdf_check and misc.which('convert'):
            filename, ext = os.path.splitext('position')
            os.system('convert ' + position + ' ' + filename + '.pdf')

            # Try is use to ensure that no file are left on disk
            try:
                self.assertTrue(os.path.isfile(filename + '.pdf'))
            except:
                os.remove(position)
                raise
            os.remove(filename + '.pdf')
        os.remove(position)
        return
    
    def test_schedular(self):
        """Test the multidiagram drawing"""
        
        opt = drawing.DrawOption()
        self.setUp()
        self.plot.draw(opt=opt)
        self.output_is_valid('__testdiag__.eps')


if __name__ == '__main__':

    # For debugging it's interesting to store problematic diagram in one file.
    #Those one are generated with cmd and store in files with pickle module.

    process_diag = {}
    process_diag['mu- > vm e- ve~'] = [0]
    process_diag['d > d d g d~ QED=0'] = [0]
    process_diag['u d~ > c s~'] = [0]
    process_diag['g g > g g'] = [0, 1, 2, 3]
    process_diag['g g > g g g'] = [0, 1]
    process_diag['g g > g g u u~'] = [18, 100]
    process_diag['g g > g g g g'] = [0, 26, 92, 93, 192]
    process_diag['g g > g g g g g g'] = [73, 2556]
    process_diag['mu+ mu- > w+ w- a'] = [6, 7]
    process_diag['t h > t g w+ w-'] = [0, 1, 2, 3, 4, 5, 6, 7]
    process_diag['u u > z u u g'] = [26]
    process_diag['u~ u~ > z u~ u~ g'] = [26]
    process_diag['u~ u~ > e+ e- u~ u~ g'] = [1, 8]
    process_diag['e- e+ > t t~, t > w+ b'] = [0]
    process_diag['u u~ > w+ w- e+ e-'] = [0]
    process_diag['OUTPUT: u u~ > w+ w- e+ e-'] = [0]
    process_diag['w+ w- > w+ w- z z'] = [7,9,10]
    process_diag['OUTPUT: w+ w- > w+ w- z z'] = [7,9,10]
    
    from madgraph.interface.master_interface import MasterCmdS
    cmd = MasterCmd()
    cmd.exec_cmd('import model %s/models/sm' % root_path)
    #cmd.do_import('model_v4 ' + os.path.join(_file_path, \
    #                                    '../input_files/v4_sm_particles.dat'))
    #cmd.do_import('model_v4 ' + os.path.join(_file_path, \
    #                                '../input_files/v4_sm_interactions.dat'))
    cmd._curr_model = _model 
    
    # Create the diagrams
    diag_content = {}
    for gen_line, pos_list in process_diag.items():
        output=False
        if gen_line.startswith('OUTPUT:'):
            output = True
            gen_line = gen_line[7:]
            print 'WITH OUTPUT :',
        print gen_line, ':',
        gen_line_with_order = gen_line + ' QCD=99'
        cmd.exec_cmd('generate ' + gen_line_with_order)
        if output:
            gen_line = 'OUTPUT:'+gen_line
            #cmd.export.generate_matrix_elements()
            try:
                cmd.do_output('/tmp/MGPROC -f')
            except: 
                pass
        #Look for decay chains
        if ',' in gen_line:
            amp = cmd._curr_amps[0]
            import madgraph.core.helas_objects as helas_objects
            matrix_elements = \
            helas_objects.HelasDecayChainProcess(amp).combine_decay_chain_processes()
            if matrix_elements:
                amplitude = matrix_elements[0].get('base_amplitude')     
        else:
            amplitude = cmd._curr_amps[0]
        print len(amplitude['diagrams'])
            
                                                    
        diag_content[gen_line] = {}
        for pos in pos_list:
            diag_content[gen_line][pos] = amplitude['diagrams'][pos]


    # Store the diagrams  
    file_test_diagram = open(os.path.join(_file_path , \
                                    '../input_files/test_draw.obj'), 'w')
    pickle.dump(diag_content, file_test_diagram)
    print 'done'





