#! /usr/bin/env python
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
import os
import sys
import logging
import time
# Get the parent directory (mg root) of the script real path (bin)
# and add it to the current PYTHONPATH
root_path = os.path.split(os.path.dirname(os.path.realpath( __file__ )))[0]
sys.path.append(root_path)

from madgraph import MG5DIR
import madgraph.iolibs.import_v4 as import_v4
import models.import_ufo as import_ufo
import aloha.create_aloha as create_aloha


# Set logging level to error
logging.basicConfig(level=vars(logging)['INFO'],
                    format="%(message)s")

class Compile_MG5:
    
    def __init__(self):
        """ launch all the compilation """
        self.make_UFO_pkl()
        self.make_v4_pkl()
        
    @staticmethod
    def make_v4_pkl():
        """create the model.pkl for each directory"""
        file_cond = lambda p :  (os.path.exists(os.path.join(MG5DIR,'models',p,'particles.dat'))) 
        #1. find v4 model:
        v4_model = [os.path.join(MG5DIR,'models',p) 
                        for p in os.listdir(os.path.join(MG5DIR,'models')) 
                            if file_cond(p)]
            
        for model_path in v4_model:
            #remove old pkl
            start = time.time()
            print 'make pkl for %s :' % os.path.basename(model_path),
            try:
                os.remove(os.path.join(model_path,'model.pkl'))
            except:
                pass
            import_v4.import_model(model_path)
            print '%2fs' % (time.time() - start)
    
    @staticmethod
    def make_UFO_pkl():
        """ """
        file_cond = lambda p : os.path.exists(os.path.join(MG5DIR,'models',p,'particles.py'))
        #1. find UFO model:
        ufo_model = [os.path.join(MG5DIR,'models',p) 
                        for p in os.listdir(os.path.join(MG5DIR,'models')) 
                            if file_cond(p)]
        # model.pkl
        for model_path in ufo_model:
            start = time.time()
            print 'make model.pkl for %s :' % os.path.basename(model_path),
            #remove old pkl
            try:
                os.remove(os.path.join(model_path,'model.pkl'))
            except:
                pass
            import_ufo.import_full_model(model_path)
            print '%2fs' % (time.time() - start)
        
        return
        # aloha routine 
        for model_path in ufo_model:
            start = time.time()
            print 'make ALOHA for %s' % os.path.basename(model_path)
            #remove old pkl
            try:
                os.remove(os.path.join(model_path,'aloha.pkl'))
            except:
                pass    
            try:
                os.system('rm -rf %s &> /dev/null' % os.path.join(model_path,'Fortran'))
            except:
                pass            
            
            ufo_path, ufo_name =os.path.split(model_path)
            sys.path.insert(0, ufo_path)
            output_dir = os.path.join(model_path, 'Fortran')
            create_aloha.AbstractALOHAModel(ufo_name, write_dir=output_dir, format='Fortran')
            print 'done in %2fs' % (time.time() - start)
            
            
            
if __name__ == '__main__':
    Compile_MG5()