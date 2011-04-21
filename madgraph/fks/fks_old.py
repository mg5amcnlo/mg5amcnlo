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

"""Definitions of the objects needed for the implementation of MadFKS"""

import madgraph.core.base_objects as MG
import madgraph.core.helas_objects as helas_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra
import copy
import logging

logger = logging.getLogger('madgraph.helas_objects')
#from math import min

#===============================================================================
# FKS MultiProcess
#===============================================================================
class FKSMultiProcess(diagram_generation.MultiProcess):
    """MultiProcess: list of process definitions
                     list of processes (after cleaning)
                     list of amplitudes (after generation)
    """


    def __init__(self, argument=None):
        """Allow initialization with ProcessDefinition or
        ProcessDefinitionList"""

        if isinstance(argument, MG.ProcessDefinition):
            super(FKSMultiProcess, self).__init__()
            self['process_definitions'].append(argument)
            self.get('amplitudes')
        elif isinstance(argument, MG.ProcessDefinitionList):
            super(FKSMultiProcess, self).__init__()
            self['process_definitions'] = argument
            self.get('amplitudes')
        elif argument != None:
            # call the mother routine
            super(FKSMultiProcess, self).__init__(argument)
        else:
            # call the mother routine
            super(FKSMultiProcess, self).__init__()
        
        for amp in self.get('amplitudes'):
            FKSProcess(amp)




#===============================================================================
# FKS Process
#===============================================================================
class FKSProcess(object):
    """the class for an FKS process. """ 
    

    
    def __init__(self, start_proc = None):
        
        self.fks_dirs = []
        self.leglist = ()
        self.myorders = {}
        self.pdg_codes = []
        self.colors = []
        self.nlegs = 0
        self.fks_ipos = []
        self.fks_j_from_i = {}
#        print "INTERACTIONS  ", start_real_proc['model']['interactions']
#        print "aa", start_amp.keys(), len(start_amp['processes'])
        print "START",start_proc.input_string()
        for n, o in start_proc['orders'].items():
            if n != 'QCD':
                self.myorders[n] = o
            else:
                self.myorders[n] = o -1

 
        self.reduced_processes = []
        if start_proc:
            self.real_proc = start_proc
            self.leglist = self.real_proc.get('legs')
            self.nlegs = len(self.leglist)
            for leg in self.leglist:
                self.pdg_codes.append(leg['id'])
                part = self.real_proc['model'].get('particle_dict')[leg['id']]
                if part['is_part']:
                    self.colors.append(part['color'])
                else:
                    self.colors.append(- part['color'])
                
                
                

     #   print "INTERACTIONS  ", self.real_proc['model']['interactions']
        self.ndirs = 0
        self.fks_config_string = ""
        self.find_directories()
        
        #for i in self.reduced_processes:
         #   print i.input_string(), 'aaa\n'

    
    def find_directories(self):
        """finds the FKS subdirectories for a given process"""
        for i in self.leglist:
            if i['state']:
                for j in self.leglist:
                    if j['number'] != i['number'] :
                        self.combine_ij(i, j)


    def combine_ij(self, i, j):
        """check whether partons i and j can be combined together and if so combine them"""
        part_i = self.real_proc['model'].get('particle_dict')[i['id']]
        part_j = self.real_proc['model'].get('particle_dict')[j['id']]
#        print "PART I", part_i
        ij = None
        if part_i['color'] != 1 and part_j['color'] != 1:
            #check if we have a massless gluon
            if part_i['color'] == 8 and part_i['mass'].lower() == 'zero':
                #ij should be the same as j
                #ij = copy.deepcopy(j)
                ij = MG.Leg(j)
            
            #check if we have two color triplets -> quarks
            elif part_i['color'] == 3 and part_j['color'] == 3 and part_i['mass'].lower() =='zero':
                num = copy.copy(min(i['number'], j['number']))
                #both final state -> i is anti-j
                if j['state']: 
                    if part_i['pdg_code'] == part_j['pdg_code'] and not part_i['is_part'] and part_j['is_part']:
                      #  print "PDG ", part_i['pdg_code'] , part_j['pdg_code'], i, j
                        #ij is an outgoing gluon
                        ij = MG.Leg()
                        ij['id'] = 21
                        ij['number'] = num 
                        ij['state'] = True
                        ij['from_group'] = True
                   #     { 'id' : 21, 'number' : 0, 'state' : True, 'from_group' : True }
                
                else:
                #initial and final state ->i is j
                    if part_i['pdg_code'] == part_j['pdg_code'] and (part_i['is_part'] == part_j['is_part']):
                        #ij is an outgoing gluon
                        ij = MG.Leg()
                        ij['id'] = 21
                        ij['number'] = num 
                        ij['state'] = False
                        ij['from_group'] = True
           
            # initial gluon and final massless quark
            elif part_i['color'] == 3 and part_i['mass'].lower() == 'zero' \
                and part_j['color'] == 8 and not j['state']:
                #ij is the same as i but crossed to the initial state
                #ij = copy.deepcopy(i)
                ij = MG.Leg(i)                
                ij['id'] = - ij['id']
                ij['state'] = False
        
        if ij:
            self.ndirs += 1
            self.reduced_processes.append(self.reduce_real_process(i, j, ij, self.ndirs))
            #generate born amplitude here and call FKSdir
            amp = diagram_generation.Amplitude(self.reduced_processes[-1]['process'])
            
            self.fks_config_string += " \n\
c     FKS configuration number  %(conf_num)d \n\
      data fks_i(  %(conf_num)d  ) /  %(i_fks)d  / \n\
      data fks_j(  %(conf_num)d  ) /  %(j_fks)d  / \n\
      " % self.reduced_processes[-1]
       
#            amp = diagram_generation.Amplitude(self.real_proc)
            #print "THISAMP ",amp 
            self.fks_dirs.append(FKSDirectory(amp, i))
    #re-assign the number of the legs of the original process
 #   for i in range(len(self.leglist)):
 #       self.leglist[i]['number'] = i+1
        
        
    
    def reduce_real_process(self, i, j, ij , nproc ):
        """given the real process, i, j and ij_fks, it defines the underlying born process"""
     
        reduced = MG.Process(self.real_proc)
#        reduced.set('legs', MG.LegList(self.real_proc.get('legs')))
        reduced['legs'] = copy.deepcopy(self.real_proc['legs']) 
        
        reduced.set('orders', self.myorders)

        
        # print reduced['legs']
        

        i_i = copy.copy(reduced['legs'].index(i))
        
        i_j = copy.copy(reduced['legs'].index(j))
 #       print " I J BEFORE ", i_i+1, i_j+1
        

        reduced['legs'].remove(i)
 #       print "after i", " ".join("%d" % (a['number']) for a in self.leglist) 
        
        reduced['legs'].remove(j)
#        print "after j", " ".join("%d" % (a['number']) for a in self.leglist) 

        reduced['legs'].insert(min(i_i,i_j), ij)
#        print "after ij", " ".join("%d" % (a['number']) for a in self.leglist) 

        #redefine leg number
        for nn in range(len(reduced['legs'])):
            reduced['legs'][nn]['number'] = nn+1
      #      del leg['number']
      #      print "LEG after", leg
#        print "after ren", " ".join("%d" % (a['number']) for a in self.leglist) 

        
        ## numbering of particles?????
        #print "ORIGINAL ", self.real_proc.input_string()
        #print "REDUCED  ", reduced.input_string(), "           ",i_i, i_j 
       # print "\n" 
       # print "REDUCED LEGS", reduced['legs']   
       
        if not (i_i + 1) in self.fks_ipos:
            self.fks_ipos.append(i_i + 1)
            self.fks_j_from_i[i_i + 1] = []
        if not (i_j + 1) in self.fks_j_from_i[i_i + 1]:
            self.fks_j_from_i[i_i + 1].append(i_j + 1)

#        print "after", " ".join("%d" % (a['number']) for a in self.leglist) 
#        print " I J ", i_i+1, i_j+1
        return {'process':reduced, 'i_fks':i_i + 1, 'j_fks':i_j + 1, 'ij_fks':ij, 
                'conf_num': nproc} 

            

    

class FKSDirectory(MG.PhysicsObject):
    
    def __init__(self, born_amplitude = None, i_fks = 0):

        self.born_amp = born_amplitude
        self.spectators = []
        self.soft_borns = {}
        self.ifks = i_fks['number'] + 1
        
        if born_amplitude:
            part_i = born_amplitude['process']['model'].get('particle_dict')[i_fks['id']]
            
            #check that the unresolved particle is a gluon, otherwise no soft sing.
            if part_i['color'] == 8 and part_i['mass'].lower() == 'zero':
            #loop over possible color links
                for m in born_amplitude['process']['legs']: 
                    for n in born_amplitude['process']['legs']:
                        part_m =  born_amplitude['process']['model'].get('particle_dict')[m['id']]
                        part_n =  born_amplitude['process']['model'].get('particle_dict')[n['id']]

                        if part_m['color'] != 1 and part_n['color'] != 1:
                            if m != n or part_m['mass'].lower() != 'zero':
                                #  self.generate_color_factor( m, n)
                                self.spectators.append([m,n])
                                helas = FKSHelasMultiProcess(self.born_amp, [m,n])
                                self.soft_borns[(m['number'], n['number'])] = \
                                            helas
#                                helas = helas_objects.HelasMultiProcess(self.born_amp)

    
class FKSHelasMultiProcess(helas_objects.HelasMultiProcess):
    """class inherited from HMP to werite out the FKS matrix element"""
    
    def __init__(self, argument=None, link=[0,0]):
        """Allow initialization with AmplitudeList"""
        
        if isinstance(argument, diagram_generation.AmplitudeList):

            super(FKSHelasMultiProcess, self).__init__()
            self.generate_matrix_elementsFKS(argument, link)
        elif isinstance(argument, diagram_generation.MultiProcess):
            super(FKSHelasMultiProcess, self).__init__()
            self.generate_matrix_elementsFKS(argument.get('amplitudes'), link)
        elif isinstance(argument, diagram_generation.Amplitude):
            super(FKSHelasMultiProcess, self).__init__()
            self.generate_matrix_elementsFKS(\
                diagram_generation.AmplitudeList([argument]), link)
        elif argument:
            #call the mother routine
            super(FKSHelasMultiProcess, self).__init__(argument, link)
        else:
            # call the mother routine
            super(FKSHelasMultiProcess, self).__init__()
                
        
    def generate_matrix_elementsFKS(self, amplitudes, llink):
        """Generate the HelasMatrixElements for the Amplitudes,
        identifying processes with identical matrix elements, as
        defined by HelasMatrixElement.__eq__"""

        assert isinstance(amplitudes, diagram_generation.AmplitudeList), \
                  "%s is not valid AmplitudeList" % repr(amplitudes)

        # Keep track of already generated color objects, to reuse as
        # much as possible
        list_colorize = []
        self.list_color_basis = []
        self.list_color_basis_link = []
        self.list_color_matrices = []
        link = []
        for i in llink:
            link.append(i['number'])

        matrix_elements = self.get('matrix_elements')

        while amplitudes:
    
            # Pop the amplitude to save memory space
            amplitude = amplitudes.pop(0)
            if isinstance(amplitude, diagram_generation.DecayChainAmplitude):
                self.matrix_element_list = HelasDecayChainProcess(amplitude).\
                                      combine_decay_chain_processes()
            else:
                logger.info("Generating Helas calls for %s" % \
                         amplitude.get('process').nice_string().\
                                           replace('Process', 'process'))
                self.matrix_element_list = [helas_objects.HelasMatrixElement(amplitude,
                                                          gen_color=False)]
            for matrix_element in self.matrix_element_list:
                assert isinstance(matrix_element, helas_objects.HelasMatrixElement), \
                          "Not a HelasMatrixElement: %s" % matrix_element

                try:
                    # If an identical matrix element is already in the list,
                    # then simply add this process to the list of
                    # processes for that matrix element
                    other_processes = matrix_elements[\
                    matrix_elements.index(matrix_element)].get('processes')
                    logger.info("Combining process with %s" % \
                      other_processes[0].nice_string().replace('Process: ', ''))
                    other_processes.extend(matrix_element.get('processes'))
                except ValueError:
                    # Otherwise, if the matrix element has any diagrams,
                    # add this matrix element.

                    if matrix_element.get('processes') and \
                           matrix_element.get('diagrams'):

                        matrix_elements.append(matrix_element)
    
                        # Always create an empty color basis, and the
                        # list of raw colorize objects (before
                        # simplification) associated with amplitude
                        col_basis = color_amp.ColorBasis()
                        col_basis_link = color_amp.ColorBasis()
                        new_amp = matrix_element.get_base_amplitude()
                        matrix_element.set('base_amplitude', new_amp)
                        colorize_obj = col_basis.create_color_dict_list(new_amp)
                        print colorize_obj
                        colorize_link = col_basis_link.create_color_dict_list(\
                                        new_amp,link)
                        
                        #print "COLORIZE", colorize_obj 
                        #print "COLORIZE LINK", colorize_link 
                        #print "LINK", link
                        

                        # Create color basis and color
                        # matrix accordingly
                        list_colorize.append(colorize_obj)
                        col_basis.build()
                        col_basis_link.build()
                        self.list_color_basis.append(col_basis)
                        self.list_color_basis_link.append(col_basis_link)
                        #print "BASIS", col_basis 
                        #print "BASIS LINK", col_basis_link                         
                        col_matrix = color_amp.ColorMatrix(col_basis, col_basis_link)

                        #print "COLOR MATRIX", col_matrix
                        self.list_color_matrices.append(col_matrix)
                        
                        col_index = -1
                        logger.info(\
                              "Processing color information for %s" % \
                              matrix_element.get('processes')[0].nice_string().\
                                             replace('Process', 'process'))

                matrix_element.set('color_basis', self.list_color_basis[col_index])
                #print list_color_basis[col_index]
                #print col_basis_link
                matrix_element.set('color_matrix',
                                   self.list_color_matrices[col_index])
            #    print list_color_matrices[col_index]
            
    
    
    
    
    
    