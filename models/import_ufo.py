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
""" How to import a UFO model to the MG5 format """


import fractions
import logging
import os
import re
import sys
import time


from madgraph import MadGraph5Error, MG5DIR, ReadWrite
import madgraph.core.base_objects as base_objects
import madgraph.loop.loop_base_objects as loop_base_objects
import madgraph.core.color_algebra as color
import madgraph.iolibs.files as files
import madgraph.iolibs.save_load_object as save_load_object
from madgraph.core.color_algebra import *
import madgraph.various.misc as misc
import madgraph.iolibs.ufo_expression_parsers as parsers

import aloha
import aloha.create_aloha as create_aloha
import aloha.aloha_fct as aloha_fct

import models as ufomodels
import models.model_reader as model_reader
logger = logging.getLogger('madgraph.model')
logger_mod = logging.getLogger('madgraph.model')

root_path = os.path.dirname(os.path.realpath( __file__ ))
sys.path.append(root_path)

sys.path.append(os.path.join(root_path, os.path.pardir, 'Template', 'bin', 'internal'))
import check_param_card 

pjoin = os.path.join
logger = logging.getLogger("madgraph.model")

# Suffixes to employ for the various poles of CTparameters
pole_dict = {-2:'2EPS',-1:'1EPS',0:'FIN'}

class UFOImportError(MadGraph5Error):
    """ a error class for wrong import of UFO model""" 

class InvalidModel(MadGraph5Error):
    """ a class for invalid Model """

last_model_path =''
def find_ufo_path(model_name):
    """ find the path to a model """

    global last_model_path

    # Check for a valid directory
    if model_name.startswith('./') and os.path.isdir(model_name):
        return model_name
    elif os.path.isdir(os.path.join(MG5DIR, 'models', model_name)):
        return os.path.join(MG5DIR, 'models', model_name)
    elif 'PYTHONPATH' in os.environ:
        for p in os.environ['PYTHONPATH'].split(':'):
            if os.path.isdir(os.path.join(MG5DIR, p, model_name)):
                if last_model_path != os.path.join(MG5DIR, p, model_name):
                    logger.info("model loaded from PYTHONPATH: %s", os.path.join(MG5DIR, p, model_name))
                    last_model_path = os.path.join(MG5DIR, p, model_name)
                return os.path.join(MG5DIR, p, model_name)
    if os.path.isdir(model_name):
        if last_model_path != os.path.join(MG5DIR, p, model_name):
            logger.info("model loaded from: %s", os.path.join(os.getcwd(), model_name))
            last_model_path = os.path.join(MG5DIR, p, model_name)
        return model_name   
    else:
        raise UFOImportError("Path %s is not a valid pathname" % model_name)    
    

    return model_path

def import_model(model_name, decay=False, restrict=True, prefix='mdl_',
                                                    complex_mass_scheme = None):
    """ a practical and efficient way to import a model"""
    
    # check if this is a valid path or if this include restriction file       
    try:
        model_path = find_ufo_path(model_name)
    except UFOImportError:
        if '-' not in model_name:
            raise
        split = model_name.split('-')
        model_name = '-'.join([text for text in split[:-1]])
        model_path = find_ufo_path(model_name)
        restrict_name = split[-1]
         
        restrict_file = os.path.join(model_path, 'restrict_%s.dat'% restrict_name)
        
        #if restriction is full, then we by pass restriction (avoid default)
        if split[-1] == 'full':
            restrict_file = None
    else:
        # Check if by default we need some restrictions
        restrict_name = ""
        if restrict and os.path.exists(os.path.join(model_path,'restrict_default.dat')):
            restrict_file = os.path.join(model_path,'restrict_default.dat')
        else:
            restrict_file = None

        if isinstance(restrict, str):
            if os.path.exists(os.path.join(model_path, restrict)):
                restrict_file = os.path.join(model_path, restrict)
            elif os.path.exists(restrict):
                restrict_file = restrict
            else:
                raise Exception, "%s is not a valid path for restrict file" % restrict
    
    #import the FULL model
    model = import_full_model(model_path, decay, prefix)

    if os.path.exists(pjoin(model_path, "README")):
        logger.info("Please read carefully the README of the model file for instructions/restrictions of the model.",'$MG:color:BLACK') 
    # restore the model name
    if restrict_name:
        model["name"] += '-' + restrict_name
    
    # Decide whether complex mass scheme is on or not
    useCMS = (complex_mass_scheme is None and aloha.complex_mass) or \
                                                      complex_mass_scheme==True
    #restrict it if needed       
    if restrict_file:
        try:
            logger.info('Restrict model %s with file %s .' % (model_name, os.path.relpath(restrict_file)))
        except OSError:
            # sometimes has trouble with relative path
            logger.info('Restrict model %s with file %s .' % (model_name, restrict_file))
            
        if logger_mod.getEffectiveLevel() > 10:
            logger.info('Run \"set stdout_level DEBUG\" before import for more information.')
        # Modify the mother class of the object in order to allow restriction
        model = RestrictModel(model)

        # Change to complex mass scheme if necessary. This must be done BEFORE
        # the restriction.
        if useCMS:
            # We read the param_card a first time so that the function 
            # change_mass_to_complex_scheme can know if a particle is to
            # be considered massive or not and with zero width or not.
            # So we read the restrict card a first time, with the CMS set to
            # False because we haven't changed the model yet.
            model.set_parameters_and_couplings(param_card = restrict_file,
                                                      complex_mass_scheme=False)
            model.change_mass_to_complex_scheme(toCMS=True)
        else:
            # Make sure that the parameter 'CMSParam' of the model is set to 0.0
            # as it should in order to have the correct NWA renormalization condition.
            # It might be that the default of the model is CMS.
            model.change_mass_to_complex_scheme(toCMS=False)

        if model_name == 'mssm' or os.path.basename(model_name) == 'mssm':
            keep_external=True
        else:
            keep_external=False
        model.restrict_model(restrict_file, rm_parameter=not decay,
           keep_external=keep_external, complex_mass_scheme=complex_mass_scheme)
        model.path = model_path
    else:
        # Change to complex mass scheme if necessary
        if useCMS:
            model.change_mass_to_complex_scheme(toCMS=True)
        else:
            # It might be that the default of the model (i.e. 'CMSParam') is CMS.
            model.change_mass_to_complex_scheme(toCMS=False)

        
    return model
    

_import_once = []
def import_full_model(model_path, decay=False, prefix=''):
    """ a practical and efficient way to import one of those models 
        (no restriction file use)"""

    assert model_path == find_ufo_path(model_path)
    
    if prefix is True:
        prefix='mdl_'
        
    # Check the validity of the model
    files_list_prov = ['couplings.py','lorentz.py','parameters.py',
                       'particles.py', 'vertices.py', 'function_library.py',
                       'propagators.py', 'coupling_orders.py']
    
    if decay:
        files_list_prov.append('decays.py')    
    
    files_list = []
    for filename in files_list_prov:
        filepath = os.path.join(model_path, filename)
        if not os.path.isfile(filepath):
            if filename not in ['propagators.py', 'decays.py', 'coupling_orders.py']:
                raise UFOImportError,  "%s directory is not a valid UFO model: \n %s is missing" % \
                                                         (model_path, filename)
        files_list.append(filepath)
    # use pickle files if defined and up-to-date
    if aloha.unitary_gauge: 
        pickle_name = 'model.pkl'
    else:
        pickle_name = 'model_Feynman.pkl'
    if decay:
        pickle_name = 'dec_%s' % pickle_name
    
    allow_reload = False
    if files.is_uptodate(os.path.join(model_path, pickle_name), files_list):
        allow_reload = True
        try:
            model = save_load_object.load_from_file( \
                                          os.path.join(model_path, pickle_name))
        except Exception, error:
            logger.info('failed to load model from pickle file. Try importing UFO from File')
        else:
            # We don't care about the restrict_card for this comparison
            if model.has_key('version_tag') and not model.get('version_tag') is None and \
                model.get('version_tag').startswith(os.path.realpath(model_path)) and \
                model.get('version_tag').endswith('##' + str(misc.get_pkg_info())):
                #check if the prefix is correct one.
                for key in model.get('parameters'):
                    for param in model['parameters'][key]:
                        value = param.name.lower()
                        if value in ['as','mu_r', 'zero','aewm1']:
                            continue
                        if prefix:
                            if value.startswith(prefix):
                                _import_once.append((model_path, aloha.unitary_gauge, prefix, decay))
                                return model
                            else:
                                logger.info('reload from .py file')
                                break
                        else:
                            if value.startswith('mdl_'):
                                logger.info('reload from .py file')
                                break                   
                            else:
                                _import_once.append((model_path, aloha.unitary_gauge, prefix, decay))
                                return model
                    else:
                        continue
                    break                                         
            else:
                logger.info('reload from .py file')

    if (model_path, aloha.unitary_gauge, prefix, decay) in _import_once and not allow_reload:
        raise MadGraph5Error, 'This model %s is modified on disk. To reload it you need to quit/relaunch MG5_aMC ' % model_path
     
    # Load basic information
    ufo_model = ufomodels.load_model(model_path, decay)
    ufo2mg5_converter = UFOMG5Converter(ufo_model)    
    model = ufo2mg5_converter.load_model()
    if model_path[-1] == '/': model_path = model_path[:-1] #avoid empty name
    model.set('name', os.path.split(model_path)[-1])

    # Load the Parameter/Coupling in a convenient format.
    parameters, couplings = OrganizeModelExpression(ufo_model).main(\
             additional_couplings =(ufo2mg5_converter.wavefunction_CT_couplings
                           if ufo2mg5_converter.perturbation_couplings else []))
    
    model.set('parameters', parameters)
    model.set('couplings', couplings)
    model.set('functions', ufo_model.all_functions)

    # Optional UFO part: decay_width information


    if decay and hasattr(ufo_model, 'all_decays') and ufo_model.all_decays:       
        start = time.time()
        for ufo_part in ufo_model.all_particles:
            name =  ufo_part.name
            if not model['case_sensitive']:
                name = name.lower() 
            p = model['particles'].find_name(name)
            if hasattr(ufo_part, 'partial_widths'):
                p.partial_widths = ufo_part.partial_widths
            elif p and not hasattr(p, 'partial_widths'):
                p.partial_widths = {}
            # might be None for ghost
        logger.debug("load width takes %s", time.time()-start)
    
    if prefix:
        start = time.time()
        model.change_parameter_name_with_prefix()
        logger.debug("model prefixing  takes %s", time.time()-start)
                     
    path = os.path.dirname(os.path.realpath(model_path))
    path = os.path.join(path, model.get('name'))
    model.set('version_tag', os.path.realpath(path) +'##'+ str(misc.get_pkg_info()))
    
    # save in a pickle files to fasten future usage
    if ReadWrite:
        save_load_object.save_to_file(os.path.join(model_path, pickle_name),
                                   model, log=False)

    #if default and os.path.exists(os.path.join(model_path, 'restrict_default.dat')):
    #    restrict_file = os.path.join(model_path, 'restrict_default.dat') 
    #    model = import_ufo.RestrictModel(model)
    #    model.restrict_model(restrict_file)

    return model

class UFOMG5Converter(object):
    """Convert a UFO model to the MG5 format"""

    def __init__(self, model, auto=False):
        """ initialize empty list for particles/interactions """
       
        self.particles = base_objects.ParticleList()
        self.interactions = base_objects.InteractionList()
        self.wavefunction_CT_couplings = []
 
        # Check here if we can extract the couplings perturbed in this model
        # which indicate a loop model or if this model is only meant for 
        # tree-level computations
        self.perturbation_couplings = {}
        try:
            for order in model.all_orders:
                if(order.perturbative_expansion>0):
                    self.perturbation_couplings[order.name]=order.perturbative_expansion
        except AttributeError,error:
            pass

        if self.perturbation_couplings!={}:
            self.model = loop_base_objects.LoopModel({'perturbation_couplings':\
                                                self.perturbation_couplings.keys()})
        else:
            self.model = base_objects.Model()                        
        self.model.set('particles', self.particles)
        self.model.set('interactions', self.interactions)
        self.conservecharge = set(['charge'])
        
        self.ufomodel = model
        self.checked_lor = set()

        if auto:
            self.load_model()

    def load_model(self):
        """load the different of the model first particles then interactions"""

        # Check the validity of the model
        # 1) check that all lhablock are single word.
        def_name = []
        for param in self.ufomodel.all_parameters:
            if param.nature == "external":
                if len(param.lhablock.split())>1:
                    raise InvalidModel, '''LHABlock should be single word which is not the case for
    \'%s\' parameter with lhablock \'%s\' ''' % (param.name, param.lhablock)
            if param.name in def_name:
                raise InvalidModel, "name %s define multiple time. Please correct the UFO model!" \
                                                                  % (param.name)
            else:
                def_name.append(param.name)
                                                                  
        # For each CTParameter, check that there is no name conflict with the
        # set of re-defined CTParameters with EPS and FIN suffixes.
        if hasattr(self.ufomodel,'all_CTparameters'):
            for CTparam in self.ufomodel.all_CTparameters:
                for pole in pole_dict:
                    if CTparam.pole(pole)!='ZERO':
                        new_param_name = '%s_%s_'%(CTparam.name,pole_dict[pole])
                        if new_param_name in def_name:
                            raise InvalidModel, "CT name %s"% (new_param_name)+\
                                           " the model. Please change its name."

        if hasattr(self.ufomodel, 'gauge'):    
            self.model.set('gauge', self.ufomodel.gauge)
        logger.info('load particles')
        # Check if multiple particles have the same name but different case.
        # Otherwise, we can use lowercase particle names.
        if len(set([p.name for p in self.ufomodel.all_particles] + \
                   [p.antiname for p in self.ufomodel.all_particles])) == \
           len(set([p.name.lower() for p in self.ufomodel.all_particles] + \
                   [p.antiname.lower() for p in self.ufomodel.all_particles])):
            self.model['case_sensitive'] = False
            

        # check which of the fermion/anti-fermion should be set as incoming
        self.detect_incoming_fermion()

        for particle_info in self.ufomodel.all_particles:            
            self.add_particle(particle_info)

        # Find which particles is in the 3/3bar color states (retrun {id: 3/-3})
        color_info = self.find_color_anti_color_rep()

        # load the lorentz structure.
        self.model.set('lorentz', self.ufomodel.all_lorentz)
        
        # Substitute the expression of CT couplings which include CTparameters
        # in their definition with the corresponding dictionaries, e.g.
        #      CTCoupling.value = 2*CTParam           ->
        #      CTCoupling.value = {-1: 2*CTParam_1EPS_, 0: 2*CTParam_FIN_}
        # for example if CTParam had a non-zero finite and single pole.
        # This change affects directly the UFO model and it will be reverted in
        # OrganizeModelExpression only, so that the main() function of this class
        # *must* be run on the UFO to have this change reverted.
        if hasattr(self.ufomodel,'all_CTparameters'):
            logger.debug('Handling couplings defined with CTparameters...')
            start_treat_coupling = time.time()
            self.treat_couplings(self.ufomodel.all_couplings, 
                                                 self.ufomodel.all_CTparameters)
            tot_time = time.time()-start_treat_coupling
            if tot_time>5.0:
                logger.debug('... done in %s'%misc.format_time(tot_time))        

        logger.info('load vertices')
        for interaction_info in self.ufomodel.all_vertices:
            self.add_interaction(interaction_info, color_info)

        if self.perturbation_couplings:
            try:
                self.ufomodel.add_NLO()
            except Exception, error:
                pass 

            for interaction_info in self.ufomodel.all_CTvertices:
                self.add_CTinteraction(interaction_info, color_info)
    
        self.model.set('conserved_charge', self.conservecharge)

        # If we deal with a Loop model here, the order hierarchy MUST be 
        # defined in the file coupling_orders.py and we import it from 
        # there.
        all_orders = []
        try:
            all_orders = self.ufomodel.all_orders
        except AttributeError:
            if self.perturbation_couplings:
                raise MadGraph5Error, "The loop model MG5 attemps to import does not specify the attribute 'all_order'." 
            else:
                pass            

        hierarchy={}
        try:
            for order in all_orders:
                hierarchy[order.name]=order.hierarchy
        except AttributeError:
            if self.perturbation_couplings:
                raise MadGraph5Error, 'The loop model MG5 attemps to import does not specify an order hierarchy.' 
            else:
                pass
        else:
            self.model.set('order_hierarchy', hierarchy)            
        
        # Also set expansion_order, i.e., maximum coupling order per process
        expansion_order={}
        # And finally the UVCT coupling order counterterms        
        coupling_order_counterterms={}        
        try:
            for order in all_orders:
                expansion_order[order.name]=order.expansion_order
                coupling_order_counterterms[order.name]=order.expansion_order                
        except AttributeError:
            if self.perturbation_couplings:
                raise MadGraph5Error, 'The loop model MG5 attemps to import does not specify an expansion_order for all coupling orders.' 
            else:
                pass
        else:
            self.model.set('expansion_order', expansion_order)
            self.model.set('expansion_order', expansion_order)            
            
        #clean memory
        del self.checked_lor

        return self.model
        
    
    def add_particle(self, particle_info):
        """ convert and add a particle in the particle list """
                
        loop_particles = [[[]]]
        counterterms = {}
        
        # MG5 have only one entry for particle and anti particles.
        #UFO has two. use the color to avoid duplictions
        pdg = particle_info.pdg_code
        if pdg in self.incoming or (pdg not in self.outcoming and pdg <0):
            return
        
        # MG5 doesn't use ghost for tree models: physical sum on the polarization
        if not self.perturbation_couplings and particle_info.spin < 0:
            return
        
        if (aloha.unitary_gauge and 0 in self.model['gauge']) \
                            or (1 not in self.model['gauge']): 
        
            # MG5 doesn't use goldstone boson 
            if hasattr(particle_info, 'GoldstoneBoson') and particle_info.GoldstoneBoson:
                return
            elif hasattr(particle_info, 'goldstone') and particle_info.goldstone:
                return      
        # Initialize a particles
        particle = base_objects.Particle()

        # MG5 doesn't use goldstone boson 
        if (hasattr(particle_info, 'GoldstoneBoson') and particle_info.GoldstoneBoson) \
                or (hasattr(particle_info, 'goldstoneboson') and particle_info.goldstoneboson):
            particle.set('type', 'goldstone')
        elif hasattr(particle_info, 'goldstone') and particle_info.goldstone:
            particle.set('type', 'goldstone')
        
        nb_property = 0   #basic check that the UFO information is complete
        # Loop over the element defining the UFO particles
        for key,value in particle_info.__dict__.items():
            # Check if we use it in the MG5 definition of a particles
            if key in base_objects.Particle.sorted_keys and not key=='counterterm':
                nb_property +=1
                if key in ['name', 'antiname']:
                    if not self.model['case_sensitive']:
                        particle.set(key, value.lower())
                    else:
                        particle.set(key, value)
                elif key == 'charge':
                    particle.set(key, float(value))
                elif key in ['mass','width']:
                    particle.set(key, str(value))
                elif key == 'spin':
                    # MG5 internally treats ghost with positive spin for loop models and 
                    # ignore them otherwise
                    particle.set(key,abs(value))
                    if value<0:
                        particle.set('type','ghost')
                elif key == 'propagating':
                    if not value:
                        particle.set('line', None)
                elif key == 'line':
                    if particle.get('line') is None:
                        pass # This means that propagating is on False 
                    else:
                        particle.set('line', value)
                elif key == 'propagator':
                    if value:
                        if aloha.unitary_gauge:
                            particle.set(key, str(value[0]))
                        else: 
                            particle.set(key, str(value[1]))
                    else:
                        particle.set(key, '')
                else:
                    particle.set(key, value)    
            elif key == 'loop_particles':
                loop_particles = value
            elif key == 'counterterm':
                counterterms = value
            elif key.lower() not in ('ghostnumber','selfconjugate','goldstone',
                                             'goldstoneboson','partial_widths',
                                     'texname', 'antitexname', 'propagating', 'ghost'
                                             ):
                # add charge -we will check later if those are conserve 
                self.conservecharge.add(key)
                particle.set(key,value, force=True)

        if not hasattr(particle_info, 'propagator'):
            nb_property += 1
            if particle.get('spin') >= 3:
                if particle.get('mass').lower() == 'zero':
                    particle.set('propagator', 0) 
                elif particle.get('spin') == 3 and not aloha.unitary_gauge:
                    particle.set('propagator', 0)
               
        assert(10 == nb_property) #basic check that all the information is there         
        
        # Identify self conjugate particles
        if particle_info.name == particle_info.antiname:
            particle.set('self_antipart', True)
                    
        # Proceed only if we deal with a loop model and that this particle
        # has wavefunction renormalization
        if not self.perturbation_couplings or counterterms=={}:
            self.particles.append(particle)
            return
        
        # Set here the 'counterterm' attribute to the particle.
        # First we must change the couplings dictionary keys from the entry format
        # (order1,order2,...,orderN,loop_particle#):LaurentSerie
        # two a dictionary with format 
        # ('ORDER_OF_COUNTERTERM',((Particle_list_PDG))):{laurent_order:CTCouplingName}
        particle_counterterms = {}
        for key, counterterm in counterterms.items():
            # Makes sure this counterterm contributes at one-loop.
            if len([1 for k in key[:-1] if k==1])==1 and \
               not any(k>1 for k in key[:-1]):
                newParticleCountertermKey=[None,\
#                  The line below is for loop UFO Model with the 'attribute' 
#                  'loop_particles' of the Particle objects to be defined with
#                  instances of the particle class. The new convention is to use
#                  pdg numbers instead.
#                  tuple([tuple([abs(part.pdg_code) for part in loop_parts]) for\
                  tuple([tuple(loop_parts) for\
                    loop_parts in loop_particles[key[-1]]])]
                for i, order in enumerate(self.ufomodel.all_orders[:-1]):
                    if key[i]==1:
                        newParticleCountertermKey[0]=order.name
                newCouplingName='UVWfct_'+particle_info.name+'_'+str(key[-1])
                particle_counterterms[tuple(newParticleCountertermKey)]=\
                  dict([(key,newCouplingName+('' if key==0 else '_'+str(-key)+'eps'))\
                        for key in counterterm])
                # We want to create the new coupling for this wavefunction
                # renormalization.
                self.ufomodel.object_library.Coupling(\
                    name = newCouplingName,
                    value = counterterm,
                    order = {newParticleCountertermKey[0]:2})
                self.wavefunction_CT_couplings.append(self.ufomodel.all_couplings.pop())

        particle.set('counterterm',particle_counterterms)
        self.particles.append(particle)
        return

    def treat_couplings(self, couplings, all_CTparameters):
        """ This function scan each coupling to see if it contains a CT parameter.
        when it does, it changes its value to a dictionary with the CT parameter
        changed to a new parameter for each pole and finite part. For instance,
        the following coupling:
              coupling.value = '2*(myCTParam1 + myParam*(myCTParam2 + myCTParam3)'
        with CTparameters
              myCTParam1 = {0: Something, -1: SomethingElse}
              myCTParam2 = {0: OtherSomething }
              myCTParam3 = {-1: YetOtherSomething }              
        would be turned into
              coupling.value = {0: '2*(myCTParam1_FIN_ + myParam*(myCTParam2_FIN_ + ZERO)'
                               -1: '2*(myCTParam1_EPS_ + myParam*(ZERO + myCTParam2_EPS_)'}              
        
        all_CTParameter is the list of all CTParameters in the model"""
        
        # First define a list of regular expressions for each CT parameter 
        # and put them in a dictionary whose keys are the CT parameter names
        # and the values are a tuple with the substituting patter in the first
        # entry and the list of substituting functions (one for each pole)
        # as the second entry of this tuple.
        CTparameter_patterns = {}
        zero_substitution = lambda matchedObj: matchedObj.group('first')+\
                                               'ZERO'+matchedObj.group('second')
        def function_factory(arg):
            return lambda matchedObj: \
                    matchedObj.group('first')+arg+matchedObj.group('second')
        for CTparam in all_CTparameters:
            pattern_finder = re.compile(r"(?P<first>\A|\*|\+|\-|\(|\s)(?P<name>"+
                        CTparam.name+r")(?P<second>\Z|\*|\+|\-|\)|/|\\|\s)")
            
            sub_functions = [None if CTparam.pole(pole)=='ZERO' else
                function_factory('%s_%s_'%(CTparam.name,pole_dict[-pole]))
                                                           for pole in range(3)]
            CTparameter_patterns[CTparam.name] = (pattern_finder,sub_functions)
        
        times_zero = re.compile('\*\s*-?ZERO')
        zero_times = re.compile('ZERO\s*(\*|\/)')
        def is_expr_zero(expresson):
            """ Checks whether a single term (involving only the operations
            * or / is zero. """
            for term in expresson.split('-'):
                for t in term.split('+'):
                    t = t.strip()
                    if t in ['ZERO','']:
                        continue
                    if not (times_zero.search(t) or zero_times.search(t)):
                        return False
            return True
        
        def find_parenthesis(expr):
            end = expr.find(')')
            if end == -1:
                return None
            start = expr.rfind('(',0,end+1)
            if start ==-1:
                raise InvalidModel,\
                               'Parenthesis of expression %s are malformed'%expr
            return [expr[:start],expr[start+1:end],expr[end+1:]]
        
        start_parenthesis = re.compile(r".*\s*[\+\-\*\/\)\(]\s*$")

        def is_value_zero(value):
            """Check whether an expression like ((A+B)*ZERO+C)*ZERO is zero.
            Only +,-,/,* operations are allowed and 'ZERO' is a tag for an
            analytically zero quantity."""

            curr_value = value
            parenthesis = find_parenthesis(curr_value)
            while parenthesis:
                # Allow the complexconjugate function
                if parenthesis[0].endswith('complexconjugate'):
                    # Then simply remove it
                    parenthesis[0] = parenthesis[0][:-16]
                if parenthesis[0]=='' or re.match(start_parenthesis,
                                                                parenthesis[0]):
                    if is_value_zero(parenthesis[1]):
                        new_parenthesis = 'ZERO'
                    else:
                        new_parenthesis = 'PARENTHESIS'
                else:
                    new_parenthesis = '_FUNCTIONARGS'
                curr_value = parenthesis[0]+new_parenthesis+parenthesis[2]
                parenthesis = find_parenthesis(curr_value)
            return is_expr_zero(curr_value)

        def CTCoupling_pole(CTCoupling, pole):
            """Compute the pole of the CTCoupling in two cases:
               a) Its value is a dictionary, then just return the corresponding
                  entry in the dictionary.
               b) It is expressed in terms of CTParameters which are themselves
                  dictionary so we want to substitute their expression to get
                  the value of the pole. In the current implementation, this is
                  just to see if the pole is zero or not.
            """

            if isinstance(CTCoupling.value,dict):
                if -pole in CTCoupling.value.keys():
                    return CTCoupling.value[-pole], [], 0
                else:
                    return 'ZERO', [], 0              

            new_expression           = CTCoupling.value
            CTparamNames = []
            n_CTparams   = 0
            for paramname, value in CTparameter_patterns.items():
                pattern = value[0]
                # Keep track of which CT parameters enter in the definition of
                # which coupling.
                if not re.search(pattern,new_expression):
                    continue
                n_CTparams += 1
                # If the contribution of this CTparam to this pole is non
                # zero then the substituting function is not None:
                if not value[1][pole] is None:
                    CTparamNames.append('%s_%s_'%(paramname,pole_dict[-pole]))
  
                substitute_function = zero_substitution if \
                                      value[1][pole] is None else value[1][pole]
                new_expression = pattern.sub(substitute_function,new_expression)

            # If no CTParam was found and we ask for a pole, then it can only
            # be zero.
            if pole!=0 and n_CTparams==0:
                return 'ZERO', [], n_CTparams

            # Check if resulting expression is analytically zero or not.
            # Remember that when the value of a CT_coupling is not a dictionary
            # then the only operators allowed in the definition are +,-,*,/
            # and each term added or subtracted must contain *exactly one*
            # CTParameter and never at the denominator.   
            if n_CTparams > 0 and is_value_zero(new_expression):
                return 'ZERO', [], n_CTparams
            else:
                return new_expression, CTparamNames, n_CTparams

        # For each coupling we substitute its value if necessary
        for coupl in couplings:
            new_value = {}
            for pole in range(0,3):
                expression, CTparamNames, n_CTparams = CTCoupling_pole(coupl, pole)
                # Make sure it uses CT parameters, otherwise do nothing
                if n_CTparams == 0:
                    break
                elif expression!='ZERO':
                    new_value[-pole] = expression
                    couplname = coupl.name
                    if pole!=0:
                        couplname += "_%deps"%pole
                    # Add the parameter dependency found to the dependency map
                    # of the model being built. In principle, since we should
                    # be building a loop model now, it should always have this
                    # attribute defined, but it is better to make sure.
                    if hasattr(self.model, 'map_CTcoup_CTparam'):
                        self.model.map_CTcoup_CTparam[couplname] = CTparamNames

            # Finally modify the value of this CTCoupling so that it is no
            # longer a string expression in terms of CTParameters but rather
            # a dictionary with the CTparameters replaced by their _FIN_ and
            # _EPS_ counterparts.
            # This is useful for the addCT_interaction() step. I will be reverted
            # right after the addCT_interaction() function so as to leave
            # the UFO intact, as it should. 
            if new_value:
                coupl.old_value = coupl.value
                coupl.value = new_value

    def add_CTinteraction(self, interaction, color_info):
        """ Split this interaction in order to call add_interaction for
        interactions for each element of the loop_particles list. Also it
        is necessary to unfold here the contributions to the different laurent
        expansion orders of the couplings."""

        # Work on a local copy of the interaction provided
        interaction_info=copy.copy(interaction)
        
        intType=''
        if interaction_info.type not in ['UV','UVloop','UVtree','UVmass','R2']:
            raise MadGraph5Error, 'MG5 only supports the following types of'+\
              ' vertices, R2, UV and UVmass. %s is not in this list.'%interaction_info.type
        else:
            intType=interaction_info.type
            # If not specified and simply set to UV, guess the appropriate type
            if interaction_info.type=='UV':
                if len(interaction_info.particles)==2 and interaction_info.\
                          particles[0].name==interaction_info.particles[1].name:
                    intType='UVmass'
                else:
                    intType='UVloop'
        
        # Make sure that if it is a UV mass renromalization counterterm it is
        # defined as such.
#        if len(intType)>2 and intType[:2]=='UV' and len(interaction_info.particles)==2 \
#           and interaction_info.particles[0].name==interaction_info.particles[1].name:
#            intType='UVmass'

        # Now we create a couplings dictionary for each element of the loop_particles list
        # and for each expansion order of the laurent serie in the coupling.
        # and for each coupling order
        # Format is new_couplings[loop_particles][laurent_order] and each element
        # is a couplings dictionary.
        order_to_interactions= {}
        # will contains the new coupling of form
        #new_couplings=[[{} for j in range(0,3)] for i in \
        #               range(0,max(1,len(interaction_info.loop_particles)))]
        # So sort all entries in the couplings dictionary to put them a the
        # correct place in new_couplings.
        for key, couplings in interaction_info.couplings.items():
            if not isinstance(couplings, list):
                couplings = [couplings]
            for coupling in couplings:
                order = tuple(coupling.order.items())
                if order not in order_to_interactions:
                    order_to_interactions[order] = [
                           [{} for j in range(0,3)] for i in \
                           range(0,max(1,len(interaction_info.loop_particles)))]
                    new_couplings = order_to_interactions[order]
                else:
                    new_couplings = order_to_interactions[order]
                    
                for poleOrder in range(0,3):
                    expression = coupling.pole(poleOrder)
                    if expression!='ZERO':
                        if poleOrder==2:
                            raise InvalidModel, """
    The CT coupling %s was found with a contribution to the double pole. 
    This is either an error in the model or a parsing error in the function 'is_value_zero'.
    The expression of the non-zero double pole coupling is:
    %s
    """%(coupling.name,str(coupling.value))
                        # It is actually safer that the new coupling associated to
                        # the interaction added is not a reference to an original 
                        # coupling in the ufo model. So copy.copy is right here.   
                        newCoupling = copy.copy(coupling)
                        if poleOrder!=0:
                            newCoupling.name=newCoupling.name+"_"+str(poleOrder)+"eps"
                        newCoupling.value = expression
                        # assign the CT parameter dependences
                        #if hasattr(coupling,'CTparam_dependence') and \
                        #        (-poleOrder in coupling.CTparam_dependence) and \
                        #        coupling.CTparam_dependence[-poleOrder]:
                        #    newCoupling.CTparam_dependence = coupling.CTparam_dependence[-poleOrder]
                        #elif hasattr(newCoupling,'CTparam_dependence'):
                        #    delattr(newCoupling,"CTparam_dependence")
                        new_couplings[key[2]][poleOrder][(key[0],key[1])] = newCoupling  
            
        for new_couplings in order_to_interactions.values():
            # Now we can add an interaction for each.         
            for i, all_couplings in enumerate(new_couplings):
                loop_particles=[[]]
                if len(interaction_info.loop_particles)>0:
                    loop_particles=[[part.pdg_code for part in loop_parts] \
                        for loop_parts in interaction_info.loop_particles[i]]
                for poleOrder in range(0,3):
                    if all_couplings[poleOrder]!={}:
                        interaction_info.couplings=all_couplings[poleOrder]
                        self.add_interaction(interaction_info, color_info,\
                          (intType if poleOrder==0 else (intType+str(poleOrder)+\
                                                             'eps')),loop_particles)


    def find_color_anti_color_rep(self, output=None):
        """find which color are in the 3/3bar states"""
        # method look at the 3 3bar 8 configuration.
        # If the color is T(3,2,1) and the interaction F1 F2 V
        # Then set F1 to anticolor (and F2 to color)
        # if this is T(3,1,2) set the opposite
        if not output:
            output = {}
             
        for interaction_info in self.ufomodel.all_vertices:
            if len(interaction_info.particles) != 3:
                continue
            colors = [abs(p.color) for p in interaction_info.particles]
            if colors[:2] == [3,3]:
                if 'T(3,2,1)' in interaction_info.color:
                    color, anticolor, other = interaction_info.particles
                elif 'T(3,1,2)' in interaction_info.color:
                    anticolor, color, _ = interaction_info.particles
                elif 'Identity(1,2)' in interaction_info.color  or \
                     'Identity(2,1)' in interaction_info.color:
                    first, second, _ = interaction_info.particles
                    if first.pdg_code in output:
                        if output[first.pdg_code] == 3:
                            color, anticolor = first, second
                        else:
                            color, anticolor = second, first
                    elif second.pdg_code in output:
                        if output[second.pdg_code] == 3:
                            color, anticolor = second, first                        
                        else:
                            color, anticolor = first, second
                    else:
                        continue
                else:
                    continue
            elif colors[1:] == [3,3]:
                if 'T(1,2,3)' in interaction_info.color:
                    other, anticolor, color = interaction_info.particles
                elif 'T(1,3,2)' in interaction_info.color:
                    other, color, anticolor = interaction_info.particles
                elif 'Identity(2,3)' in interaction_info.color  or \
                     'Identity(3,2)' in interaction_info.color:
                    _, first, second = interaction_info.particles
                    if first.pdg_code in output:
                        if output[first.pdg_code] == 3:
                            color, anticolor = first, second
                        else:
                            color, anticolor = second, first
                    elif second.pdg_code in output:
                        if output[second.pdg_code] == 3:
                            color, anticolor = second, first                        
                        else:
                            color, anticolor = first, second
                    else:
                        continue
                else:
                    continue                  
               
            elif colors.count(3) == 2:
                if 'T(2,3,1)' in interaction_info.color:
                    color, other, anticolor = interaction_info.particles
                elif 'T(2,1,3)' in interaction_info.color:
                    anticolor, other, color = interaction_info.particles
                elif 'Identity(1,3)' in interaction_info.color  or \
                     'Identity(3,1)' in interaction_info.color:
                    first, _, second = interaction_info.particles
                    if first.pdg_code in output:
                        if output[first.pdg_code] == 3:
                            color, anticolor = first, second
                        else:
                            color, anticolor = second, first
                    elif second.pdg_code in output:
                        if output[second.pdg_code] == 3:
                            color, anticolor = second, first                        
                        else:
                            color, anticolor = first, second
                    else:
                        continue
                else:
                    continue                 
            else:
                continue    
            
            # Check/assign for the color particle
            if color.pdg_code in output: 
                if output[color.pdg_code] == -3:
                    raise InvalidModel, 'Particles %s is sometimes in the 3 and sometimes in the 3bar representations' \
                                    % color.name
            else:
                output[color.pdg_code] = 3
            
            # Check/assign for the anticolor particle
            if anticolor.pdg_code in output: 
                if output[anticolor.pdg_code] == 3:
                    raise InvalidModel, 'Particles %s is sometimes set as in the 3 and sometimes in the 3bar representations' \
                                    % anticolor.name
            else:
                output[anticolor.pdg_code] = -3
        
        return output
    
    def detect_incoming_fermion(self):
        """define which fermion should be incoming
           for that we look at F F~ X interactions
        """
        self.incoming = [] 
        self.outcoming = []       
        for interaction_info in self.ufomodel.all_vertices:
            # check if the interaction meet requirements:
            pdg = [p.pdg_code for p in interaction_info.particles if p.spin in [2,4]]
            if len(pdg) % 2:
                raise InvalidModel, 'Odd number of fermion in vertex: %s' % [p.pdg_code for p in interaction_info.particles]
            for i in range(0, len(pdg),2):
                if pdg[i] == - pdg[i+1]:
                    if pdg[i] in self.outcoming:
                        raise InvalidModel, '%s has not coherent incoming/outcoming status between interactions' %\
                            [p for p in interaction_info.particles if p.spin in [2,4]][i].name
                            
                    elif not pdg[i] in self.incoming:
                        self.incoming.append(pdg[i])
                        self.outcoming.append(pdg[i+1])
                     
    def add_interaction(self, interaction_info, color_info, type='base', loop_particles=None):            
        """add an interaction in the MG5 model. interaction_info is the 
        UFO vertices information."""
        # Import particles content:
        particles = [self.model.get_particle(particle.pdg_code) \
                                    for particle in interaction_info.particles]
        if None in particles:
            # Interaction with a ghost/goldstone
            return 
        particles = base_objects.ParticleList(particles)

        # Import Lorentz content:
        lorentz = [helas for helas in interaction_info.lorentz]            
        
        # Check the coherence of the Fermion Flow
        nb_fermion = sum([ 1 if p.is_fermion() else 0 for p in particles])
        try:
            if nb_fermion == 2:
                # Fermion Flow is suppose to be dealt by UFO
                [aloha_fct.check_flow_validity(helas.structure, nb_fermion) \
                                          for helas in interaction_info.lorentz
                                          if helas.name not in self.checked_lor]
                self.checked_lor.update(set([helas.name for helas in interaction_info.lorentz]))
            elif nb_fermion:
                if any(p.selfconjugate for p in interaction_info.particles if p.spin % 2 == 0):
                    text = "Majorana can not be dealt in 4/6/... fermion interactions"
                    raise InvalidModel, text
        except aloha_fct.WrongFermionFlow, error:
            text = 'Fermion Flow error for interactions %s: %s: %s\n %s' % \
             (', '.join([p.name for p in interaction_info.particles]), 
                                             helas.name, helas.structure, error)
            raise InvalidModel, text
        
        
        
        # Now consider the name only
        lorentz = [helas.name for helas in lorentz] 
        # Import color information:
        colors = [self.treat_color(color_obj, interaction_info, color_info) 
                                    for color_obj in interaction_info.color]
        
        
        order_to_int={}

        for key, couplings in interaction_info.couplings.items():
            if not isinstance(couplings, list):
                couplings = [couplings]
            if interaction_info.lorentz[key[1]].name not in lorentz:
                continue 
            # get the sign for the coupling (if we need to adapt the flow)
            if nb_fermion > 2:
                flow = aloha_fct.get_fermion_flow(interaction_info.lorentz[key[1]].structure, 
                                                                     nb_fermion)
                coupling_sign = self.get_sign_flow(flow, nb_fermion)
            else:                
                coupling_sign = ''            
            for coupling in couplings:
                order = tuple(coupling.order.items())
                if '1' in order:
                    raise InvalidModel, '''Some couplings have \'1\' order. 
                    This is not allowed in MG. 
                    Please defines an additional coupling to your model''' 
                if order in order_to_int:
                    order_to_int[order].get('couplings')[key] = '%s%s' % \
                                               (coupling_sign,coupling.name)
                else:
                    # Initialize a new interaction with a new id tag
                    interaction = base_objects.Interaction({'id':len(self.interactions)+1})                
                    interaction.set('particles', particles)              
                    interaction.set('lorentz', lorentz)
                    interaction.set('couplings', {key: 
                                     '%s%s' %(coupling_sign,coupling.name)})
                    interaction.set('orders', coupling.order)            
                    interaction.set('color', colors)
                    interaction.set('type', type)
                    interaction.set('loop_particles', loop_particles)                    
                    order_to_int[order] = interaction                        
                    # add to the interactions
                    self.interactions.append(interaction)
        
        # check if this interaction conserve the charge defined
 #       if type=='base':
        for charge in list(self.conservecharge): #duplicate to allow modification
            total = 0
            for part in interaction_info.particles:
                try:
                    total += getattr(part, charge)
                except AttributeError:
                    pass
            if abs(total) > 1e-12:
                logger.info('The model has interaction violating the charge: %s' % charge)
                self.conservecharge.discard(charge)
        
        
    def get_sign_flow(self, flow, nb_fermion):
        """ensure that the flow of particles/lorentz are coherent with flow 
           and return a correct version if needed"""
           
        if not flow or nb_fermion < 4:
            return ''
           
        expected = {}
        for i in range(nb_fermion//2):
            expected[i+1] = i+2
        
        if flow == expected:
            return ''

        switch = {}
        for i in range(1, nb_fermion+1):
            if not i in flow:
                continue
            switch[i] = len(switch)
            switch[flow[i]] = len(switch)

        # compute the sign of the permutation
        sign = 1
        done = []
   
        # make a list of consecutive number which correspond to the new
        # order of the particles in the new list.
        new_order = []
        for id in range(nb_fermion): # id is the position in the particles order (starts 0)
            nid = switch[id+1]-1 # nid is the position in the new_particles 
                                 #order (starts 0)
            new_order.append(nid)
             
        # compute the sign:
        sign =1
        for k in range(len(new_order)-1):
            for l in range(k+1,len(new_order)):
                if new_order[l] < new_order[k]:
                    sign *= -1     
                    
        return  '' if sign ==1 else '-'



    
    def add_lorentz(self, name, spins , expr):
        """ Add a Lorentz expression which is not present in the UFO """
        
        new = self.model['lorentz'][0].__class__(name = name,
                spins = spins,
                structure = expr)
        
        self.model['lorentz'].append(new)
        self.model.create_lorentz_dict()
        return name
    
    _pat_T = re.compile(r'T\((?P<first>\d*),(?P<second>\d*)\)')
    _pat_id = re.compile(r'Identity\((?P<first>\d*),(?P<second>\d*)\)')
    
    def treat_color(self, data_string, interaction_info, color_info):
        """ convert the string to ColorString"""
        
        #original = copy.copy(data_string)
        #data_string = p.sub('color.T(\g<first>,\g<second>)', data_string)
        
        
        output = []
        factor = 1
        for term in data_string.split('*'):
            pattern = self._pat_id.search(term)
            if pattern:
                particle = interaction_info.particles[int(pattern.group('first'))-1]
                particle2 = interaction_info.particles[int(pattern.group('second'))-1]
                if particle.color == particle2.color and particle.color in [-6, 6]:
                    error_msg = 'UFO model have inconsistency in the format:\n'
                    error_msg += 'interactions for  particles %s has color information %s\n'
                    error_msg += ' but both fermion are in the same representation %s'
                    raise InvalidModel, error_msg % (', '.join([p.name for p in interaction_info.particles]),data_string, particle.color)
                if particle.color == particle2.color and particle.color in [-3, 3]:
                    if particle.pdg_code in color_info and particle2.pdg_code in color_info:
                      if color_info[particle.pdg_code] == color_info[particle2.pdg_code]:
                        error_msg = 'UFO model have inconsistency in the format:\n'
                        error_msg += 'interactions for  particles %s has color information %s\n'
                        error_msg += ' but both fermion are in the same representation %s'
                        raise InvalidModel, error_msg % (', '.join([p.name for p in interaction_info.particles]),data_string, particle.color)
                    elif particle.pdg_code in color_info:
                        color_info[particle2.pdg_code] = -particle.pdg_code
                    elif particle2.pdg_code in color_info:
                        color_info[particle.pdg_code] = -particle2.pdg_code
                    else:
                        error_msg = 'UFO model have inconsistency in the format:\n'
                        error_msg += 'interactions for  particles %s has color information %s\n'
                        error_msg += ' but both fermion are in the same representation %s'
                        raise InvalidModel, error_msg % (', '.join([p.name for p in interaction_info.particles]),data_string, particle.color)
                
                
                if particle.color == 6:
                    output.append(self._pat_id.sub('color.T6(\g<first>,\g<second>)', term))
                elif particle.color == -6 :
                    output.append(self._pat_id.sub('color.T6(\g<second>,\g<first>)', term))
                elif particle.color == 8:
                    output.append(self._pat_id.sub('color.Tr(\g<first>,\g<second>)', term))
                    factor *= 2
                elif particle.color in [-3,3]:
                    if particle.pdg_code not in color_info:
                        #try to find it one more time 3 -3 1 might help
                        logger.debug('fail to find 3/3bar representation: Retry to find it')
                        color_info = self.find_color_anti_color_rep(color_info)
                        if particle.pdg_code not in color_info:
                            logger.debug('Not able to find the 3/3bar rep from the interactions for particle %s' % particle.name)
                            color_info[particle.pdg_code] = particle.color
                        else:
                            logger.debug('succeed')
                    if particle2.pdg_code not in color_info:
                        #try to find it one more time 3 -3 1 might help
                        logger.debug('fail to find 3/3bar representation: Retry to find it')
                        color_info = self.find_color_anti_color_rep(color_info)
                        if particle2.pdg_code not in color_info:
                            logger.debug('Not able to find the 3/3bar rep from the interactions for particle %s' % particle2.name)
                            color_info[particle2.pdg_code] = particle2.color                    
                        else:
                            logger.debug('succeed')
                
                    if color_info[particle.pdg_code] == 3 :
                        output.append(self._pat_id.sub('color.T(\g<second>,\g<first>)', term))
                    elif color_info[particle.pdg_code] == -3:
                        output.append(self._pat_id.sub('color.T(\g<first>,\g<second>)', term))
                else:
                    raise MadGraph5Error, \
                          "Unknown use of Identity for particle with color %d" \
                          % particle.color
            else:
                output.append(term)
        data_string = '*'.join(output)

        # Change convention for summed indices
        p = re.compile(r'\'\w(?P<number>\d+)\'')
        data_string = p.sub('-\g<number>', data_string)
         
        # Shift indices by -1
        new_indices = {}
        new_indices = dict([(j,i) for (i,j) in \
                           enumerate(range(1,
                                    len(interaction_info.particles)+1))])

                        
        output = data_string.split('*')
        output = color.ColorString([eval(data) \
                                    for data in output if data !='1'])
        output.coeff = fractions.Fraction(factor)
        for col_obj in output:
            col_obj.replace_indices(new_indices)

        return output
      
class OrganizeModelExpression:
    """Organize the couplings/parameters of a model"""
    
    track_dependant = ['aS','aEWM1','MU_R'] # list of variable from which we track 
                                   #dependencies those variables should be define
                                   #as external parameters
    
    # regular expression to shorten the expressions
    complex_number = re.compile(r'''complex\((?P<real>[^,\(\)]+),(?P<imag>[^,\(\)]+)\)''')
    expo_expr = re.compile(r'''(?P<expr>[\w.]+)\s*\*\*\s*(?P<expo>[+-]?[\d.]+)''')
    cmath_expr = re.compile(r'''cmath.(?P<operation>\w+)\((?P<expr>\w+)\)''')
    #operation is usualy sqrt / sin / cos / tan
    conj_expr = re.compile(r'''complexconjugate\((?P<expr>\w+)\)''')
    
    #RE expression for is_event_dependent
    separator = re.compile(r'''[+,\-*/()\s]*''')
    
    
    def __init__(self, model):
    
        self.model = model  # UFOMODEL
        self.perturbation_couplings = {}
        try:
            for order in model.all_orders: # Check if it is a loop model or not
                if(order.perturbative_expansion>0):
                    self.perturbation_couplings[order.name]=order.perturbative_expansion
        except AttributeError:
            pass
        self.params = {}     # depend on -> ModelVariable
        self.couplings = {}  # depend on -> ModelVariable
        self.all_expr = {} # variable_name -> ModelVariable
    
    def main(self, additional_couplings = []):
        """Launch the actual computation and return the associate 
        params/couplings. Possibly consider additional_couplings in addition
        to those defined in the UFO model attribute all_couplings """

        additional_params = []
        if hasattr(self.model,'all_CTparameters'):
            additional_params = self.get_additional_CTparameters()

        self.analyze_parameters(additional_params = additional_params)
        self.analyze_couplings(additional_couplings = additional_couplings)
        
        # Finally revert the possible modifications done by treat_couplings()
        if hasattr(self.model,'all_CTparameters'):
            self.revert_CTCoupling_modifications()

        return self.params, self.couplings

    def revert_CTCoupling_modifications(self):
        """ Finally revert the possible modifications done by treat_couplings()
        in UFOMG5Converter which were useful for the add_CTinteraction() in 
        particular. This modification consisted in expanding the value of a
        CTCoupling which consisted in an expression in terms of a CTParam to 
        its corresponding dictionary (e.g 
              CTCoupling.value = 2*CTParam           ->
              CTCoupling.value = {-1: 2*CTParam_1EPS_, 0: 2*CTParam_FIN_}
        for example if CTParam had a non-zero finite and single pole."""
        
        for coupl in self.model.all_couplings:
            if hasattr(coupl,'old_value'):
                coupl.value = coupl.old_value
                del(coupl.old_value)

    def get_additional_CTparameters(self):
        """ For each CTparameter split it into spimple parameter for each pole
        and the finite part if not zero."""

        additional_params = []
        for CTparam in self.model.all_CTparameters:
            for pole in range(3):
                if CTparam.pole(pole) != 'ZERO':
                  CTparam_piece = copy.copy(CTparam)
                  CTparam_piece.name = '%s_%s_'%(CTparam.name,pole_dict[-pole])
                  CTparam_piece.nature = 'internal'
                  CTparam_piece.type = CTparam.type
                  CTparam_piece.value = CTparam.pole(pole)
                  CTparam_piece.texname = '%s_{%s}'%\
                                              (CTparam.texname,pole_dict[-pole])
                  additional_params.append(CTparam_piece)
        return additional_params

    def analyze_parameters(self, additional_params=[]):
        """ separate the parameters needed to be recomputed events by events and
        the others"""
        # in order to match in Gmu scheme
        # test whether aEWM1 is the external or not
        # if not, take Gf as the track_dependant variable
        present_aEWM1 = any(param.name == 'aEWM1' for param in
                        self.model.all_parameters if param.nature == 'external')

        if not present_aEWM1:
            self.track_dependant = ['aS','Gf','MU_R']

        for param in self.model.all_parameters+additional_params:
            if param.nature == 'external':
                parameter = base_objects.ParamCardVariable(param.name, param.value, \
                                               param.lhablock, param.lhacode)
                
            else:
                expr = self.shorten_expr(param.value)
                depend_on = self.find_dependencies(expr)
                parameter = base_objects.ModelVariable(param.name, expr, param.type, depend_on)
            
            self.add_parameter(parameter)     
            
    def add_parameter(self, parameter):
        """ add consistently the parameter in params and all_expr.
        avoid duplication """
        
        assert isinstance(parameter, base_objects.ModelVariable)
        
        if parameter.name in self.all_expr:
            return
        
        self.all_expr[parameter.name] = parameter
        try:
            self.params[parameter.depend].append(parameter)
        except:
            self.params[parameter.depend] = [parameter]
            
    def add_coupling(self, coupling):
        """ add consistently the coupling in couplings and all_expr.
        avoid duplication """
        
        assert isinstance(coupling, base_objects.ModelVariable)
        
        if coupling.name in self.all_expr:
            return
        self.all_expr[coupling.value] = coupling
        try:
            self.coupling[coupling.depend].append(coupling)
        except:
            self.coupling[coupling.depend] = [coupling]

    def analyze_couplings(self,additional_couplings=[]):
        """creates the shortcut for all special function/parameter
        separate the couplings dependent of track variables of the others"""
        
        # For loop models, make sure that all couplings with dictionary values
        # are turned into set of couplings, one for each pole and finite part.
        if self.perturbation_couplings:
            couplings_list=[]
            for coupling in self.model.all_couplings + additional_couplings:
                if not isinstance(coupling.value,dict):
                    couplings_list.append(coupling)
                else:
                    for poleOrder in range(0,3):
                        if coupling.pole(poleOrder)!='ZERO':                    
                            newCoupling=copy.copy(coupling)
                            if poleOrder!=0:
                                newCoupling.name += "_%deps"%poleOrder
                            newCoupling.value=coupling.pole(poleOrder)
                            # assign the CT parameter dependences
#                             if hasattr(coupling,'CTparam_dependence') and \
#                                     (-poleOrder in coupling.CTparam_dependence) and \
#                                     coupling.CTparam_dependence[-poleOrder]:
#                                 newCoupling.CTparam_dependence = coupling.CTparam_dependence[-poleOrder]
#                             elif hasattr(newCoupling,'CTparam_dependence'):
#                                 delattr(newCoupling,"CTparam_dependence")
                            couplings_list.append(newCoupling)
        else:
            couplings_list = self.model.all_couplings + additional_couplings
            couplings_list = [c for c in couplings_list if not isinstance(c.value, dict)] 
            
        for coupling in couplings_list:
            # shorten expression, find dependencies, create short object
            expr = self.shorten_expr(coupling.value)
            depend_on = self.find_dependencies(expr)
            parameter = base_objects.ModelVariable(coupling.name, expr, 'complex', depend_on)
            # Add consistently in the couplings/all_expr
            try:
                self.couplings[depend_on].append(parameter)
            except KeyError:
                self.couplings[depend_on] = [parameter]
            self.all_expr[coupling.value] = parameter                

    def find_dependencies(self, expr):
        """check if an expression should be evaluated points by points or not
        """
        depend_on = set()

        # Treat predefined result
        #if name in self.track_dependant:  
        #    return tuple()
        
        # Split the different part of the expression in order to say if a 
        #subexpression is dependent of one of tracked variable
        expr = self.separator.split(expr)
        
        # look for each subexpression
        for subexpr in expr:
            if subexpr in self.track_dependant:
                depend_on.add(subexpr)
                
            elif subexpr in self.all_expr and self.all_expr[subexpr].depend:
                [depend_on.add(value) for value in self.all_expr[subexpr].depend 
                                if  self.all_expr[subexpr].depend != ('external',)]
        if depend_on:
            return tuple(depend_on)
        else:
            return tuple()


    def shorten_expr(self, expr):
        """ apply the rules of contraction and fullfill
        self.params with dependent part"""
        try:
            expr = self.complex_number.sub(self.shorten_complex, expr)
            expr = self.expo_expr.sub(self.shorten_expo, expr)
            expr = self.cmath_expr.sub(self.shorten_cmath, expr)
            expr = self.conj_expr.sub(self.shorten_conjugate, expr)
        except Exception:
            logger.critical("fail to handle expression: %s, type()=%s", expr,type(expr))
            raise
        return expr
    

    def shorten_complex(self, matchobj):
        """add the short expression, and return the nice string associate"""
        
        float_real = float(eval(matchobj.group('real')))
        float_imag = float(eval(matchobj.group('imag')))
        if float_real == 0 and float_imag ==1:
            new_param = base_objects.ModelVariable('complexi', 'complex(0,1)', 'complex')
            self.add_parameter(new_param)
            return 'complexi'
        else:
            return 'complex(%s, %s)' % (matchobj.group('real'), matchobj.group('imag'))
        
        
    def shorten_expo(self, matchobj):
        """add the short expression, and return the nice string associate"""
        
        expr = matchobj.group('expr')
        exponent = matchobj.group('expo')
        new_exponent = exponent.replace('.','_').replace('+','').replace('-','_m_')
        output = '%s__exp__%s' % (expr, new_exponent)
        old_expr = '%s**%s' % (expr,exponent)

        if expr.startswith('cmath'):
            return old_expr
        
        if expr.isdigit():
            output = 'nb__' + output #prevent to start with a number
            new_param = base_objects.ModelVariable(output, old_expr,'real')
        else:
            depend_on = self.find_dependencies(expr)
            type = self.search_type(expr)
            new_param = base_objects.ModelVariable(output, old_expr, type, depend_on)
        self.add_parameter(new_param)
        return output
        
    def shorten_cmath(self, matchobj):
        """add the short expression, and return the nice string associate"""
        
        expr = matchobj.group('expr')
        operation = matchobj.group('operation')
        output = '%s__%s' % (operation, expr)
        old_expr = ' cmath.%s(%s) ' %  (operation, expr)
        if expr.isdigit():
            new_param = base_objects.ModelVariable(output, old_expr , 'real')
        else:
            depend_on = self.find_dependencies(expr)
            type = self.search_type(expr)
            new_param = base_objects.ModelVariable(output, old_expr, type, depend_on)
        self.add_parameter(new_param)
        
        return output        
        
    def shorten_conjugate(self, matchobj):
        """add the short expression, and retrun the nice string associate"""
        
        expr = matchobj.group('expr')
        output = 'conjg__%s' % (expr)
        old_expr = ' complexconjugate(%s) ' % expr
        depend_on = self.find_dependencies(expr)
        type = 'complex'
        new_param = base_objects.ModelVariable(output, old_expr, type, depend_on)
        self.add_parameter(new_param)  
                    
        return output            
    

     
    def search_type(self, expr, dep=''):
        """return the type associate to the expression if define"""
        
        try:
            return self.all_expr[expr].type
        except:
            return 'complex'
            
class RestrictModel(model_reader.ModelReader):
    """ A class for restricting a model for a given param_card.
    rules applied:
     - Vertex with zero couplings are throw away
     - external parameter with zero/one input are changed into internal parameter.
     - identical coupling/mass/width are replace in the model by a unique one
     """
  
    def default_setup(self):
        """define default value"""
        self.del_coup = []
        super(RestrictModel, self).default_setup()
        self.rule_card = check_param_card.ParamCardRule()
        self.restrict_card = None
     
    def restrict_model(self, param_card, rm_parameter=True, keep_external=False,
                                                      complex_mass_scheme=None):
        """apply the model restriction following param_card.
        rm_parameter defines if the Zero/one parameter are removed or not from
        the model.
        keep_external if the param_card need to be kept intact
        """
        
        if self.get('name') == "mssm" and not keep_external:
            raise Exception
        self.restrict_card = param_card
        # Reset particle dict to ensure synchronized particles and interactions
        self.set('particles', self.get('particles'))

        # compute the value of all parameters
        # Get the list of definition of model functions, parameter values. 
        self.set_parameters_and_couplings(param_card, 
                                        complex_mass_scheme=complex_mass_scheme)
        
        
        # Keep the list of definition of model functions, parameter values. 
        model_definitions = self.set_parameters_and_couplings(param_card)
        
        # Simplify conditional statements
        logger.debug('Simplifying conditional expressions')
        modified_params, modified_couplings = \
            self.detect_conditional_statements_simplifications(model_definitions)
        
        # Apply simplifications
        self.apply_conditional_simplifications(modified_params, modified_couplings)
        
        # associate to each couplings the associated vertex: def self.coupling_pos
        self.locate_coupling()
        # deal with couplings
        zero_couplings, iden_couplings = self.detect_identical_couplings()

        # remove the out-dated interactions
        self.remove_interactions(zero_couplings)
        
        # replace in interactions identical couplings
        for iden_coups in iden_couplings:
            self.merge_iden_couplings(iden_coups)

        # remove zero couplings and other pointless couplings
        self.del_coup += zero_couplings
        self.remove_couplings(self.del_coup)
       
        # deal with parameters
        parameters = self.detect_special_parameters()
        self.fix_parameter_values(*parameters, simplify=rm_parameter, 
                                                    keep_external=keep_external)

        # deal with identical parameters
        if not keep_external:
            iden_parameters = self.detect_identical_parameters()
            for iden_param in iden_parameters:
                self.merge_iden_parameters(iden_param)
    
        iden_parameters = self.detect_identical_parameters()
        for iden_param in iden_parameters:
            self.merge_iden_parameters(iden_param, keep_external)
              
        # change value of default parameter if they have special value:
        # 9.999999e-1 -> 1.0
        # 0.000001e-99 -> 0 Those value are used to avoid restriction
        for name, value in self['parameter_dict'].items():
            if value == 9.999999e-1:
                self['parameter_dict'][name] = 1
            elif value == 0.000001e-99:
                self['parameter_dict'][name] = 0

                    
    def locate_coupling(self):
        """ create a dict couplings_name -> vertex or (particle, counterterm_key) """
        
        self.coupling_pos = {}
        for vertex in self['interactions']:
            for key, coupling in vertex['couplings'].items():
                if coupling in self.coupling_pos:
                    if vertex not in self.coupling_pos[coupling]:
                        self.coupling_pos[coupling].append(vertex)
                else:
                    self.coupling_pos[coupling] = [vertex]
        
        for particle in self['particles']:
            for key, coupling_dict in particle['counterterm'].items():
                for LaurentOrder, coupling in coupling_dict.items():
                    if coupling in self.coupling_pos:
                        if (particle,key) not in self.coupling_pos[coupling]:
                            self.coupling_pos[coupling].append((particle,key))
                    else:
                        self.coupling_pos[coupling] = [(particle,key)]

        return self.coupling_pos
        
    def detect_identical_couplings(self, strict_zero=False):
        """return a list with the name of all vanishing couplings"""
        
        dict_value_coupling = {}
        iden_key = set()
        zero_coupling = []
        iden_coupling = []
        
        for name, value in self['coupling_dict'].items():
            if value == 0:
                zero_coupling.append(name)
                continue
            elif not strict_zero and abs(value) < 1e-13:
                logger.debug('coupling with small value %s: %s treated as zero' %
                             (name, value))
                zero_coupling.append(name)
            elif not strict_zero and abs(value) < 1e-10:
                return self.detect_identical_couplings(strict_zero=True)

            
            if value in dict_value_coupling:
                iden_key.add(value)
                dict_value_coupling[value].append(name)
            else:
                dict_value_coupling[value] = [name]
        
        for key in iden_key:
            iden_coupling.append(dict_value_coupling[key])

        return zero_coupling, iden_coupling
    
    
    def detect_special_parameters(self):
        """ return the list of (name of) parameter which are zero """
        
        null_parameters = []
        one_parameters = []
        for name, value in self['parameter_dict'].items():
            if value == 0 and name != 'ZERO':
                null_parameters.append(name)
            elif value == 1:
                one_parameters.append(name)
        
        return null_parameters, one_parameters
    
    def apply_conditional_simplifications(self, modified_params,
                                                            modified_couplings):
        """ Apply the conditional statement simplifications for parameters and
        couplings detected by 'simplify_conditional_statements'.
        modified_params (modified_couplings) are list of tuples (a,b) with a
        parameter (resp. coupling) instance and b is the simplified expression."""
        
        if modified_params:
            logger.debug("Conditional expressions are simplified for parameters:")
            logger.debug(",".join("%s"%param[0].name for param in modified_params))
        for param, new_expr in modified_params:
            param.expr = new_expr
        
        if modified_couplings:
            logger.debug("Conditional expressions are simplified for couplings:")
            logger.debug(",".join("%s"%coupl[0].name for coupl in modified_couplings))
        for coupl, new_expr in modified_couplings:
            coupl.expr = new_expr
    
    def detect_conditional_statements_simplifications(self, model_definitions,
          objects=['couplings','parameters']):
        """ Simplifies the 'if' statements in the pythonic UFO expressions
        of parameters using the default variables specified in the restrict card.
        It returns a list of objects (parameters or couplings) and the new
        expression that they should take. Model definitions include all definitons
        of the model functions and parameters."""
        
        param_modifications = []
        coupl_modifications = []
        ifparser = parsers.UFOExpressionParserPythonIF(model_definitions)
        
        start_param = time.time()
        if 'parameters' in objects:
            for dependences, param_list in self['parameters'].items():
                if 'external' in dependences:
                    continue
                for param in param_list:
                    new_expr, n_changes = ifparser.parse(param.expr)
                    if n_changes > 0:
                        param_modifications.append((param, new_expr))
      
        end_param = time.time()
  
        if 'couplings' in objects:         
            for dependences, coupl_list in self['couplings'].items():
                for coupl in coupl_list:
                    new_expr, n_changes = ifparser.parse(coupl.expr)
                    if n_changes > 0:
                        coupl_modifications.append((coupl, new_expr))        

        end_coupl = time.time()
        
        tot_param_time = end_param-start_param
        tot_coupl_time = end_coupl-end_param
        if tot_param_time>5.0:
            logger.debug("Simplification of conditional statements"+\
              " in parameter expressions done in %s."%misc.format_time(tot_param_time))
        if tot_coupl_time>5.0:
            logger.debug("Simplification of conditional statements"+\
              " in couplings expressions done in %s."%misc.format_time(tot_coupl_time))

        return param_modifications, coupl_modifications
    
    def detect_identical_parameters(self):
        """ return the list of tuple of name of parameter with the same 
        input value """

        # Extract external parameters
        external_parameters = self['parameters'][('external',)]
        
        # define usefull variable to detect identical input
        block_value_to_var={} #(lhablok, value): list_of_var
        mult_param = set([])       # key of the previous dict with more than one
                              #parameter.
                              
        #detect identical parameter and remove the duplicate parameter
        for param in external_parameters[:]:
            value = self['parameter_dict'][param.name]
            if value in [0,1,0.000001e-99,9.999999e-1]:
                continue
            if param.lhablock.lower() == 'decay':
                continue
            
            key = (param.lhablock, value)
            mkey =  (param.lhablock, -value)
            if key in block_value_to_var:
                block_value_to_var[key].append((param,1))
                mult_param.add(key)
            elif mkey in block_value_to_var:
                block_value_to_var[mkey].append((param,-1))
                mult_param.add(mkey)
            else: 
                block_value_to_var[key] = [(param,1)]        
        
        output=[]  
        for key in mult_param:
            output.append(block_value_to_var[key])
            
        return output


    def merge_iden_couplings(self, couplings):
        """merge the identical couplings in the interactions and particle 
        counterterms"""

        
        logger_mod.debug(' Fuse the Following coupling (they have the same value): %s '% \
                        ', '.join([obj for obj in couplings]))
        
        main = couplings[0]
        self.del_coup += couplings[1:] # add the other coupl to the suppress list
        
        for coupling in couplings[1:]:
            # check if param is linked to an interaction
            if coupling not in self.coupling_pos:
                continue
            # replace the coupling, by checking all coupling of the interaction
            vertices = [ vert for vert in self.coupling_pos[coupling] if 
                         isinstance(vert, base_objects.Interaction)]
            for vertex in vertices:
                for key, value in vertex['couplings'].items():
                    if value == coupling:
                        vertex['couplings'][key] = main

            # replace the coupling appearing in the particle counterterm
            particles_ct = [ pct for pct in self.coupling_pos[coupling] if 
                         isinstance(pct, tuple)]
            for pct in particles_ct:
                for key, value in pct[0]['counterterm'][pct[1]].items():
                    if value == coupling:
                        pct[0]['counterterm'][pct[1]][key] = main

         
    def merge_iden_parameters(self, parameters, keep_external=False):
        """ merge the identical parameters given in argument.
        keep external force to keep the param_card untouched (up to comment)"""
            
        logger_mod.debug('Parameters set to identical values: %s '% \
                 ', '.join(['%s*%s' % (f, obj.name.replace('mdl_','')) for (obj,f) in parameters]))

        # Extract external parameters
        external_parameters = self['parameters'][('external',)]
        for i, (obj, factor) in enumerate(parameters):
            # Keeped intact the first one and store information
            if i == 0:
                obj.info = 'set of param :' + \
                                     ', '.join([str(f)+'*'+param.name.replace('mdl_','')
                                                 for (param, f) in parameters])
                expr = obj.name
                continue
            # Add a Rule linked to the param_card
            if factor ==1:
                self.rule_card.add_identical(obj.lhablock.lower(), obj.lhacode, 
                                                         parameters[0][0].lhacode )
            else:
                self.rule_card.add_opposite(obj.lhablock.lower(), obj.lhacode, 
                                                         parameters[0][0].lhacode )
            obj_name = obj.name
            # delete the old parameters
            if not keep_external:                
                external_parameters.remove(obj)
            elif obj.lhablock.upper() in ['MASS','DECAY']:
                external_parameters.remove(obj)
            else:
                obj.name = ''
                obj.info = 'MG5 will not use this value use instead %s*%s' %(factor,expr)    
            # replace by the new one pointing of the first obj of the class
            new_param = base_objects.ModelVariable(obj_name, '%s*%s' %(factor, expr), 'real')
            self['parameters'][()].insert(0, new_param)
        
        # For Mass-Width, we need also to replace the mass-width in the particles
        #This allows some optimization for multi-process.
        if parameters[0][0].lhablock in ['MASS','DECAY']:
            new_name = parameters[0][0].name
            if parameters[0][0].lhablock == 'MASS':
                arg = 'mass'
            else:
                arg = 'width'
            change_name = [p.name for (p,f) in parameters[1:]]
            [p.set(arg, new_name) for p in self['particle_dict'].values() 
                                                       if p[arg] in change_name]
            
    def remove_interactions(self, zero_couplings):
        """ remove the interactions and particle counterterms 
        associated to couplings"""
        
        
        mod_vertex = []
        mod_particle_ct = []
        for coup in zero_couplings:
            # some coupling might be not related to any interactions
            if coup not in self.coupling_pos:
                continue
            
            # Remove the corresponding interactions.

            vertices = [ vert for vert in self.coupling_pos[coup] if 
                         isinstance(vert, base_objects.Interaction) ]
            for vertex in vertices:
                modify = False
                for key, coupling in vertex['couplings'].items():
                    if coupling in zero_couplings:
                        modify=True
                        del vertex['couplings'][key]
                    elif coupling.startswith('-'):
                        coupling = coupling[1:]
                        if coupling in zero_couplings:
                            modify=True
                            del vertex['couplings'][key]                      
                        
                if modify:
                    mod_vertex.append(vertex)
            
            # Remove the corresponding particle counterterm
            particles_ct = [ pct for pct in self.coupling_pos[coup] if 
                         isinstance(pct, tuple)]
            for pct in particles_ct:
                modify = False
                for key, coupling in pct[0]['counterterm'][pct[1]].items():
                    if coupling in zero_couplings:
                        modify=True
                        del pct[0]['counterterm'][pct[1]][key]
                if modify:
                    mod_particle_ct.append(pct)

        # print useful log and clean the empty interaction
        for vertex in mod_vertex:
            part_name = [part['name'] for part in vertex['particles']]
            orders = ['%s=%s' % (order,value) for order,value in vertex['orders'].items()]
                                        
            if not vertex['couplings']:
                logger_mod.debug('remove interactions: %s at order: %s' % \
                                        (' '.join(part_name),', '.join(orders)))
                self['interactions'].remove(vertex)
            else:
                logger_mod.debug('modify interactions: %s at order: %s' % \
                                (' '.join(part_name),', '.join(orders)))

        # print useful log and clean the empty counterterm values
        for pct in mod_particle_ct:
            part_name = pct[0]['name']
            order = pct[1][0]
            loop_parts = ','.join(['('+','.join([\
                         self.get_particle(p)['name'] for p in part])+')' \
                         for part in pct[1][1]])
                                        
            if not pct[0]['counterterm'][pct[1]]:
                logger_mod.debug('remove counterterm of particle %s'%part_name+\
                                 ' with loop particles (%s)'%loop_parts+\
                                 ' perturbing order %s'%order)
                del pct[0]['counterterm'][pct[1]]
            else:
                logger_mod.debug('Modify counterterm of particle %s'%part_name+\
                                 ' with loop particles (%s)'%loop_parts+\
                                 ' perturbing order %s'%order)  

        return
                
    def remove_couplings(self, couplings):               
        #clean the coupling list:
        for name, data in self['couplings'].items():
            for coupling in data[:]:
                if coupling.name in couplings:
                    data.remove(coupling)
                            
        
    def fix_parameter_values(self, zero_parameters, one_parameters, 
                                            simplify=True, keep_external=False):
        """ Remove all instance of the parameters in the model and replace it by 
        zero when needed."""


        # treat specific cases for masses and width
        for particle in self['particles']:
            if particle['mass'] in zero_parameters:
                particle['mass'] = 'ZERO'
            if particle['width'] in zero_parameters:
                particle['width'] = 'ZERO'
            if particle['width'] in one_parameters:
                one_parameters.remove(particle['width'])                
                
        for pdg, particle in self['particle_dict'].items():
            if particle['mass'] in zero_parameters:
                particle['mass'] = 'ZERO'
            if particle['width'] in zero_parameters:
                particle['width'] = 'ZERO'


        # Add a rule for zero/one parameter
        external_parameters = self['parameters'][('external',)]
        for param in external_parameters[:]:
            value = self['parameter_dict'][param.name]
            block = param.lhablock.lower()
            if value == 0:
                self.rule_card.add_zero(block, param.lhacode)
            elif value == 1:
                self.rule_card.add_one(block, param.lhacode)

        special_parameters = zero_parameters + one_parameters
        
            

        if simplify:
            # check if the parameters is still useful:
            re_str = '|'.join(special_parameters)
            if len(re_str) > 25000: # size limit on mac
                split = len(special_parameters) // 2
                re_str = ['|'.join(special_parameters[:split]),
                          '|'.join(special_parameters[split:])]
            else:
                re_str = [ re_str ]
            used = set()
            for expr in re_str:
                re_pat = re.compile(r'''\b(%s)\b''' % expr)
                # check in coupling
                for name, coupling_list in self['couplings'].items():
                    for coupling in coupling_list:
                        for use in  re_pat.findall(coupling.expr):
                            used.add(use)
        else:
            used = set([i for i in special_parameters if i])
        
        # simplify the regular expression
        re_str = '|'.join([param for param in special_parameters if param not in used])
        if len(re_str) > 25000: # size limit on mac
            split = len(special_parameters) // 2
            re_str = ['|'.join(special_parameters[:split]),
                          '|'.join(special_parameters[split:])]
        else:
            re_str = [ re_str ]
        for expr in re_str:                                                      
            re_pat = re.compile(r'''\b(%s)\b''' % expr)
               
            param_info = {}
            # check in parameters
            for dep, param_list in self['parameters'].items():
                for tag, parameter in enumerate(param_list):
                    # update information concerning zero/one parameters
                    if parameter.name in special_parameters:
                        param_info[parameter.name]= {'dep': dep, 'tag': tag, 
                                                               'obj': parameter}
                        continue
                                        
                    # Bypass all external parameter
                    if isinstance(parameter, base_objects.ParamCardVariable):
                        continue
    
                    if simplify:
                        for use in  re_pat.findall(parameter.expr):
                            used.add(use)
                        
        # modify the object for those which are still used
        for param in used:
            if not param:
                continue
            data = self['parameters'][param_info[param]['dep']]
            data.remove(param_info[param]['obj'])
            tag = param_info[param]['tag']
            data = self['parameters'][()]
            if param in zero_parameters:
                data.insert(0, base_objects.ModelVariable(param, '0.0', 'real'))
            else:
                data.insert(0, base_objects.ModelVariable(param, '1.0', 'real'))
                
        # remove completely useless parameters
        for param in special_parameters:
            #by pass parameter still in use
            if param in used or \
                  (keep_external and param_info[param]['dep'] == ('external',)):
                logger_mod.debug('fix parameter value: %s' % param)
                continue 
            logger_mod.debug('remove parameters: %s' % (param))
            data = self['parameters'][param_info[param]['dep']]
            data.remove(param_info[param]['obj'])

                



                
        
        
        
         
      
    
      
        
        
    
