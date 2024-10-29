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

"""Definitions of the objects needed both for MadFKS from real 
and MadFKS from born"""

from __future__ import absolute_import
from __future__ import print_function
import logging
import copy
from itertools import chain, combinations
import madgraph.core.base_objects as MG
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.color_amp as color_amp
import madgraph.core.color_algebra as color_algebra


logger = logging.getLogger('madgraph.sudakov')
    
    
    
class SudakovError(Exception):
    """Exception for the Sudakov module"""
    pass


def get_isospin_partners_diffcharge(pid, model):
    """returns a list of isospin partners of particle pid, with electric charge
    shifted by +/-1
    """

    # iso dict should eventually become an attribute of the model, here we code
    # the case of the SM with goldsones
    iso_dict = {1: [2], 2: [1], 3: [4], 4: [3], 5: [6], 6: [5],
                -1: [-2], -2: [-1], -3: [-4], -4: [-3], -5: [-6], -6: [-5],
                11: [12], 12: [11], 13: [14], 14: [13], 15: [16], 16: [15],
                -11: [-12], -12: [-11], -13: [-14], -14: [-13], -15: [-16], -16: [-15],
                22: [24, -24], 23: [24, -24], 24: [22, 23], -24: [22, 23], 
                21: [],
                25: [251, -251], 250: [251, -251], 251: [25, 250], -251: [25, 250]} 

    partners = iso_dict[pid]

    # check that the model includes all particles
    if not all([ppid in model.get('particle_dict').keys() for ppid in partners + [pid]]):
        raise SudakovError('Invalid ids %s for model %s ' % \
                            (str(partners + [pid]), model.get('name')))

    return partners



def get_isospin_partners_samecharge_cew(pid, model):
    """returns a list of isospin partners of particle pid, with the same charge.
    These enter the LSC terms, proportional to C_EW (Z<->gamma mixing)
    """

    # iso dict should eventually become an attribute of the model, here we code
    # the case of the SM with goldsones
    iso_dict = {22: [23], 23: [22]} 
    ##iso_dict = {22: [23], 23: [22]} 

    try:
        return iso_dict[pid]
    except KeyError:
        return []


def get_isospin_partners_samecharge_iz(pid, model):
    """returns a list of isospin partners of particle pid, with the same charge
    These enter the SSC terms, proportional to I_Z (Chi<->Higgs)
    """

    # iso dict should eventually become an attribute of the model, here we code
    # the case of the SM with goldsones
    iso_dict = {25: [250], 250: [25]} 
    ##iso_dict = {22: [23], 23: [22]} 

    try:
        return iso_dict[pid]
    except KeyError:
        return []


def get_goldstone(pid, model):
    """returns the goldstone boson correspoinding to the longitudinal polarisation 
    of particlle pid
    """
    gold_dict = {23: 250, 24: 251, -24: -251}

    try:
        return gold_dict[pid]
    except KeyError:
        return None


def is_charge_conserved(proc):
    """ verifies if charge conservation is satisfied for the input
    process
    """
    model = proc['model']
    totcharge = 0.
    for leg in proc['legs']:
        part = model.get('particle_dict')[leg.get('id')]
        charge = part.get('charge')
        if (leg.get('id') != part['pdg_code']) != leg['state']:
            totcharge -= charge
        else:
            totcharge += charge

    return  abs(totcharge) < 1e-10


def get_sudakov_amps(born_amp):
    """returns all the amplitudes needed to compute EW 
    corrections to born_amp in the sudakov approximation
    """

    amplitudes = []
    goldstone_amplitudes = []
    model = born_amp['process']['model']

    # first, find all the legs that have a corresponding Goldstone
    goldstone_legs = []
    for leg in born_amp['process']['legs']:
        goldstone = get_goldstone(leg['id'], model) 
        # skip if no goldstone exist
        if goldstone is None: continue
        goldstone_legs.append(leg)

    # 0) find all amplitudes that can be obtained from the born, replacing any 
    # possible combination of goldstone_legs with goldstones
    # (labeled by 'Goldstone : True')
    # the "reversed" function is needed for the output
    for goldstone_comb in \
       chain.from_iterable(combinations(goldstone_legs, r) for r in reversed(range(1, len(goldstone_legs)+1))):

        ####born_proc = copy.deepcopy(born_amp['process'])
        # MZ: NEVER deepcopy a process!!!
        born_proc = copy.copy(born_amp['process'])
        # copy the legs as a LegList (not FKSLegList) in order 
        # not to have them re-ordered
        born_proc['legs'] = MG.LegList(copy.deepcopy(born_amp['process']['legs']))
        pdgs = [[],[]] # old and new pdgs
        # replace all legs listed in goldstone_comb
        for leg in goldstone_comb:

            newleg = copy.copy(leg)
            newleg['id'] = get_goldstone(newleg['id'], model)
            born_proc['legs'][born_proc['legs'].index(leg)] = newleg
            pdgs[0].append(leg['id'])
            pdgs[1].append(newleg['id'])
        # now generate the amplitude
        amp = diagram_generation.Amplitude(born_proc)
        # skip amplitudes without diagrams
        if not amp['diagrams'] : continue

        logger.info('   Found Sudakov amplitude (goldstone) for %s' % born_proc.nice_string())
        amplitudes.append({'type': 'goldstone', 'legs': goldstone_comb, 'base_amp': 0, 'amplitude': amp, 'pdgs': pdgs})
        # for these amplitudes, keep track in a separate list
        goldstone_amplitudes.append(amp)

    # 1) single loop over the born legs to find amplitudes needed for the 
    # LSC part, proportional to C_EW
    # (in the SM, this is relevant only for a <-> z)
    # Note that one has to loop on the base amplitude and on those with the
    # goldstones
    for iamp, base_amp in enumerate([born_amp] + goldstone_amplitudes):
        logger.info("Generating Sudakov amplitudes (LSC) based on " + base_amp['process'].nice_string())
        for ileg, leg in enumerate(base_amp['process']['legs']):
            iso_part_list = get_isospin_partners_samecharge_cew(leg['id'], model)

            # skip if no partners exist
            if not iso_part_list: continue

            for part in iso_part_list:
                born_proc = copy.copy(base_amp['process'])
                # copy the legs as a LegList (not FKSLegList) in order 
                # not to have them re-ordered
                born_proc['legs'] = MG.LegList(copy.deepcopy(base_amp['process']['legs']))
                newleg = copy.copy(leg)
                newleg['id'] = part
                born_proc['legs'][ileg] = newleg
                pdgs = [[leg['id']],[newleg['id']]] # old and new pdgs
                amp = diagram_generation.Amplitude(born_proc)
                # skip amplitudes without diagrams
                if not amp['diagrams'] : continue

                logger.info('   Found Sudakov amplitude (isospin same-charge, C_EW) for %s' % born_proc.nice_string())
                amplitudes.append({'type': 'cew', 'legs': [leg], 'base_amp': iamp, 'amplitude': amp, 'pdgs': pdgs})

    # 2) single loop over the born legs to find amplitudes needed for the 
    # SSC part, proportional to I_Z
    # (in the SM, this is relevant only for chi <-> h)
    # Note that one has to loop on the base amplitude and on those with the
    # goldstones
    for iamp, base_amp in enumerate([born_amp] + goldstone_amplitudes):
        logger.info("Generating Sudakov amplitudes (SSC-n1) based on " + base_amp['process'].nice_string())
        for ileg, leg in enumerate(base_amp['process']['legs']):
            iso_part_list = get_isospin_partners_samecharge_iz(leg['id'], model)

            # skip if no partners exist
            if not iso_part_list: continue

            for part in iso_part_list:
                born_proc = copy.copy(base_amp['process'])
                # copy the legs as a LegList (not FKSLegList) in order 
                # not to have them re-ordered
                born_proc['legs'] = MG.LegList(copy.deepcopy(base_amp['process']['legs']))
                newleg = copy.copy(leg)
                newleg['id'] = part
                born_proc['legs'][ileg] = newleg
                pdgs = [[leg['id']],[newleg['id']]] # old and new pdgs
                amp = diagram_generation.Amplitude(born_proc)
                # skip amplitudes without diagrams
                if not amp['diagrams'] : continue

                logger.info('   Found Sudakov amplitude (isospin same-charge, I_Z) for %s' % born_proc.nice_string())
                amplitudes.append({'type': 'iz1', 'legs': [leg], 'base_amp': iamp, 'amplitude': amp, 'pdgs': pdgs})

    # 3) double loop over the born legs to find amplitudes needed for the SSC 
    # part, proportional to I_z, where
    # two particles are switched for their isospin partner(s), in the case
    # the charge is unchanged
    # Note that one has to loop on the base amplitude and on those with the
    # goldstones
    for iamp, base_amp in enumerate([born_amp] + goldstone_amplitudes):
        logger.info("Generating Sudakov amplitudes (SSC-n2) based on " + base_amp['process'].nice_string())
        for ileg1, leg1 in enumerate(base_amp['process']['legs']):
            iso_part_list1 = get_isospin_partners_samecharge_iz(leg1['id'], model)

            # skip if no partners exist
            if not iso_part_list1: continue

            for ileg2, leg2 in enumerate(base_amp['process']['legs']):
                if ileg1 >= ileg2: continue

                iso_part_list2 = get_isospin_partners_samecharge_iz(leg2['id'], model)

                # skip if no partners exist
                if not iso_part_list2: continue

                for part1 in iso_part_list1:
                    for part2 in iso_part_list2:
                        born_proc = copy.copy(base_amp['process'])
                        # copy the legs as a LegList (not FKSLegList) in order 
                        # not to have them re-ordered
                        born_proc['legs'] = MG.LegList(copy.deepcopy(base_amp['process']['legs']))
                        # replace leg1
                        newleg1 = copy.copy(leg1)
                        newleg1['id'] = part1
                        born_proc['legs'][ileg1] = newleg1
                        # replace leg2
                        newleg2 = copy.copy(leg2)
                        newleg2['id'] = part2
                        born_proc['legs'][ileg2] = newleg2
                        pdgs = [[leg1['id'],leg2['id']], [newleg1['id'],newleg2['id']]] # old and new pdgs

                        amp = diagram_generation.Amplitude(born_proc)
                        # skip amplitudes without diagrams
                        if not amp['diagrams'] : continue

                        logger.info('   Found Sudakov amplitude (isospin same-charge, I_Z x I_Z) for %s' % born_proc.nice_string())
                        amplitudes.append({'type': 'iz2', 'legs': [leg1, leg2], 'base_amp': iamp, 'amplitude': amp, 'pdgs': pdgs})

    # 4) double loop over the born legs to find amplitudes needed for the SSC 
    # part, proportional to I_pm, where
    # two particles are switched for their isospin partner(s), in the case
    # the charge is changed
    # Note that one has to loop on the base amplitude and on those with the
    # goldstones
    for iamp, base_amp in enumerate([born_amp] + goldstone_amplitudes):
        logger.info("Generating Sudakov amplitudes (SSC-c2) based on " + base_amp['process'].nice_string())
        for ileg1, leg1 in enumerate(base_amp['process']['legs']):
            iso_part_list1 = get_isospin_partners_diffcharge(leg1['id'], model)

            # skip if no partners exist
            if not iso_part_list1: continue

            for ileg2, leg2 in enumerate(base_amp['process']['legs']):
                if ileg1 >= ileg2: continue

                iso_part_list2 = get_isospin_partners_diffcharge(leg2['id'], model)

                # skip if no partners exist
                if not iso_part_list2: continue

                for part1 in iso_part_list1:
                    for part2 in iso_part_list2:
                        born_proc = copy.copy(base_amp['process'])
                        # copy the legs as a LegList (not FKSLegList) in order 
                        # not to have them re-ordered
                        born_proc['legs'] = MG.LegList(copy.deepcopy(base_amp['process']['legs']))
                        # replace leg1
                        newleg1 = copy.copy(leg1)
                        newleg1['id'] = part1
                        born_proc['legs'][ileg1] = newleg1
                        # replace leg2
                        newleg2 = copy.copy(leg2)
                        newleg2['id'] = part2
                        born_proc['legs'][ileg2] = newleg2
                        pdgs = [[leg1['id'],leg2['id']], [newleg1['id'],newleg2['id']]] # old and new pdgs
                        # check charge conservation
                        if not is_charge_conserved(born_proc): continue

                        amp = diagram_generation.Amplitude(born_proc)
                        # skip amplitudes without diagrams
                        if not amp['diagrams'] : continue

                        logger.info('   Found Sudakov amplitude (isospin diff-charge, I_pm x I_pm) for %s' % born_proc.nice_string())
                        amplitudes.append({'type': 'ipm2', 'legs': [leg1, leg2], 'base_amp': iamp, 'amplitude': amp, 'pdgs': pdgs})

    return amplitudes
