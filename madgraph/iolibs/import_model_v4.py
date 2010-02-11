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

"""Methods and classes to import v4 format model files."""

import fractions
import logging

import madgraph.core.color_algebra as color
from madgraph.core.base_objects import Particle, ParticleList
from madgraph.core.base_objects import Interaction, InteractionList


#===============================================================================
# read_particles_v4
#===============================================================================
def read_particles_v4(fsock):
    """Read a list of particle from stream fsock, using the old v4 format"""

    spin_equiv = {'s': 1,
                  'f': 2,
                  'v': 3,
                  't': 5}

    color_equiv = {'s': 1,
                   't': 3,
                   'o': 8}

    line_equiv = {'d': 'dashed',
                  's': 'straight',
                  'w': 'wavy',
                  'c': 'curly'}


    mypartlist = ParticleList()

    for line in fsock:
        mypart = Particle()

        if line.find("MULTIPARTICLES") != -1:
            break # stop scanning if old MULTIPARTICLES tag found

        line = line.split("#", 2)[0] # remove any comment
        line = line.strip() # makes the string clean

        if line != "":
            values = line.split()
            if len(values) != 9:
                # Not the right number tags on the line
                raise ValueError, \
                    "Unvalid initialization string:" + line
            else:
                try:
                    mypart.set('name', values[0])
                    mypart.set('antiname', values[1])

                    if mypart['name'] == mypart['antiname']:
                        mypart['self_antipart'] = True

                    if values[2].lower() in spin_equiv.keys():
                        mypart.set('spin',
                                   spin_equiv[values[2].lower()])
                    else:
                        raise ValueError, "Unvalid spin %s" % \
                                values[2]

                    if values[3].lower() in line_equiv.keys():
                        mypart.set('line',
                                   line_equiv[values[3].lower()])
                    else:
                        raise ValueError, \
                                "Unvalid line type %s" % values[3]

                    mypart.set("mass", values[4])
                    mypart.set("width", values[5])

                    if values[6].lower() in color_equiv.keys():
                        mypart.set('color',
                                   color_equiv[values[6].lower()])
                    else:
                        raise ValueError, \
                            "Unvalid color rep %s" % values[6]

                    mypart.set("texname", values[7])
                    mypart.set("pdg_code", int(values[8]))

                    mypart.set('charge', 0.)
                    mypart.set('antitexname', mypart.get('texname'))

                except (Particle.PhysicsObjectError, ValueError), why:
                    logging.warning("Warning: %s, particle ignored" % why)
                else:
                    mypartlist.append(mypart)

    return mypartlist


#===============================================================================
# read_interactions_v4
#===============================================================================
def read_interactions_v4(fsock, ref_part_list):
    """Read a list of interactions from stream fsock, using the old v4 format.
    Requires a ParticleList object as an input to recognize particle names."""

    myinterlist = InteractionList()

    if not isinstance(ref_part_list, ParticleList):
        raise ValueError, \
            "Object %s is not a valid ParticleList" % repr(ref_part_list)

    for line in fsock:
        myinter = Interaction()

        line = line.split("#", 2)[0] # remove any comment
        line = line.strip() # makes the string clean

        if line != "": # skip blank
            values = line.split()
            part_list = ParticleList()

            try:
                for str_name in values:
                    curr_part = ref_part_list.find_name(str_name)
                    if isinstance(curr_part, Particle):
                        # Look at the total number of strings, stop if 
                        # anyway not enough, required if a variable name 
                        # corresponds to a particle! (eg G)
                        if len(values) >= 2 * len(part_list) + 1:
                            part_list.append(curr_part)
                        else: break
                    # also stops if string does not correspond to 
                    # a particle name
                    else: break

                if len(part_list) < 3:
                    raise Interaction.PhysicsObjectError, \
                        "Vertex with less than 3 known particles found."

                # Flip part/antipart of second part when needed 
                # according to v4 convention
                spin_array = [part['spin'] for part in part_list]
                if spin_array in [[2, 2, 1], # FFS
                                  [2, 2, 3]]:  # FFV
                    part_list[0]['is_part'] = False

                myinter.set('particles', part_list)

                # Give color structure
                # Order particles according to color
                # Don't consider singlets
                color_parts = sorted(part_list, lambda p1, p2:\
                                            p1.get_color() - p2.get_color())
                color_parts = filter(lambda p: p.get_color() != 1, color_parts)
                colors = [part.get_color() for part in color_parts]

                # Set color empty by default
                myinter.set('color', [])
                if not colors:
                    # All color singlets - set empty
                    pass
                elif colors == [-3, 3]:
                    # triplet-triplet-singlet coupling
                    myinter.set('color', [color.ColorString(\
                        [color.T(part_list.index(color_parts[1]),
                                 part_list.index(color_parts[0]))])])
                elif colors == [8, 8]:
                    # octet-octet-singlet coupling
                    my_cs = color.ColorString(\
                        [color.Tr(part_list.index(color_parts[0]),
                                 part_list.index(color_parts[1]))])
                    my_cs.coeff = fractions.Fraction(2)
                    myinter.set('color', [my_cs])
                elif colors == [-3, 3, 8]:
                    # triplet-triplet-octet coupling
                    myinter.set('color', [color.ColorString(\
                        [color.T(part_list.index(color_parts[2]),
                                 part_list.index(color_parts[1]),
                                 part_list.index(color_parts[0]))])])
                elif colors == [8, 8, 8]:
                    # Triple glue coupling
                    my_color_string = color.ColorString(\
                        [color.f(0, 1, 2)])
                    my_color_string.is_imaginary = True
                    myinter.set('color', [my_color_string])
                elif colors == [-3, 3, 8, 8]:
                    my_cs = color.ColorString(\
                        [color.f(part_list.index(color_parts[2]),
                                 part_list.index(color_parts[3]),
                                 - 1),
                         color.T(-1,
                                 part_list.index(color_parts[1]),
                                 part_list.index(color_parts[0]))])
                    my_cs.is_imaginary = True
                    myinter.set('color', [my_cs])
                elif colors == [8, 8, 8, 8]:
                    # 4-glue / glue-glue-gluino-gluino coupling
                    cs1 = color.ColorString([color.f(0, 1, -1),
                                                   color.f(2, 3, -1)])
                    #cs1.coeff = fractions.Fraction(-1)
                    cs2 = color.ColorString([color.f(2, 0, -1),
                                                   color.f(1, 3, -1)])
                    #cs2.coeff = fractions.Fraction(-1)
                    cs3 = color.ColorString([color.f(1, 2, -1),
                                                   color.f(0, 3, -1)])
                    #cs3.coeff = fractions.Fraction(-1)
                    myinter.set('color', [cs1, cs2, cs3])
                else:
                    logging.warning(\
                        "Color combination %s not yet implemented." % \
                        repr(colors))

                # REMEMBER to set the 4g structure once we have
                # decided how to deal with this vertex

                # Set the Lorentz structure. Default for 3-particle
                # vertices is empty string, for 4-particle pair of
                # empty strings
                myinter.set('lorentz', [''])

                pdg_codes = sorted([part.get_pdg_code() for part in part_list])

                # WWWW and WWVV
                if pdg_codes == [-24, -24, 24, 24]:
                    myinter.set('lorentz', ['WWWW'])
                elif spin_array == [3, 3, 3, 3] and \
                             24 in pdg_codes and - 24 in pdg_codes:
                    myinter.set('lorentz', ['WWVV'])

                # gggg
                if pdg_codes == [21, 21, 21, 21]:
                    myinter.set('lorentz', ['gggg1', 'gggg2', 'gggg3'])

                # If extra flag, add this to Lorentz    
                if len(values) > 3 * len(part_list) - 4:
                    myinter.get('lorentz')[0] = \
                                      myinter.get('lorentz')[0]\
                                      + values[3 * len(part_list) - 4].upper()

                # Use the other strings to fill variable names and tags

                # Couplings: special treatment for 4-vertices, where MG4 used
                # two couplings, while MG5 only uses one (with the exception
                # of the 4g vertex, which needs special treatment)
                # DUM0 and DUM1 are used as placeholders by FR, corresponds to 1
                if len(part_list) == 3 or \
                   values[len(part_list) + 1] in ['DUM', 'DUM0', 'DUM1']:
                    # We can just use the first coupling, since the second
                    # is a dummy
                    myinter.set('couplings', {(0, 0):values[len(part_list)]})
                    if myinter.get('lorentz')[0] == 'WWWWN':
                        # Should only use one Helas amplitude for electroweak
                        # 4-vector vertices with FR. I choose W3W3NX.
                        myinter.set('lorentz', ['WWVVN'])
                elif pdg_codes == [21, 21, 21, 21]:
                    # gggg
                    myinter.set('couplings', {(0, 0):values[len(part_list)],
                                              (1, 1):values[len(part_list)],
                                              (2, 2):values[len(part_list)]})
                elif myinter.get('lorentz')[0] == 'WWWW':
                    # Need special treatment of v4 SM WWWW couplings since 
                    # MG5 can only have one coupling per Lorentz structure
                    myinter.set('couplings', {(0, 0):\
                                              'sqrt(' +
                                              values[len(part_list)] + \
                                             '**2+' + \
                                              values[len(part_list) + 1] + \
                                              '**2)'})
                elif myinter.get('lorentz')[0] == 'WWVV':
                    # Need special treatment of v4 SM WWVV couplings since 
                    # MG5 can only have one coupling per Lorentz structure
                    myinter.set('couplings', {(0, 0):values[len(part_list)] + \
                                             '*' + \
                                              values[len(part_list) + 1]})
                    #raise Interaction.PhysicsObjectError, \
                    #    "Only FR-style 4-vertices implemented."

                # Coupling orders - needs to be fixed
                order_list = values[2 * len(part_list) - 2: \
                                    3 * len(part_list) - 4]

                def count_duplicates_in_list(dupedList):
                    uniqueSet = set(item for item in dupedList)
                    ret_dict = {}
                    for item in uniqueSet:
                        ret_dict[item] = dupedList.count(item)
                    return ret_dict

                myinter.set('orders', count_duplicates_in_list(order_list))

                myinter.set('id', len(myinterlist) + 1)

                myinterlist.append(myinter)

            except Interaction.PhysicsObjectError, why:
                logging.error("Interaction ignored: %s" % why)

    return myinterlist
