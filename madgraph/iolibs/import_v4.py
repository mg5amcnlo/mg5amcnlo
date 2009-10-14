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

import logging

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
                raise self.ValueError, \
                    "Unvalid initialization string:" + line
            else:
                try:
                    mypart.set('name', values[0])
                    mypart.set('antiname', values[1])

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
                            # Comment by Johan: Note that here (or
                            # slightly later) we need to add a kinda
                            # elaborate check to harmonize the old MG
                            # standard for interaction, where FFS/FFV
                            # and VVV/SSS/etc interactions are treated
                            # differently in terms of
                            # particles/antiparticles. Alternatively
                            # (which I like less, but is also
                            # possible) we need to keep track of this
                            # during generation of the dictionnaries
                        else: break
                    # also stops if string does not correspond to 
                    # a particle name
                    else: break

                if len(part_list) < 3:
                    raise Interaction.PhysicsObjectError, \
                        "Vertex with less than 3 known particles found."

                myinter.set('particles', part_list)

                # Give a dummy 'guess' values for color and Lorentz structures
                # Those will have to be replaced by a proper guess!

                myinter.set('color', ['guess'])
                myinter.set('lorentz', ['guess'])

                # Use the other strings to fill variable names and tags
                myinter.set('couplings', {(0, 0):values[len(part_list)]})

                order_list = values[2 * len(part_list) - 2: \
                                   3 * len(part_list) - 4]

                def count_duplicates_in_list(dupedList):
                    uniqueSet = set(item for item in dupedList)
                    ret_dict = {}
                    for item in uniqueSet:
                        ret_dict[item] = dupedList.count(item)
                    return ret_dict

                myinter.set('orders', count_duplicates_in_list(order_list))

                myinterlist.append(myinter)

            except Interaction.PhysicsObjectError, why:
                logging.warning("Interaction ignored: %s" % why)

    return myinterlist
