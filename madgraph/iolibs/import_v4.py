##############################################################################
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
##############################################################################

"""Methods and classes to import v4 format model files."""

from madgraph.core.base_objects import Particle
from madgraph.core.base_objects import ParticleList

##############################################################################
##  read_particles_v4
##############################################################################

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

                except (Particle.ParticleError, ValueError), why:
                        print "Warning:", why, ", particle ignored"
                else:
                    mypartlist.append(mypart)

    return mypartlist
