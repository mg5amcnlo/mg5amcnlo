################################################################################
#
# Copyright (c) 2012 The MadGraph5_aMC@NLO Development team and Contributors
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
""" This part is not part of the UFO Model but only of MG5 suite. 
This files defines how the restrict card can be build automatically """ 

import models.build_restriction_lib as build_restrict_lib
all_categories = []


first_category = build_restrict_lib.Category('sm customization')
all_categories.append(first_category)

first_category.add_options(name='light mass = 0 (e mu)', # name
                           default=True,        # default
                           rules=[('MASS',[11], 0.0),                 
                                  ('MASS',[13], 0.0)]
                           )

first_category.add_options(name='b mass = 0',
                           default=False,
                           rules=[('MASS',[5], 0.0)]
                           )

first_category.add_options(name='tau mass = 0',
                           default=False,
                           rules=[('MASS',[15], 0.0)]
                           )

sec_category = build_restrict_lib.Category('mssm customization')
all_categories.append(sec_category)

sec_category.add_options(name='diagonal usqmix (stop) matrix',
                           default=False,        # default
                           rules=[('USQMIX',[3,3], 1.0),
                                  ('USQMIX',[6,6], 1.0),
                                  ('USQMIX',[3,6], 0.0),
                                  ('USQMIX',[6,3], 0.0)]
                           )

sec_category.add_options(name='diagonal dsqmix (sbottom) matrix',
                           default=False,        # default
                           rules=[('DSQMIX',[3,3], 1.0),
                                  ('DSQMIX',[6,6], 1.0),
                                  ('DSQMIX',[3,6], 0.0),
                                  ('DSQMIX',[6,3], 0.0)]
                           )

sec_category.add_options(name='diagonal selmix (stau) matrix',
                           default=False,        # default
                           rules=[('SELMIX',[3,3], 1.0),
                                  ('SELMIX',[6,6], 1.0),
                                  ('SELMIX',[3,6], 0.0),
                                  ('SELMIX',[6,3], 0.0)]
                           )
