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

### Important Warning ###
# When you develop such file. Please cross check that they are NO
# unwanted simplification to your model. This can happen especially
# if a bunch of value are set to 1.0 by default. This YOUR responsability
# to check that you get the expected behavior                            
### Important Warning ###

import models.build_restriction_lib as build_restrict_lib
all_categories = []


first_category = build_restrict_lib.Category('sm customization')
all_categories.append(first_category)


first_category.add_options(name='diagonal ckm', # name
                           default=True,          # default
                           rules=[('WOLFENSTEIN',[1], 0.0),
                                  ('WOLFENSTEIN',[2], 0.0),
                                  ('WOLFENSTEIN',[3], 0.0),
                                  ('WOLFENSTEIN',[4], 0.0)]
                           )

first_category.add_options(name='c mass = 0', # name
                           default=True,        # default
                           rules=[('MASS',[4], 0.0),
                                  ('YUKAWA',[4], 0.0)]
                           )

first_category.add_options(name='b mass = 0',
                           default=False,
                           rules=[('MASS',[5], 0.0),
                                  ('YUKAWA',[5], 0.0)]
                           )

first_category.add_options(name='tau mass = 0',
                           default=False,
                           rules=[('MASS',[15], 0.0),
                                  ('YUKAWA',[15], 0.0)]
                           )

first_category.add_options(name='muon mass = 0',
                           default=True,
                           rules=[('MASS',[13], 0.0),
                                  ('YUKAWA',[13], 0.0)]
                           )

#first_category.add_options(name='electron mass = 0',
#                           default=True,
#                           rules=[('MASS',[11], 0.0),
#                                  ('YUKAWA',[11], 0.0)]
#                           )

dim6_category = build_restrict_lib.Category('Adding Dim6 Operator')
all_categories.append(dim6_category)

dim6_category.add_options(name='CWWW',
                          default=False,
                          inverted_display=True,
                          rules=[('DIM6',[1], 0.0)],
                          )

dim6_category.add_options(name='CW',
                          default=False,
                          inverted_display=True,
                          rules=[('DIM6',[2], 0.0)],
                          )

dim6_category.add_options(name='CB',
                          default=False,
                          rules=[('DIM6',[3], 0.0)],
                          inverted_display=True
                          )


dim6_category.add_options(name='CWWW CP violating',
                          default=True,
                          rules=[('DIM6',[4], 0.0)],
                          inverted_display=True
                          )

dim6_category.add_options(name='CW CP violating',
                          default=True,
                          rules=[('DIM6',[5], 0.0)],
                          inverted_display=True
                          )
dim6_category.add_options(name='Cphid',
                          default=True,
                          rules=[('DIM6',[6], 0.0)],
                          inverted_display=True
                          )
dim6_category.add_options(name='CphiW',
                          default=True,
                          rules=[('DIM6',[7], 0.0)],
                          inverted_display=True
                          )
dim6_category.add_options(name='CphiB',
                          default=True,
                          rules=[('DIM6',[8], 0.0)],
                          inverted_display=True
                          )


