# This file was automatically created by FeynRules $Revision: 623 $
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (November 6, 2010)
# Date: Thu 9 Jun 2011 17:49:34


from __future__ import absolute_import
from .object_library import all_orders, CouplingOrder

QCD = CouplingOrder(name = 'QCD',
                    hierarchy = 1,
                    expansion_order = -1,
                    perturbative_expansion = 1)

QED = CouplingOrder(name = 'QED',
                    hierarchy = 2,
                    expansion_order = -1,
                    perturbative_expansion =0)
