# This file was automatically created by FeynRules 2.0.6
# Mathematica version: 8.0 for Mac OS X x86 (64-bit) (February 23, 2011)
# Date: Wed 11 Dec 2013 19:27:13


from object_library import all_orders, CouplingOrder


QCD = CouplingOrder(name = 'QCD',
                    expansion_order = 99,
                    perturbative_expansion =1,
                    hierarchy = 1)

QED = CouplingOrder(name = 'QED',
                    expansion_order = 99,
                    perturbative_expansion =1,
                    hierarchy = 2)

QNP = CouplingOrder(name = 'QNP',
                    expansion_order = 2,
                    perturbative_expansion =1,
                    hierarchy = 2)

