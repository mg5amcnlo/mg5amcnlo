# This file was automatically created by FeynRules 2.4.54
# Mathematica version: 11.0.0 for Linux x86 (64-bit) (July 28, 2016)
# Date: Tue 25 Oct 2016 14:05:35


from object_library import all_orders, CouplingOrder


QCD = CouplingOrder(name = 'QCD',
                    expansion_order = 99,
                    hierarchy = 1,
                    perturbative_expansion = 1)

QED = CouplingOrder(name = 'QED',
                    expansion_order = 99,
                    hierarchy = 2)

GGG = CouplingOrder(name = 'GGG',
                    expansion_order = 99,
                    hierarchy = 1,
                    perturbative_expansion = 1)

DGDG = CouplingOrder(name = 'DGDG',
                     expansion_order = 99,
                     hierarchy = 1,
                    perturbative_expansion = 1)

