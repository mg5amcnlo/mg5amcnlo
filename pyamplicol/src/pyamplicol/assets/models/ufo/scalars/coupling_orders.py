# type: ignore
from __future__ import absolute_import
from .object_library import all_orders, CouplingOrder

# Keep the coupling order names QCD and QED as they are hard-coded standards in some generators
QCD = CouplingOrder(name='QCD',
                    expansion_order=99,
                    hierarchy=1)

QED = CouplingOrder(name='QED',
                    expansion_order=99,
                    hierarchy=2)
