# type: ignore
from __future__ import absolute_import
from . import particles
from . import couplings
from . import lorentz
from . import parameters
from . import vertices
from . import coupling_orders
from . import function_library
from . import object_library
# from . import write_param_card
try:
    from . import decays
except ImportError:
    pass
try:
    from . import build_restrict
except ImportError:
    pass

# model options
gauge = [0, 1]


all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_orders = coupling_orders.all_orders
all_functions = function_library.all_functions
# all_decays = decays.all_decays


__author__ = "N. Christensen, C. Duhr"
__version__ = "1.3"
__email__ = "neil@pa.msu.edu, claude.duhr@uclouvain.be"
