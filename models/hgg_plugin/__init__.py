
import particles
import couplings
import lorentz
import parameters
import vertices
import coupling_orders
import write_param_card


all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_orders = coupling_orders.all_orders
all_functions = function_library.all_functions

try:
   import decays
except ImportError:
   pass
else:
   all_decays = decays.all_decays


gauge = [0, 1]


__author__ = "N. Christensen, C. Duhr"
__date__ = "02. 06. 2009"
__version__= "1.3"
