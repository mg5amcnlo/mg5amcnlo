
import particles
import couplings
import lorentz
import parameters
import vertices
import coupling_orders
import write_param_card
import propagators


all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_orders = coupling_orders.all_orders
all_functions = function_library.all_functions
all_propagators = propagators.all_propagators

try:
     import build_restrict
except ImportError:
     pass


try:
   import form_factors
except ImportError:
   pass
else:
   all_form_factors = form_factors.all_form_factors

try:
   import CT_vertices
except ImportError:
   pass
else:
   all_CTvertices = CT_vertices.all_CTvertices


gauge = [0]


__author__ = "C. Degrande"
__date__ = "08.04. 2013"
__version__= "1.1"
